"""
Step 4 & 5: Modélisation et Évaluation
- Modèles: Baseline, Logistic, Cox PH, Gradient Boosting
- Métriques business: top-k capture, lift
- Comparaison par horizon
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from lifelines import CoxPHFitter, WeibullAFTFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False

from config import (
    ARTIFACTS_DIR, REPORTS_DIR, HORIZONS, RANDOM_SEED, TOP_K_PERCENTAGES
)
from dataset_builder import get_feature_columns


def load_dataset(horizon: int) -> pd.DataFrame:
    """Charge le dataset pour un horizon donné."""
    return pd.read_pickle(ARTIFACTS_DIR / f"dataset_freeze_h{horizon}.pkl")


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prépare les features pour la modélisation.
    Retourne X, y et les encoders.
    """
    num_feats, cat_feats = get_feature_columns()

    # Vérifier quelles features existent
    num_feats = [f for f in num_feats if f in df.columns]
    cat_feats = [f for f in cat_feats if f in df.columns]

    # Copie pour éviter les warnings
    df_prep = df.copy()

    # Encoder les catégorielles
    encoders = {}
    for col in cat_feats:
        le = LabelEncoder()
        df_prep[col + '_encoded'] = le.fit_transform(df_prep[col].astype(str))
        encoders[col] = le

    # Construire X
    feature_cols = num_feats + [f + '_encoded' for f in cat_feats]
    X = df_prep[feature_cols].copy()

    # Remplacer les valeurs infinies et NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    y = df_prep['event'].values

    return X, y, encoders, feature_cols


def compute_business_metrics(y_true: np.ndarray, y_scores: np.ndarray,
                             top_k_pcts: list = TOP_K_PERCENTAGES) -> dict:
    """
    Calcule les métriques business: capture@k et lift@k.

    Capture@k = % des événements réels dans le top-k% des scores
    Lift@k = Capture@k / k
    """
    n = len(y_true)
    n_events = y_true.sum()

    if n_events == 0:
        return {f'capture_{int(k*100)}pct': 0.0 for k in top_k_pcts}

    # Trier par score décroissant
    sorted_idx = np.argsort(-y_scores)
    y_sorted = y_true[sorted_idx]

    metrics = {}
    for k_pct in top_k_pcts:
        k = int(np.ceil(n * k_pct))
        events_in_top_k = y_sorted[:k].sum()
        capture = events_in_top_k / n_events
        lift = capture / k_pct

        pct_label = int(k_pct * 100)
        metrics[f'capture_{pct_label}pct'] = capture
        metrics[f'lift_{pct_label}pct'] = lift
        metrics[f'n_top_{pct_label}pct'] = k
        metrics[f'events_in_top_{pct_label}pct'] = events_in_top_k

    return metrics


def compute_stability_metrics(y_true: np.ndarray, y_scores: np.ndarray,
                              df: pd.DataFrame, group_col: str,
                              top_k_pct: float = 0.10) -> dict:
    """
    Calcule les métriques de stabilité par groupe (matériau, décennie, etc.)
    """
    results = {}
    n = len(y_true)
    k = int(np.ceil(n * top_k_pct))

    # Top-k global
    sorted_idx = np.argsort(-y_scores)
    in_top_k = np.zeros(n, dtype=bool)
    in_top_k[sorted_idx[:k]] = True

    for group in df[group_col].unique():
        mask = df[group_col].values == group
        n_group = mask.sum()
        if n_group < 50:
            continue

        n_events_group = y_true[mask].sum()
        if n_events_group == 0:
            continue

        # Événements du groupe qui sont dans le top-k global
        events_captured = (y_true[mask] & in_top_k[mask]).sum()
        capture = events_captured / n_events_group

        results[str(group)] = {
            'n_total': int(n_group),
            'n_events': int(n_events_group),
            'events_captured': int(events_captured),
            f'capture_{int(top_k_pct*100)}pct': float(capture)
        }

    return results


class BaselineScorer:
    """
    Modèle baseline basé sur des règles métier simples.
    Score = combinaison de l'âge, du nombre de fuites et du ratio vs médiane.
    """

    def __init__(self):
        self.name = "Baseline (Rule-based)"

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        """
        Score basé sur:
        - age_at_freeze (normalisé)
        - n_fuites_total
        - ratio_age_median
        """
        scores = np.zeros(len(X))

        if 'age_at_freeze' in X.columns:
            age_norm = X['age_at_freeze'] / X['age_at_freeze'].max()
            scores += 0.3 * age_norm

        if 'n_fuites_total' in X.columns:
            fuites_norm = X['n_fuites_total'] / (X['n_fuites_total'].max() + 1)
            scores += 0.4 * fuites_norm

        if 'ratio_age_median' in X.columns:
            ratio_norm = X['ratio_age_median'] / (X['ratio_age_median'].max() + 1)
            scores += 0.3 * ratio_norm

        # Normaliser entre 0 et 1
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

        return np.column_stack([1 - scores, scores])


class CoxPHWrapper:
    """
    Wrapper pour le modèle Cox PH de lifelines.
    """

    def __init__(self):
        self.name = "Cox Proportional Hazards"
        self.model = None
        self.feature_cols = None

    def fit(self, X, y, duration=None):
        if not HAS_LIFELINES:
            raise ImportError("lifelines n'est pas installé")

        df = X.copy()
        df['event'] = y
        df['duration'] = duration if duration is not None else 1

        # Sélectionner uniquement les features numériques
        self.feature_cols = [c for c in X.columns if X[c].dtype in ['int64', 'float64']]

        df_cox = df[self.feature_cols + ['event', 'duration']].copy()

        # Standardiser les features
        self.scaler = StandardScaler()
        df_cox[self.feature_cols] = self.scaler.fit_transform(df_cox[self.feature_cols])

        self.model = CoxPHFitter(penalizer=0.1)
        self.model.fit(df_cox, duration_col='duration', event_col='event')

        return self

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not fitted")

        df = X[self.feature_cols].copy()
        df[self.feature_cols] = self.scaler.transform(df[self.feature_cols])

        # Le risque partiel (hazard) comme score
        risk_scores = self.model.predict_partial_hazard(df).values

        # Normaliser entre 0 et 1
        risk_scores = (risk_scores - risk_scores.min()) / (risk_scores.max() - risk_scores.min() + 1e-10)

        return np.column_stack([1 - risk_scores, risk_scores])


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test,
                             df_test, duration_train=None, duration_test=None):
    """
    Entraîne un modèle et calcule toutes les métriques.
    """
    # Entraînement
    if hasattr(model, 'fit') and 'duration' in model.fit.__code__.co_varnames:
        model.fit(X_train, y_train, duration=duration_train)
    else:
        model.fit(X_train, y_train)

    # Prédictions
    if hasattr(model, 'predict_proba'):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.predict(X_test)

    # Métriques classiques
    results = {
        'model': model.name if hasattr(model, 'name') else type(model).__name__,
        'roc_auc': roc_auc_score(y_test, y_scores) if y_test.sum() > 0 else 0,
        'pr_auc': average_precision_score(y_test, y_scores) if y_test.sum() > 0 else 0,
        'brier_score': brier_score_loss(y_test, y_scores)
    }

    # Métriques business
    business_metrics = compute_business_metrics(y_test, y_scores)
    results.update(business_metrics)

    # Métriques de stabilité par matériau
    if 'materiau' in df_test.columns:
        stability_mat = compute_stability_metrics(y_test, y_scores, df_test, 'materiau')
        results['stability_materiau'] = stability_mat

    # Métriques de stabilité par décennie
    if 'decade_install' in df_test.columns:
        stability_decade = compute_stability_metrics(y_test, y_scores, df_test, 'decade_install')
        results['stability_decade'] = stability_decade

    return results, y_scores


def run_experiment(horizon: int) -> dict:
    """
    Exécute l'expérience complète pour un horizon donné.
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENTATION - HORIZON {horizon} AN(S)")
    print(f"{'='*60}")

    # Chargement des données
    df = load_dataset(horizon)
    print(f"Dataset: {len(df):,} troncons, {df['event'].sum():,} evenements ({100*df['event'].mean():.2f}%)")

    # Préparation des features
    X, y, encoders, feature_cols = prepare_features(df)
    print(f"Features: {len(feature_cols)}")

    # Split train/test (stratifié)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    df_test = df.loc[idx_test].copy()
    df_train = df.loc[idx_train].copy()

    print(f"Train: {len(X_train):,} ({y_train.sum():,} evenements)")
    print(f"Test: {len(X_test):,} ({y_test.sum():,} evenements)")

    # Durées pour les modèles de survie
    duration_train = df_train['duration_days'].values
    duration_test = df_test['duration_days'].values

    # Liste des modèles à tester
    models = []

    # 1. Baseline
    models.append(BaselineScorer())

    # 2. Logistic Regression
    lr = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        solver='lbfgs',
        C=1.0
    )
    lr.name = "Logistic Regression"
    models.append(lr)

    # 3. Cox PH (si disponible)
    if HAS_LIFELINES:
        try:
            cox = CoxPHWrapper()
            models.append(cox)
        except Exception as e:
            print(f"Cox PH non disponible: {e}")

    # 4. Gradient Boosting (sklearn)
    gb = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_SEED,
        class_weight='balanced'
    )
    gb.name = "HistGradientBoosting"
    models.append(gb)

    # 5. LightGBM (si disponible)
    if HAS_LIGHTGBM:
        # Calculer le ratio pour scale_pos_weight
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos = n_neg / n_pos if n_pos > 0 else 1

        lgbm = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            scale_pos_weight=scale_pos,
            random_state=RANDOM_SEED,
            verbose=-1
        )
        lgbm.name = "LightGBM"
        models.append(lgbm)

    # Entraînement et évaluation
    results = []
    best_model = None
    best_lift = 0
    best_scores = None

    for model in models:
        print(f"\nEntrainement: {model.name if hasattr(model, 'name') else type(model).__name__}...")
        try:
            res, y_scores = train_and_evaluate_model(
                model, X_train, y_train, X_test, y_test,
                df_test, duration_train, duration_test
            )
            results.append(res)

            # Tracker le meilleur modèle (basé sur lift@10%)
            if res['lift_10pct'] > best_lift:
                best_lift = res['lift_10pct']
                best_model = model
                best_scores = y_scores

            print(f"  ROC-AUC: {res['roc_auc']:.3f}")
            print(f"  Capture@5%: {res['capture_5pct']:.1%}, Lift: {res['lift_5pct']:.2f}")
            print(f"  Capture@10%: {res['capture_10pct']:.1%}, Lift: {res['lift_10pct']:.2f}")
            print(f"  Capture@20%: {res['capture_20pct']:.1%}, Lift: {res['lift_20pct']:.2f}")

        except Exception as e:
            print(f"  ERREUR: {e}")

    return {
        'horizon': horizon,
        'results': results,
        'best_model': best_model,
        'best_scores': best_scores,
        'encoders': encoders,
        'feature_cols': feature_cols,
        'X_test': X_test,
        'y_test': y_test,
        'df_test': df_test,
        'X_train': X_train,
        'y_train': y_train
    }


def create_benchmark_table(all_results: dict) -> pd.DataFrame:
    """
    Crée le tableau comparatif de tous les modèles et horizons.
    """
    rows = []
    for horizon, exp in all_results.items():
        for res in exp['results']:
            row = {
                'Horizon': f"{horizon} an(s)",
                'Model': res['model'],
                'ROC-AUC': res['roc_auc'],
                'PR-AUC': res['pr_auc'],
                'Capture@5%': res['capture_5pct'],
                'Lift@5%': res['lift_5pct'],
                'Capture@10%': res['capture_10pct'],
                'Lift@10%': res['lift_10pct'],
                'Capture@20%': res['capture_20pct'],
                'Lift@20%': res['lift_20pct']
            }
            rows.append(row)

    return pd.DataFrame(rows)


def save_best_model(exp: dict, horizon: int):
    """
    Sauvegarde le meilleur modèle et ses métadonnées.
    """
    model = exp['best_model']
    encoders = exp['encoders']
    feature_cols = exp['feature_cols']

    # Métadonnées
    metadata = {
        'horizon_years': horizon,
        'freeze_date': str(exp['df_test']['freeze_date'].iloc[0]),
        'model_type': model.name if hasattr(model, 'name') else type(model).__name__,
        'feature_cols': feature_cols,
        'n_train': len(exp['X_train']),
        'n_test': len(exp['X_test']),
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Calcul des métriques finales
    y_scores = exp['best_scores']
    y_test = exp['y_test']
    metrics = compute_business_metrics(y_test, y_scores)
    metadata['test_metrics'] = metrics

    # Sauvegarde
    artifact = {
        'model': model,
        'encoders': encoders,
        'metadata': metadata
    }

    output_path = ARTIFACTS_DIR / f"model_h{horizon}.joblib"
    joblib.dump(artifact, output_path)
    print(f"\nModele sauvegarde: {output_path}")

    # Sauvegarder aussi les métadonnées en JSON
    with open(ARTIFACTS_DIR / f"model_h{horizon}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    return output_path


def main():
    """
    Exécute l'ensemble des expériences.
    """
    print("=" * 60)
    print("STEP 4 & 5: MODELISATION ET EVALUATION")
    print("=" * 60)

    all_results = {}

    for horizon in HORIZONS:
        exp = run_experiment(horizon)
        all_results[horizon] = exp
        save_best_model(exp, horizon)

    # Tableau comparatif
    print("\n" + "=" * 60)
    print("TABLEAU COMPARATIF")
    print("=" * 60)

    benchmark = create_benchmark_table(all_results)
    print("\n" + benchmark.to_string(index=False))

    # Sauvegarder le benchmark
    benchmark.to_csv(REPORTS_DIR / "benchmark_results.csv", index=False)

    # Identifier le meilleur horizon/modèle
    best_row = benchmark.loc[benchmark['Lift@10%'].idxmax()]
    print(f"\n\nMEILLEUR MODELE:")
    print(f"  Horizon: {best_row['Horizon']}")
    print(f"  Modele: {best_row['Model']}")
    print(f"  Lift@10%: {best_row['Lift@10%']:.2f}")
    print(f"  Capture@10%: {best_row['Capture@10%']:.1%}")

    # Sauvegarder le meilleur modèle comme modèle final
    best_horizon = int(best_row['Horizon'].split()[0])
    best_exp = all_results[best_horizon]

    # Copier comme best_model
    best_artifact = joblib.load(ARTIFACTS_DIR / f"model_h{best_horizon}.joblib")
    joblib.dump(best_artifact, ARTIFACTS_DIR / "best_model.joblib")

    print("\n" + "=" * 60)
    print("MODELISATION TERMINEE")
    print("=" * 60)

    return all_results, benchmark


if __name__ == "__main__":
    main()
