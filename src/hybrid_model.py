"""
Modèle Hybride - Combinaison HistGradientBoosting + Random Survival Forest
==========================================================================

Combine les forces des deux approches :
- HistGB : Bon ranking court terme (Lift@10% = 6.57)
- RSF : Meilleure discrimination (C-index = 0.95) + temps de survie
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

ARTIFACTS_DIR = Path("artifacts")


class HybridSurvivalModel:
    """
    Modèle hybride combinant classification et analyse de survie.

    Sorties :
    - score_combined : Score de risque combiné (0-1)
    - score_histgb : Probabilité de défaillance à 1 an
    - score_rsf : Score de risque RSF normalisé
    - median_survival : Temps médian avant défaillance (années)
    - survival_proba : Dict avec P(survie) à 1, 3, 5, 10 ans
    """

    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: Poids du modèle HistGB (0-1).
                   alpha=1 → 100% HistGB, alpha=0 → 100% RSF
        """
        self.alpha = alpha
        self.histgb = None
        self.rsf = None
        self.scaler_rsf = MinMaxScaler()
        self.label_encoders = {}
        self.features_histgb = None
        self.features_rsf = None
        self.is_fitted = False

    def _prepare_features_histgb(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prépare les features pour HistGradientBoosting."""
        features = []

        # Numériques
        for col in ['DIAMETRE', 'LNG', 'age', 'n_anomalies']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
                features.append(col)

        # Dérivées
        if 'age' in df.columns:
            df['age_years'] = df['age'] / 365.25
            features.append('age_years')

        if 'n_anomalies' in df.columns and 'LNG' in df.columns:
            df['anomaly_rate'] = df['n_anomalies'] / (df['LNG'] / 1000 + 0.01)
            features.append('anomaly_rate')

        # Catégorielles encodées
        for col in ['MAT', 'decade_install']:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_enc'] = self.label_encoders[col].fit_transform(
                        df[col].fillna('INCONNU').astype(str)
                    )
                else:
                    # Handle unseen labels
                    known = set(self.label_encoders[col].classes_)
                    df[col] = df[col].fillna('INCONNU').astype(str)
                    df[col] = df[col].apply(lambda x: x if x in known else 'INCONNU')
                    df[f'{col}_enc'] = self.label_encoders[col].transform(df[col])
                features.append(f'{col}_enc')

        self.features_histgb = features
        return df[features].copy()

    def _prepare_features_rsf(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prépare les features pour RSF."""
        features = []

        for col in ['DIAMETRE', 'LNG', 'age', 'n_anomalies']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
                features.append(col)

        if 'age' in df.columns:
            df['age_years'] = df['age'] / 365.25
            df['age_x_diam'] = df['age_years'] * df['DIAMETRE']
            features.extend(['age_years', 'age_x_diam'])

        if 'n_anomalies' in df.columns and 'LNG' in df.columns:
            df['anomaly_rate'] = df['n_anomalies'] / (df['LNG'] / 1000 + 0.01)
            features.append('anomaly_rate')

        for col in ['MAT', 'decade_install']:
            if col in df.columns:
                enc_col = f'{col}_enc'
                if enc_col in df.columns:
                    features.append(enc_col)

        self.features_rsf = features
        X = df[features].copy()

        # Target pour RSF
        y = np.array(
            [(bool(e), d) for e, d in zip(df['event'], df['duration_years'])],
            dtype=[('event', bool), ('duration', float)]
        )

        return X, y

    def fit(self, df: pd.DataFrame, verbose: bool = True) -> 'HybridSurvivalModel':
        """
        Entraîne les deux modèles.

        Args:
            df: DataFrame avec colonnes event, duration_years, et features
            verbose: Afficher la progression
        """
        if verbose:
            print("=" * 60)
            print("ENTRAINEMENT MODELE HYBRIDE")
            print("=" * 60)
            print(f"\nDonnées: {len(df):,} tronçons, {df['event'].sum():,} événements")

        # Préparer features HistGB
        if verbose:
            print("\n1. Préparation des features...")
        X_histgb = self._prepare_features_histgb(df.copy())
        y_histgb = df['event'].values

        # Préparer features RSF
        X_rsf, y_rsf = self._prepare_features_rsf(df.copy())

        # Split
        (X_histgb_train, X_histgb_test,
         y_histgb_train, y_histgb_test,
         X_rsf_train, X_rsf_test,
         y_rsf_train, y_rsf_test) = train_test_split(
            X_histgb, y_histgb, X_rsf, y_rsf,
            test_size=0.2, random_state=42
        )

        if verbose:
            print(f"   Train: {len(X_histgb_train):,}, Test: {len(X_histgb_test):,}")

        # Entraîner HistGB
        if verbose:
            print("\n2. Entraînement HistGradientBoosting...")
        self.histgb = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_depth=8,
            min_samples_leaf=50,
            random_state=42
        )
        self.histgb.fit(X_histgb_train, y_histgb_train)

        # Évaluer HistGB
        proba_histgb = self.histgb.predict_proba(X_histgb_test)[:, 1]
        from sklearn.metrics import roc_auc_score
        auc_histgb = roc_auc_score(y_histgb_test, proba_histgb)
        if verbose:
            print(f"   AUC-ROC: {auc_histgb:.4f}")

        # Entraîner RSF
        if verbose:
            print("\n3. Entraînement Random Survival Forest...")
        self.rsf = RandomSurvivalForest(
            n_estimators=100,
            max_depth=8,
            min_samples_split=30,
            min_samples_leaf=15,
            n_jobs=-1,
            random_state=42
        )
        self.rsf.fit(X_rsf_train, y_rsf_train)

        # Évaluer RSF
        risk_rsf = self.rsf.predict(X_rsf_test)
        c_index = concordance_index_censored(
            y_rsf_test['event'], y_rsf_test['duration'], risk_rsf
        )[0]
        if verbose:
            print(f"   C-index: {c_index:.4f}")

        # Fit scaler pour normaliser les scores RSF
        risk_rsf_train = self.rsf.predict(X_rsf_train)
        self.scaler_rsf.fit(risk_rsf_train.reshape(-1, 1))

        # Évaluer le modèle combiné
        if verbose:
            print("\n4. Évaluation du modèle combiné...")

        score_histgb_test = proba_histgb
        score_rsf_test = self.scaler_rsf.transform(risk_rsf.reshape(-1, 1)).flatten()
        score_combined = self.alpha * score_histgb_test + (1 - self.alpha) * score_rsf_test

        # C-index du score combiné
        c_index_combined = concordance_index_censored(
            y_rsf_test['event'], y_rsf_test['duration'], score_combined
        )[0]

        if verbose:
            print(f"   C-index combiné (alpha={self.alpha}): {c_index_combined:.4f}")
            print(f"\n   Comparaison:")
            print(f"   - HistGB seul:  AUC={auc_histgb:.4f}")
            print(f"   - RSF seul:     C-index={c_index:.4f}")
            print(f"   - Combiné:      C-index={c_index_combined:.4f}")

        self.is_fitted = True
        self.metrics = {
            'auc_histgb': auc_histgb,
            'c_index_rsf': c_index,
            'c_index_combined': c_index_combined,
            'alpha': self.alpha
        }

        if verbose:
            print("\n" + "=" * 60)
            print("ENTRAINEMENT TERMINE")
            print("=" * 60)

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prédit les scores et temps de survie.

        Returns:
            DataFrame avec colonnes:
            - score_combined: Score de risque combiné (0-1)
            - score_histgb: Proba défaillance 1 an
            - score_rsf: Score RSF normalisé
            - median_survival: Temps médian (années)
            - surv_1y, surv_3y, surv_5y, surv_10y: Probabilités de survie
        """
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas entraîné. Appelez fit() d'abord.")

        # Préparer features
        df_copy = df.copy()
        X_histgb = self._prepare_features_histgb(df_copy)

        # Re-préparer pour RSF (certaines colonnes peuvent avoir changé)
        for col in ['MAT', 'decade_install']:
            if col in df.columns:
                enc_col = f'{col}_enc'
                if enc_col not in df_copy.columns and col in self.label_encoders:
                    known = set(self.label_encoders[col].classes_)
                    df_copy[col] = df_copy[col].fillna('INCONNU').astype(str)
                    df_copy[col] = df_copy[col].apply(lambda x: x if x in known else 'INCONNU')
                    df_copy[enc_col] = self.label_encoders[col].transform(df_copy[col])

        X_rsf = df_copy[self.features_rsf].copy()

        # Scores HistGB
        score_histgb = self.histgb.predict_proba(X_histgb)[:, 1]

        # Scores RSF
        risk_rsf = self.rsf.predict(X_rsf)
        score_rsf = self.scaler_rsf.transform(risk_rsf.reshape(-1, 1)).flatten()
        score_rsf = np.clip(score_rsf, 0, 1)

        # Score combiné
        score_combined = self.alpha * score_histgb + (1 - self.alpha) * score_rsf

        # Temps médian et probabilités de survie
        surv_funcs = self.rsf.predict_survival_function(X_rsf)

        median_times = []
        surv_1y = []
        surv_3y = []
        surv_5y = []
        surv_10y = []

        for sf in surv_funcs:
            # Temps médian
            idx = np.searchsorted(-sf.y, -0.5)
            if idx < len(sf.x):
                median_times.append(sf.x[idx])
            else:
                median_times.append(sf.x[-1])

            # Probabilités de survie
            surv_1y.append(sf(1.0) if 1.0 <= sf.x[-1] else sf.y[-1])
            surv_3y.append(sf(3.0) if 3.0 <= sf.x[-1] else sf.y[-1])
            surv_5y.append(sf(5.0) if 5.0 <= sf.x[-1] else sf.y[-1])
            surv_10y.append(sf(10.0) if 10.0 <= sf.x[-1] else sf.y[-1])

        # Résultats
        results = pd.DataFrame({
            'score_combined': score_combined,
            'score_histgb': score_histgb,
            'score_rsf': score_rsf,
            'median_survival': median_times,
            'surv_1y': surv_1y,
            'surv_3y': surv_3y,
            'surv_5y': surv_5y,
            'surv_10y': surv_10y,
        })

        return results

    def save(self, path: Path = ARTIFACTS_DIR / "model_hybrid.joblib"):
        """Sauvegarde le modèle."""
        artifact = {
            'histgb': self.histgb,
            'rsf': self.rsf,
            'scaler_rsf': self.scaler_rsf,
            'label_encoders': self.label_encoders,
            'features_histgb': self.features_histgb,
            'features_rsf': self.features_rsf,
            'alpha': self.alpha,
            'metrics': self.metrics,
        }
        joblib.dump(artifact, path)
        print(f"Modèle sauvegardé: {path}")

    @classmethod
    def load(cls, path: Path = ARTIFACTS_DIR / "model_hybrid.joblib") -> 'HybridSurvivalModel':
        """Charge un modèle sauvegardé."""
        artifact = joblib.load(path)
        model = cls(alpha=artifact['alpha'])
        model.histgb = artifact['histgb']
        model.rsf = artifact['rsf']
        model.scaler_rsf = artifact['scaler_rsf']
        model.label_encoders = artifact['label_encoders']
        model.features_histgb = artifact['features_histgb']
        model.features_rsf = artifact['features_rsf']
        model.metrics = artifact['metrics']
        model.is_fitted = True
        return model


def train_hybrid_model(sample_size: int = 50000, alpha: float = 0.5):
    """Entraîne et évalue le modèle hybride."""

    # Charger les données
    print("Chargement des données...")
    df = pd.read_pickle(ARTIFACTS_DIR / "df_survival_prepared.pkl")

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Échantillon: {len(df):,} tronçons")

    # Entraîner
    model = HybridSurvivalModel(alpha=alpha)
    model.fit(df, verbose=True)

    # Sauvegarder
    model.save()

    # Test de prédiction
    print("\nTest de prédiction sur 5 tronçons:")
    sample = df.head(5)
    preds = model.predict(sample)

    print("\n| Score combiné | Score HistGB | Score RSF | Survie médiane | Survie 5 ans |")
    print("|---------------|--------------|-----------|----------------|--------------|")
    for i, row in preds.iterrows():
        print(f"| {row['score_combined']:.3f}         | {row['score_histgb']:.3f}        | {row['score_rsf']:.3f}     | {row['median_survival']:.0f} ans          | {row['surv_5y']*100:.0f}%          |")

    return model


if __name__ == "__main__":
    # Tester différentes valeurs de alpha
    print("\n" + "=" * 60)
    print("TEST DIFFERENTES VALEURS DE ALPHA")
    print("=" * 60)

    df = pd.read_pickle(ARTIFACTS_DIR / "df_survival_prepared.pkl")
    df = df.sample(n=30000, random_state=42)

    results = []
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        print(f"\n--- Alpha = {alpha} ---")
        model = HybridSurvivalModel(alpha=alpha)
        model.fit(df, verbose=False)
        results.append({
            'alpha': alpha,
            'c_index': model.metrics['c_index_combined']
        })
        print(f"C-index combiné: {model.metrics['c_index_combined']:.4f}")

    print("\n" + "=" * 60)
    print("RESUME")
    print("=" * 60)
    print("\n| Alpha | C-index |")
    print("|-------|---------|")
    for r in results:
        print(f"| {r['alpha']:.1f}   | {r['c_index']:.4f}  |")

    # Trouver le meilleur alpha
    best = max(results, key=lambda x: x['c_index'])
    print(f"\nMeilleur alpha: {best['alpha']} (C-index: {best['c_index']:.4f})")

    # Entraîner le modèle final avec le meilleur alpha
    print("\n" + "=" * 60)
    print(f"ENTRAINEMENT FINAL (alpha={best['alpha']})")
    print("=" * 60)

    final_model = HybridSurvivalModel(alpha=best['alpha'])
    final_model.fit(df, verbose=True)
    final_model.save()
