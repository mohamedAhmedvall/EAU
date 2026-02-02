"""
Modèles de survie pour la prédiction de défaillance des canalisations

1. Random Survival Forest (RSF) - scikit-survival
2. Survival Neural Network (DeepSurv) - PyTorch

Ces modèles prédisent le TEMPS avant défaillance, pas juste une probabilité binaire.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import joblib
import json
from datetime import datetime

# Scikit-survival
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.nonparametric import kaplan_meier_estimator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Paths
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
REPORTS_DIR = Path(__file__).parent.parent / "reports" / "survival_models"


def load_survival_data() -> pd.DataFrame:
    """Charge le dataset de survie préparé."""
    df = pd.read_pickle(ARTIFACTS_DIR / "df_survival_prepared.pkl")
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Prépare les features et la target pour l'analyse de survie.

    Returns:
        X: DataFrame des features
        y: Array structuré (event, duration) pour scikit-survival
    """
    # Features à utiliser
    feature_cols = []

    # Features numériques
    num_features = ["DIAMETRE", "LNG", "age", "n_anomalies"]
    for col in num_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            feature_cols.append(col)

    # Encoder le matériau
    if "MAT" in df.columns:
        le_mat = LabelEncoder()
        df["MAT_enc"] = le_mat.fit_transform(df["MAT"].fillna("INCONNU"))
        feature_cols.append("MAT_enc")

    # Encoder la décennie d'installation
    if "decade_install" in df.columns:
        df["decade_enc"] = df["decade_install"].fillna(1970)
        feature_cols.append("decade_enc")

    # Features dérivées
    if "age" in df.columns and "DIAMETRE" in df.columns:
        df["age_years"] = df["age"] / 365.25
        df["age_x_diam"] = df["age_years"] * df["DIAMETRE"]
        feature_cols.extend(["age_years", "age_x_diam"])

    if "n_anomalies" in df.columns and "LNG" in df.columns:
        df["anomaly_rate"] = df["n_anomalies"] / (df["LNG"] / 1000 + 0.01)
        feature_cols.append("anomaly_rate")

    X = df[feature_cols].copy()

    # Target pour scikit-survival: array structuré (event, duration)
    y = np.array(
        [(bool(e), d) for e, d in zip(df["event"], df["duration_years"])],
        dtype=[("event", bool), ("duration", float)]
    )

    return X, y


def train_random_survival_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 10,
    min_samples_split: int = 20,
    min_samples_leaf: int = 10,
    random_state: int = 42,
) -> RandomSurvivalForest:
    """
    Entraîne un Random Survival Forest.

    Le RSF combine les arbres de décision avec l'analyse de survie.
    Chaque noeud maximise la différence de survie entre les groupes.
    """
    print("Entraînement Random Survival Forest...")
    print(f"  n_estimators={n_estimators}, max_depth={max_depth}")

    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=random_state,
    )

    rsf.fit(X_train, y_train)
    print("  Entraînement terminé")

    return rsf


def evaluate_survival_model(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    times: Optional[np.ndarray] = None,
) -> Dict:
    """
    Évalue un modèle de survie.

    Métriques:
    - C-index: capacité à classer correctement les paires (plus c'est haut, mieux c'est)
    - Integrated Brier Score: erreur de calibration (plus c'est bas, mieux c'est)
    """
    # C-index
    risk_scores = model.predict(X_test)
    c_index = concordance_index_censored(
        y_test["event"], y_test["duration"], risk_scores
    )[0]

    # Integrated Brier Score
    if times is None:
        times = np.percentile(y_test["duration"][y_test["event"]], [10, 25, 50, 75, 90])
        times = times[times > 0]

    try:
        # Survival function pour chaque individu
        surv_funcs = model.predict_survival_function(X_test)

        # Construire la matrice de probabilités de survie
        preds = np.zeros((len(X_test), len(times)))
        for i, sf in enumerate(surv_funcs):
            preds[i] = sf(times)

        ibs = integrated_brier_score(y_test, y_test, preds, times)
    except Exception as e:
        print(f"  Warning: IBS non calculable ({e})")
        ibs = None

    metrics = {
        "c_index": c_index,
        "integrated_brier_score": ibs,
    }

    return metrics


def predict_survival_probability(
    model,
    X: pd.DataFrame,
    time_horizon: float = 1.0,  # en années
) -> np.ndarray:
    """
    Prédit la probabilité de survie à un horizon donné.

    Args:
        model: Modèle RSF entraîné
        X: Features
        time_horizon: Horizon de prédiction en années

    Returns:
        Probabilités de survie (1 - proba_defaillance)
    """
    surv_funcs = model.predict_survival_function(X)
    probs = np.array([sf(time_horizon) for sf in surv_funcs])
    return probs


def predict_median_survival_time(model, X: pd.DataFrame) -> np.ndarray:
    """
    Prédit le temps médian de survie pour chaque tronçon.

    C'est le temps auquel la probabilité de survie atteint 50%.
    """
    surv_funcs = model.predict_survival_function(X)
    median_times = []

    for sf in surv_funcs:
        # Trouver quand S(t) = 0.5
        times = sf.x
        surv = sf.y

        idx = np.searchsorted(-surv, -0.5)
        if idx < len(times):
            median_times.append(times[idx])
        else:
            median_times.append(times[-1])  # Censuré

    return np.array(median_times)


# =============================================================================
# SURVIVAL NEURAL NETWORK (DeepSurv)
# =============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch non disponible, DeepSurv désactivé")


if TORCH_AVAILABLE:

    class DeepSurv(nn.Module):
        """
        Réseau de neurones pour l'analyse de survie (DeepSurv).

        Basé sur le modèle de Cox, mais avec un réseau de neurones
        pour apprendre les relations non-linéaires.

        Architecture: MLP qui prédit le log-risque (log hazard ratio)
        Loss: Negative partial log-likelihood de Cox
        """

        def __init__(
            self,
            n_features: int,
            hidden_layers: List[int] = [64, 32],
            dropout: float = 0.3,
        ):
            super().__init__()

            layers = []
            prev_size = n_features

            for hidden_size in hidden_layers:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                prev_size = hidden_size

            # Couche de sortie: score de risque (scalaire)
            layers.append(nn.Linear(prev_size, 1))

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)


    def cox_partial_likelihood_loss(risk_scores, events, durations):
        """
        Calcule la negative partial log-likelihood de Cox.

        Cette loss encourage le modèle à donner des scores de risque
        plus élevés aux individus qui défaillent plus tôt.
        """
        # Trier par durée décroissante
        order = torch.argsort(durations, descending=True)
        risk_scores = risk_scores[order]
        events = events[order]

        # Log-sum-exp cumulatif pour le dénominateur
        hazard_ratio = torch.exp(risk_scores)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))

        # Likelihood pour les événements observés
        uncensored_likelihood = risk_scores - log_risk
        censored_likelihood = uncensored_likelihood * events

        # Moyenne sur les événements
        num_observed_events = torch.sum(events)
        if num_observed_events > 0:
            return -torch.sum(censored_likelihood) / num_observed_events
        else:
            return torch.tensor(0.0)


    def train_deepsurv(
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        hidden_layers: List[int] = [64, 32],
        dropout: float = 0.3,
        lr: float = 0.001,
        epochs: int = 100,
        batch_size: int = 256,
        patience: int = 10,
        verbose: bool = True,
    ) -> Tuple[DeepSurv, StandardScaler]:
        """
        Entraîne un réseau DeepSurv.
        """
        print("Entraînement DeepSurv...")
        print(f"  Architecture: {X_train.shape[1]} -> {hidden_layers} -> 1")

        # Normaliser les features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        # Convertir en tenseurs
        X_tensor = torch.FloatTensor(X_scaled)
        events_tensor = torch.FloatTensor(y_train["event"].astype(float))
        durations_tensor = torch.FloatTensor(y_train["duration"])

        # DataLoader
        dataset = TensorDataset(X_tensor, events_tensor, durations_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Modèle
        model = DeepSurv(
            n_features=X_train.shape[1],
            hidden_layers=hidden_layers,
            dropout=dropout,
        )

        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Early stopping
        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0

            for X_batch, events_batch, durations_batch in loader:
                optimizer.zero_grad()

                risk_scores = model(X_batch).squeeze()
                loss = cox_partial_likelihood_loss(
                    risk_scores, events_batch, durations_batch
                )

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(loader)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Restaurer le meilleur modèle
        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        print("  Entraînement terminé")

        return model, scaler


    def evaluate_deepsurv(
        model: DeepSurv,
        scaler: StandardScaler,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
    ) -> Dict:
        """Évalue un modèle DeepSurv."""
        model.eval()

        X_scaled = scaler.transform(X_test)
        X_tensor = torch.FloatTensor(X_scaled)

        with torch.no_grad():
            risk_scores = model(X_tensor).squeeze().numpy()

        c_index = concordance_index_censored(
            y_test["event"], y_test["duration"], risk_scores
        )[0]

        return {"c_index": c_index}


# =============================================================================
# MAIN: Entraînement et comparaison
# =============================================================================

def run_experiments(sample_size: int = 50000):
    """Lance les expériences de comparaison des modèles de survie."""

    print("=" * 60)
    print("MODELES DE SURVIE - Comparaison")
    print("=" * 60)

    # Charger les données
    print("\n1. Chargement des données...")
    df = load_survival_data()
    print(f"   {len(df):,} tronçons total")

    # Échantillonner pour accélérer
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"   Échantillon: {len(df):,} tronçons")

    print(f"   Events: {df['event'].sum():,} ({df['event'].mean()*100:.1f}%)")

    # Préparer features
    print("\n2. Préparation des features...")
    X, y = prepare_features(df)
    print(f"   Features: {list(X.columns)}")
    print(f"   Shape: {X.shape}")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {len(X_train):,}, Test: {len(X_test):,}")

    results = {}

    # ==========================================================================
    # Random Survival Forest
    # ==========================================================================
    print("\n" + "=" * 60)
    print("3. RANDOM SURVIVAL FOREST")
    print("=" * 60)

    rsf = train_random_survival_forest(
        X_train, y_train,
        n_estimators=100,
        max_depth=8,
        min_samples_split=30,
        min_samples_leaf=15,
    )

    metrics_rsf = evaluate_survival_model(rsf, X_test, y_test)
    print(f"\n   Résultats RSF:")
    print(f"   - C-index: {metrics_rsf['c_index']:.4f}")
    if metrics_rsf['integrated_brier_score']:
        print(f"   - IBS: {metrics_rsf['integrated_brier_score']:.4f}")

    results["rsf"] = metrics_rsf

    # Sauvegarder le modèle RSF
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": rsf,
        "features": list(X.columns),
        "metrics": metrics_rsf,
    }, ARTIFACTS_DIR / "model_rsf.joblib")
    print(f"   Modèle sauvegardé: artifacts/model_rsf.joblib")

    # ==========================================================================
    # DeepSurv
    # ==========================================================================
    if TORCH_AVAILABLE:
        print("\n" + "=" * 60)
        print("4. DEEPSURV (Neural Network)")
        print("=" * 60)

        deepsurv, scaler = train_deepsurv(
            X_train, y_train,
            hidden_layers=[64, 32, 16],
            dropout=0.3,
            lr=0.001,
            epochs=100,
            batch_size=256,
            patience=15,
        )

        metrics_deepsurv = evaluate_deepsurv(deepsurv, scaler, X_test, y_test)
        print(f"\n   Résultats DeepSurv:")
        print(f"   - C-index: {metrics_deepsurv['c_index']:.4f}")

        results["deepsurv"] = metrics_deepsurv

        # Sauvegarder
        torch.save({
            "model_state": deepsurv.state_dict(),
            "scaler": scaler,
            "features": list(X.columns),
            "hidden_layers": [64, 32, 16],
            "metrics": metrics_deepsurv,
        }, ARTIFACTS_DIR / "model_deepsurv.pt")
        print(f"   Modèle sauvegardé: artifacts/model_deepsurv.pt")

    # ==========================================================================
    # Comparaison
    # ==========================================================================
    print("\n" + "=" * 60)
    print("5. COMPARAISON")
    print("=" * 60)

    print("\n   | Modèle     | C-index |")
    print("   |------------|---------|")
    for name, metrics in results.items():
        print(f"   | {name:<10} | {metrics['c_index']:.4f}  |")

    # Comparaison avec le modèle actuel (classification)
    print("\n   Note: Le modèle actuel (HistGradientBoosting) a:")
    print("   - AUC-ROC: 0.879")
    print("   - Lift@10%: 6.57")
    print("\n   Le C-index des modèles de survie est comparable à l'AUC,")
    print("   mais prédit aussi le TEMPS avant défaillance.")

    # ==========================================================================
    # Exemple de prédiction
    # ==========================================================================
    print("\n" + "=" * 60)
    print("6. EXEMPLE DE PREDICTION")
    print("=" * 60)

    # Prédire la survie à 1, 3, 5 ans pour quelques tronçons
    sample_idx = X_test.iloc[:5].index
    X_sample = X_test.iloc[:5]

    print("\n   Probabilité de survie (= 1 - P(défaillance)):")
    print("   | Tronçon | 1 an  | 3 ans | 5 ans | Médiane (ans) |")
    print("   |---------|-------|-------|-------|---------------|")

    surv_1y = predict_survival_probability(rsf, X_sample, 1.0)
    surv_3y = predict_survival_probability(rsf, X_sample, 3.0)
    surv_5y = predict_survival_probability(rsf, X_sample, 5.0)
    median_times = predict_median_survival_time(rsf, X_sample)

    for i, idx in enumerate(sample_idx):
        print(f"   | {idx:<7} | {surv_1y[i]:.3f} | {surv_3y[i]:.3f} | {surv_5y[i]:.3f} | {median_times[i]:.1f}           |")

    # Sauvegarder le rapport
    report = {
        "date": datetime.now().isoformat(),
        "results": {k: {kk: float(vv) if vv else None for kk, vv in v.items()} for k, v in results.items()},
        "features": list(X.columns),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    with open(REPORTS_DIR / "comparison_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n   Rapport sauvegardé: reports/survival_models/comparison_report.json")

    return results


if __name__ == "__main__":
    run_experiments()
