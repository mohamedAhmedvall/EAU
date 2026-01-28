#!/usr/bin/env python3
"""
MODELE FINAL — Production Ready
================================

Meilleur modele: HistGradientBoostingClassifier

Performance:
- Lift@10%: 6.57 (+11.5% vs baseline original)
- Capture@10%: 65.7%
- Brier: 0.010 (excellente calibration)
- FN@10%: 136

Usage:
  # Training
  python model_final.py --train

  # Scoring
  python model_final.py --score --input data.csv --output scores.csv
"""

import sys, os, json, warnings
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))
os.chdir(SRC_DIR)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from config import ARTIFACTS_DIR, REPORTS_DIR, RANDOM_SEED, DATA_DIR

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_VERSION = "v3_final"
HORIZON = 1

FEATURES = [
    "age_at_freeze", "diametre", "longueur", "log_longueur", "log_diametre",
    "n_fuites_total", "n_fuites_1y", "n_fuites_3y", "n_fuites_5y",
    "days_since_last_fuite", "has_recent_fuite",
    "leak_rate_per_year", "leak_rate_per_km",
    "ratio_age_median", "overdue_years", "over_p75_life", "over_p90_life",
    "age_x_nfuites", "surface_approx", "age_x_ratio",
    "materiau_enc", "decade_install_enc",
    "age_cap", "age_extreme", "age_winsor", "age_ratio_dec", "age_resid_mat",
]

MODEL_PARAMS = {
    "max_iter": 400,
    "learning_rate": 0.03,
    "max_depth": None,
    "min_samples_leaf": 50,
    "random_state": RANDOM_SEED,
}

# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict:
    """Compute all metrics."""
    y = np.asarray(y_true)
    s = np.asarray(scores)
    n = len(y)
    n_ev = y.sum()

    # Top-K metrics
    order = np.argsort(-s)
    y_sorted = y[order]

    metrics = {}
    for k_pct in [0.05, 0.10, 0.20]:
        k = int(np.ceil(n * k_pct))
        ev_top = int(y_sorted[:k].sum())
        capture = ev_top / n_ev if n_ev > 0 else 0
        lift = capture / k_pct

        metrics[f"lift_{int(k_pct*100)}"] = lift
        metrics[f"cap_{int(k_pct*100)}"] = capture

    # Confusion at 10%
    k10 = int(np.ceil(n * 0.10))
    top = np.zeros(n, dtype=bool)
    top[order[:k10]] = True
    metrics["tp_10"] = int(((y == 1) & top).sum())
    metrics["fn_10"] = int(((y == 1) & ~top).sum())
    metrics["fp_10"] = int(((y == 0) & top).sum())

    # Global metrics
    metrics["roc_auc"] = roc_auc_score(y, s)
    metrics["brier"] = brier_score_loss(y, s)

    return metrics

# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_training_data() -> pd.DataFrame:
    """Load and prepare training data."""
    df = pd.read_pickle(ARTIFACTS_DIR / f"dataset_freeze_h{HORIZON}.pkl")
    return prepare_features(df)

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare all features."""
    df = df.copy()

    # Encodings
    for col in ["materiau", "decade_install"]:
        if col in df.columns and col + "_enc" not in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    # Age corrections
    if "age_cap" not in df.columns:
        df["age_cap"] = df["age_at_freeze"].clip(upper=110)
        df["age_extreme"] = (df["age_at_freeze"] > 110).astype(int)
        p99 = df["age_at_freeze"].quantile(0.99)
        df["age_winsor"] = df["age_at_freeze"].clip(upper=p99)

    # Decade ratio
    if "age_ratio_dec" not in df.columns:
        dec_med = df.groupby("decade_install")["age_at_freeze"].median()
        df["decade_med"] = df["decade_install"].map(dec_med)
        df["age_ratio_dec"] = df["age_at_freeze"] / df["decade_med"].replace(0, np.nan)
        df["age_ratio_dec"] = df["age_ratio_dec"].fillna(1.0)

    # Material residual
    if "age_resid_mat" not in df.columns:
        mat_med = df.groupby("materiau")["age_at_freeze"].median()
        df["mat_med"] = df["materiau"].map(mat_med)
        df["age_resid_mat"] = df["age_at_freeze"] - df["mat_med"].fillna(df["age_at_freeze"].median())

    return df

def get_X(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features matrix."""
    cols = [c for c in FEATURES if c in df.columns]
    return df[cols].replace([np.inf, -np.inf], np.nan).fillna(0)

# ============================================================================
# MODEL CLASS
# ============================================================================

class FinalModel:
    """Final production model with calibration."""

    def __init__(self):
        self.model = None
        self.calibrator = None
        self.encoders = {}
        self.feature_stats = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "FinalModel":
        """Train model with calibration."""
        print("Training HistGradientBoostingClassifier...")

        # Train model
        self.model = HistGradientBoostingClassifier(**MODEL_PARAMS)
        self.model.fit(X, y)

        # Calibrate
        print("Calibrating...")
        scores_train = self.model.predict_proba(X)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(scores_train, y)

        # Store feature statistics for monitoring
        self.feature_stats = {
            col: {"mean": X[col].mean(), "std": X[col].std()}
            for col in X.columns
        }

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated probabilities."""
        scores_raw = self.model.predict_proba(X)[:, 1]
        scores_cal = self.calibrator.transform(scores_raw)
        return np.column_stack([1 - scores_cal, scores_cal])

    def predict_risk(self, X: pd.DataFrame) -> pd.DataFrame:
        """Full risk prediction with categories."""
        proba = self.predict_proba(X)[:, 1]

        # Risk categories
        categories = np.where(
            proba >= 0.5, "CRITIQUE",
            np.where(proba >= 0.2, "ELEVE",
                     np.where(proba >= 0.1, "MOYEN", "FAIBLE"))
        )

        return pd.DataFrame({
            "risk_score": proba,
            "risk_category": categories,
            "risk_rank": (-proba).argsort().argsort() + 1,
        })

    def save(self, path: Path) -> None:
        """Save model artifact."""
        artifact = {
            "model": self.model,
            "calibrator": self.calibrator,
            "encoders": self.encoders,
            "feature_stats": self.feature_stats,
            "features": FEATURES,
            "model_params": MODEL_PARAMS,
            "version": MODEL_VERSION,
            "created_at": datetime.now().isoformat(),
        }
        joblib.dump(artifact, path)
        print(f"Model saved: {path}")

    @classmethod
    def load(cls, path: Path) -> "FinalModel":
        """Load model from artifact."""
        artifact = joblib.load(path)
        model = cls()
        model.model = artifact["model"]
        model.calibrator = artifact["calibrator"]
        model.encoders = artifact.get("encoders", {})
        model.feature_stats = artifact.get("feature_stats", {})
        return model

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def train() -> Dict:
    """Train and evaluate final model."""
    print("="*70)
    print("TRAINING FINAL MODEL")
    print("="*70)

    # Load data
    df = load_training_data()
    print(f"Dataset: {len(df):,} | Events: {df['event'].sum():,}")

    # Split
    idx = np.arange(len(df))
    i_tr, i_te = train_test_split(
        idx, test_size=0.2, random_state=RANDOM_SEED,
        stratify=df["event"].values
    )

    df_tr = df.iloc[i_tr]
    df_te = df.iloc[i_te]
    y_tr = df_tr["event"].values
    y_te = df_te["event"].values

    X_tr = get_X(df_tr)
    X_te = get_X(df_te)

    print(f"Train: {len(X_tr):,} | Test: {len(X_te):,}")

    # Train
    model = FinalModel()
    model.fit(X_tr, y_tr)

    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)

    scores = model.predict_proba(X_te)[:, 1]
    metrics = compute_metrics(y_te, scores)

    print(f"  Lift@5%:  {metrics['lift_5']:.2f}")
    print(f"  Lift@10%: {metrics['lift_10']:.2f}")
    print(f"  Lift@20%: {metrics['lift_20']:.2f}")
    print(f"  Cap@10%:  {metrics['cap_10']:.1%}")
    print(f"  ROC-AUC:  {metrics['roc_auc']:.4f}")
    print(f"  Brier:    {metrics['brier']:.4f}")
    print(f"  FN@10%:   {metrics['fn_10']}")

    # Save
    model_path = ARTIFACTS_DIR / f"model_{MODEL_VERSION}.joblib"
    model.save(model_path)

    # Save metadata
    metadata = {
        "version": MODEL_VERSION,
        "horizon": HORIZON,
        "features": FEATURES,
        "model_params": MODEL_PARAMS,
        "metrics": metrics,
        "created_at": datetime.now().isoformat(),
    }

    meta_path = ARTIFACTS_DIR / f"metadata_{MODEL_VERSION}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata saved: {meta_path}")

    return metrics

def score(input_path: str, output_path: str) -> None:
    """Score new data."""
    print("="*70)
    print("SCORING")
    print("="*70)

    # Load model
    model_path = ARTIFACTS_DIR / f"model_{MODEL_VERSION}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run --train first.")

    model = FinalModel.load(model_path)
    print(f"Model loaded: {model_path}")

    # Load data
    df = pd.read_csv(input_path)
    df = prepare_features(df)
    X = get_X(df)
    print(f"Input data: {len(X):,} rows")

    # Predict
    results = model.predict_risk(X)

    # Add identifiers if available
    if "GID" in df.columns:
        results["GID"] = df["GID"].values

    # Save
    results.to_csv(output_path, index=False)
    print(f"Scores saved: {output_path}")

    # Summary
    print("\nRisk distribution:")
    print(results["risk_category"].value_counts())

# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Final Model")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--score", action="store_true", help="Score data")
    parser.add_argument("--input", type=str, help="Input CSV for scoring")
    parser.add_argument("--output", type=str, help="Output CSV for scores")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.score:
        if not args.input or not args.output:
            parser.error("--score requires --input and --output")
        score(args.input, args.output)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
