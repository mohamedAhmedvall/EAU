"""
Service de scoring : wrapper autour de predict_risk.py existant.
Fournit une interface propre pour l'app Streamlit avec cache.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union
import streamlit as st
import sys

# Ajouter le répertoire src du projet principal au path
_PROJECT_SRC = Path(__file__).parent.parent.parent / "src"
if str(_PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(_PROJECT_SRC))

from predict_risk import (
    load_model,
    compute_features_for_prediction,
    predict_risk as _predict_risk,
    DEFAULT_MODEL_PATH,
)

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"


@st.cache_resource
def get_model_artifact(model_path: Optional[str] = None):
    """
    Charge et cache le modèle (resource-level cache, partagé entre sessions).
    """
    if model_path is None:
        model_path = str(DEFAULT_MODEL_PATH)
    return load_model(model_path)


def score_pipes(
    df_assets: pd.DataFrame,
    df_anomalies: pd.DataFrame,
    freeze_date: Union[str, pd.Timestamp],
    horizon_years: int = 1,
    model_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Score tous les tronçons actifs à freeze_date et retourne un DataFrame enrichi.

    Returns
    -------
    pd.DataFrame
        Colonnes : GID, risk_score, MAT, DIAMETRE, LNG, age_at_freeze,
                   n_fuites_total, days_since_last_fuite, longueur_km, cost_proxy, ...
    """
    if isinstance(freeze_date, str):
        freeze_date = pd.Timestamp(freeze_date)

    # Charger le modèle
    if model_path is None:
        model_path = str(DEFAULT_MODEL_PATH)

    artifact = get_model_artifact(model_path)
    model = artifact["model"]
    encoders = artifact["encoders"]
    feature_cols = artifact["metadata"]["feature_cols"]

    # Charger life_stats
    life_stats_path = ARTIFACTS_DIR / f"life_stats_h{horizon_years}.csv"
    life_stats = pd.read_csv(life_stats_path) if life_stats_path.exists() else None

    # Calculer les features
    df_features = compute_features_for_prediction(
        df_assets, df_anomalies, freeze_date, life_stats
    )

    if len(df_features) == 0:
        return pd.DataFrame()

    # Encoder les catégorielles
    for col, le in encoders.items():
        if col in df_features.columns:
            df_features[col + "_encoded"] = (
                df_features[col]
                .astype(str)
                .apply(lambda x, _le=le: _le.transform([x])[0] if x in _le.classes_ else 0)
            )
        else:
            df_features[col + "_encoded"] = 0

    # Prédire
    X = df_features[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = model.predict(X)

    # Construire le DataFrame de sortie
    result = pd.DataFrame(
        {
            "GID": df_features["GID"].values,
            "risk_score": scores,
            "MAT": df_features["MAT"].values if "MAT" in df_features.columns else df_features["materiau"].values,
            "DIAMETRE": df_features["diametre"].values,
            "LNG": df_features["longueur"].values,
            "longueur_km": df_features["longueur_km"].values,
            "age_at_freeze": df_features["age_at_freeze"].values,
            "n_fuites_total": df_features["n_fuites_total"].values,
            "n_fuites_1y": df_features["n_fuites_1y"].values,
            "n_fuites_3y": df_features["n_fuites_3y"].values,
            "n_fuites_5y": df_features["n_fuites_5y"].values,
            "days_since_last_fuite": df_features["days_since_last_fuite"].values,
            "has_recent_fuite": df_features["has_recent_fuite"].values,
            "ratio_age_median": df_features["ratio_age_median"].values,
            "overdue_years": df_features["overdue_years"].values,
        }
    )

    # Ajouter DDP si disponible
    if "DDP_parsed" in df_features.columns:
        result["DDP"] = df_features["DDP_parsed"].values
    elif "DDP" in df_features.columns:
        result["DDP"] = df_features["DDP"].values

    # Rank
    result = result.sort_values("risk_score", ascending=False).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)

    # Catégorie de risque (terciles par défaut)
    result["risk_category"] = pd.cut(
        result["risk_score"],
        bins=[-0.001, 0.33, 0.66, 1.001],
        labels=["FAIBLE", "MOYEN", "ELEVE"],
    )

    return result


def get_model_info(model_path: Optional[str] = None) -> dict:
    """
    Retourne les métadonnées du modèle chargé.
    """
    artifact = get_model_artifact(model_path)
    meta = artifact["metadata"]
    return {
        "model_type": meta.get("model_type", "Unknown"),
        "horizon_years": meta.get("horizon_years", 1),
        "freeze_date_training": meta.get("freeze_date", "N/A"),
        "n_features": len(meta.get("feature_cols", [])),
        "feature_cols": meta.get("feature_cols", []),
        "n_train": meta.get("n_train", 0),
        "n_test": meta.get("n_test", 0),
        "test_metrics": meta.get("test_metrics", {}),
        "created_at": meta.get("created_at", "N/A"),
    }
