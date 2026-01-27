"""
Module de chargement et cache des données pour l'app Streamlit.
Wraps les données patrimoine + anomalies avec cache Streamlit.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import streamlit as st

# Chemins par défaut (relatifs au repo EAU)
DEFAULT_BASE_DIR = Path(__file__).parent.parent.parent
DEFAULT_DATA_DIR = DEFAULT_BASE_DIR / "data"
DEFAULT_ARTIFACTS_DIR = DEFAULT_BASE_DIR / "artifacts"


@st.cache_data(ttl=3600)
def load_patrimoine(filepath: Optional[str] = None, sample_n: Optional[int] = None) -> pd.DataFrame:
    """
    Charge la table patrimoine (tronçons) avec parsing des dates.

    Parameters
    ----------
    filepath : str, optional
        Chemin vers le CSV patrimoine. Si None, utilise le chemin par défaut.
    sample_n : int, optional
        Si spécifié, échantillonne N lignes (mode dev).

    Returns
    -------
    pd.DataFrame avec colonnes parsées : GID, DDP_parsed, DHS_parsed, MAT, DIAMETRE, LNG, etc.
    """
    if filepath is None:
        filepath = str(DEFAULT_DATA_DIR / "v1_trafic_prepared.csv")

    df = pd.read_csv(filepath)

    if sample_n and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)

    # Parser les dates
    df["DDP_parsed"] = pd.to_datetime(df["DDP"], errors="coerce", utc=True)
    df["DDP_parsed"] = df["DDP_parsed"].dt.tz_localize(None)

    if "DHS" in df.columns:
        df["DHS_parsed"] = pd.to_datetime(df["DHS"], errors="coerce", utc=True)
        df["DHS_parsed"] = df["DHS_parsed"].dt.tz_localize(None)
    else:
        df["DHS_parsed"] = pd.NaT

    return df


@st.cache_data(ttl=3600)
def load_anomalies(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Charge la table anomalies avec parsing des dates.

    Parameters
    ----------
    filepath : str, optional
        Chemin vers le CSV anomalies. Si None, utilise le chemin par défaut.

    Returns
    -------
    pd.DataFrame avec DATE_DETECTION parsée.
    """
    if filepath is None:
        filepath = str(DEFAULT_DATA_DIR / "historiqueanomalie.csv")

    df = pd.read_csv(filepath)

    # Parser la date de détection
    if "DATE_DETECTION_parsed" in df.columns:
        df["DATE_DETECTION"] = pd.to_datetime(
            df["DATE_DETECTION_parsed"], errors="coerce", utc=True
        )
        df["DATE_DETECTION"] = df["DATE_DETECTION"].dt.tz_localize(None)
    elif "DATE_DETECTION" in df.columns:
        df["DATE_DETECTION"] = pd.to_datetime(
            df["DATE_DETECTION"], errors="coerce", utc=True
        )
        df["DATE_DETECTION"] = df["DATE_DETECTION"].dt.tz_localize(None)

    return df


@st.cache_data(ttl=3600)
def load_life_stats(horizon: int = 1) -> Optional[pd.DataFrame]:
    """
    Charge les statistiques de durée de vie par matériau.
    """
    path = DEFAULT_ARTIFACTS_DIR / f"life_stats_h{horizon}.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def get_data_dates(df_assets: pd.DataFrame, df_anomalies: pd.DataFrame) -> dict:
    """
    Retourne les dates min/max utiles pour paramétrer le scénario.
    """
    ddp_min = df_assets["DDP_parsed"].min()
    ddp_max = df_assets["DDP_parsed"].max()
    dhs_max = df_assets["DHS_parsed"].max()

    ano_min = df_anomalies["DATE_DETECTION"].min()
    ano_max = df_anomalies["DATE_DETECTION"].max()

    max_obs = max(
        d for d in [dhs_max, ano_max] if pd.notna(d)
    )

    return {
        "ddp_min": ddp_min,
        "ddp_max": ddp_max,
        "dhs_max": dhs_max,
        "ano_min": ano_min,
        "ano_max": ano_max,
        "max_obs_date": max_obs,
    }


def has_geometry(df: pd.DataFrame) -> bool:
    """
    Vérifie si le DataFrame contient des données géométriques.
    Cherche des colonnes WKT, lat/lon, ou geometry.
    """
    geo_cols = {"geometry", "geom", "wkt", "the_geom", "shape"}
    latlon_pairs = [("lat", "lon"), ("latitude", "longitude"), ("y", "x")]

    cols_lower = {c.lower(): c for c in df.columns}

    # Check WKT / geometry columns
    for gc in geo_cols:
        if gc in cols_lower:
            return True

    # Check lat/lon pairs
    for lat_col, lon_col in latlon_pairs:
        if lat_col in cols_lower and lon_col in cols_lower:
            return True

    return False


def get_geometry_column(df: pd.DataFrame) -> Optional[str]:
    """
    Retourne le nom de la colonne géométrique si elle existe.
    """
    geo_cols = {"geometry", "geom", "wkt", "the_geom", "shape"}
    cols_lower = {c.lower(): c for c in df.columns}

    for gc in geo_cols:
        if gc in cols_lower:
            return cols_lower[gc]
    return None
