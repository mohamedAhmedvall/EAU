"""
Tests anti-leakage : vérifie qu'aucune feature n'utilise de données futures.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from predict_risk import compute_features_for_prediction


def _make_assets_df(gids, ddps, dhss, mats, diams, lngs):
    """Helper : crée un DataFrame patrimoine avec dates pré-parsées."""
    df = pd.DataFrame(
        {
            "GID": gids,
            "DDP": ddps,
            "DHS": dhss,
            "MAT": mats,
            "DIAMETRE": diams,
            "LNG": lngs,
        }
    )
    df["DDP_parsed"] = pd.to_datetime(df["DDP"], errors="coerce")
    df["DHS_parsed"] = pd.to_datetime(df["DHS"], errors="coerce")
    return df


@pytest.fixture
def sample_assets():
    """Crée un échantillon de données patrimoine pour les tests."""
    return _make_assets_df(
        gids=[1, 2, 3, 4, 5],
        ddps=["2000-01-01", "1990-06-15", "2010-03-20", "1980-01-01", "2005-09-01"],
        dhss=[pd.NaT, "2022-06-01", pd.NaT, "2019-01-01", pd.NaT],
        mats=["FT", "PVC", "PEHD", "FTG", "FT"],
        diams=[100, 150, 200, 80, 120],
        lngs=[100.0, 200.0, 50.0, 300.0, 150.0],
    )


@pytest.fixture
def sample_anomalies():
    """Crée un échantillon d'anomalies."""
    return pd.DataFrame(
        {
            "GID_OBJET": [1, 1, 2, 5, 1, 2],
            "DATE_DETECTION_parsed": [
                "2018-01-15",
                "2019-06-20",
                "2020-01-01",
                "2017-08-01",
                "2021-06-01",  # Après freeze => doit être exclu
                "2021-03-15",  # Après freeze => doit être exclu
            ],
            "TYPE_ANOMALIE": [
                "FUITE_DETECT_TR",
                "FUITE_SIGNAL_TR",
                "FUITE_DETECT_TR",
                "FUITE_DETECT_TR",
                "FUITE_DETECT_TR",
                "FUITE_SIGNAL_TR",
            ],
        }
    )


@pytest.fixture
def freeze_date():
    return pd.Timestamp("2020-06-01")


class TestAntiLeakage:
    """Tests anti-fuite de données."""

    def test_no_future_anomalies_in_features(
        self, sample_assets, sample_anomalies, freeze_date
    ):
        """
        Les features d'anomalies ne doivent utiliser que des données <= freeze_date.
        GID 1 a 3 anomalies dont 1 post-freeze. Il doit avoir n_fuites_total=2.
        """
        df_features = compute_features_for_prediction(
            sample_assets, sample_anomalies, freeze_date
        )

        pipe_1 = df_features[df_features["GID"] == 1]
        if len(pipe_1) > 0:
            assert pipe_1.iloc[0]["n_fuites_total"] == 2, (
                f"GID 1 devrait avoir 2 fuites (avant freeze), "
                f"mais a {pipe_1.iloc[0]['n_fuites_total']}"
            )

    def test_inactive_pipes_excluded(
        self, sample_assets, sample_anomalies, freeze_date
    ):
        """
        Les tronçons avec DHS <= freeze_date ne doivent pas être dans le résultat.
        GID 4 a DHS=2019-01-01 < freeze_date=2020-06-01 => exclu.
        """
        df_features = compute_features_for_prediction(
            sample_assets, sample_anomalies, freeze_date
        )

        assert 4 not in df_features["GID"].values, (
            "GID 4 (DHS=2019-01-01) ne devrait pas être dans les features "
            "(hors service avant freeze_date)"
        )

    def test_future_dhs_pipe_included(
        self, sample_assets, sample_anomalies, freeze_date
    ):
        """
        GID 2 a DHS=2022-06-01 > freeze_date => doit être inclus (encore actif à freeze).
        """
        df_features = compute_features_for_prediction(
            sample_assets, sample_anomalies, freeze_date
        )

        assert 2 in df_features["GID"].values, (
            "GID 2 (DHS=2022-06-01) devrait être dans les features "
            "(encore actif à freeze_date)"
        )

    def test_age_at_freeze_computed_correctly(
        self, sample_assets, sample_anomalies, freeze_date
    ):
        """
        L'âge doit être calculé entre DDP et freeze_date, pas la date actuelle.
        GID 1 : DDP=2000-01-01, freeze=2020-06-01 => ~20.4 ans
        """
        df_features = compute_features_for_prediction(
            sample_assets, sample_anomalies, freeze_date
        )

        pipe_1 = df_features[df_features["GID"] == 1]
        if len(pipe_1) > 0:
            age = pipe_1.iloc[0]["age_at_freeze"]
            expected_age = (freeze_date - pd.Timestamp("2000-01-01")).days / 365.25
            assert abs(age - expected_age) < 0.1, (
                f"Age devrait être ~{expected_age:.1f} ans, obtenu {age:.1f}"
            )

    def test_no_last_fuite_after_freeze(
        self, sample_assets, sample_anomalies, freeze_date
    ):
        """
        La date de dernière fuite ne doit pas dépasser freeze_date.
        """
        df_features = compute_features_for_prediction(
            sample_assets, sample_anomalies, freeze_date
        )

        if "date_last_fuite" in df_features.columns:
            valid_dates = df_features["date_last_fuite"].dropna()
            if len(valid_dates) > 0:
                max_date = valid_dates.max()
                assert max_date <= freeze_date, (
                    f"date_last_fuite max ({max_date}) dépasse freeze_date ({freeze_date})"
                )

    def test_days_since_last_fuite_positive(
        self, sample_assets, sample_anomalies, freeze_date
    ):
        """
        days_since_last_fuite doit être >= 0 (pas de fuite future).
        """
        df_features = compute_features_for_prediction(
            sample_assets, sample_anomalies, freeze_date
        )

        assert (df_features["days_since_last_fuite"] >= 0).all(), (
            "days_since_last_fuite contient des valeurs négatives (fuite future?)"
        )

    def test_future_pipe_not_included(
        self, sample_assets, sample_anomalies, freeze_date
    ):
        """
        Un tronçon avec DDP > freeze_date ne doit pas être inclus.
        """
        future_pipe = _make_assets_df(
            gids=[999],
            ddps=["2025-01-01"],
            dhss=[pd.NaT],
            mats=["PEHD"],
            diams=[100],
            lngs=[50.0],
        )
        df_with_future = pd.concat(
            [sample_assets, future_pipe], ignore_index=True
        )

        df_features = compute_features_for_prediction(
            df_with_future, sample_anomalies, freeze_date
        )

        assert 999 not in df_features["GID"].values, (
            "GID 999 (DDP=2025) ne devrait pas être dans les features"
        )
