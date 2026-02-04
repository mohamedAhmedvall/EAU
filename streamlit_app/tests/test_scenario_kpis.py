"""
Tests de cohérence des KPIs du scénario.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scenario import (
    ScenarioParams,
    ScenarioResult,
    compute_costs,
    run_baseline_scenario,
)
from src.optimization.greedy_solver import solve_greedy


@pytest.fixture
def sample_scored_df():
    """Crée un DataFrame de tronçons scorés pour les tests."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "GID": range(1, n + 1),
            "risk_score": np.random.beta(2, 5, n),  # Distribution réaliste
            "MAT": np.random.choice(["FT", "PVC", "PEHD", "FTG"], n),
            "DIAMETRE": np.random.choice([80, 100, 150, 200, 300], n),
            "LNG": np.random.uniform(10, 500, n),
            "longueur_km": np.random.uniform(0.01, 0.5, n),
            "age_at_freeze": np.random.uniform(5, 80, n),
            "n_fuites_total": np.random.poisson(1, n),
            "n_fuites_1y": np.random.poisson(0.2, n),
            "n_fuites_3y": np.random.poisson(0.5, n),
            "n_fuites_5y": np.random.poisson(0.8, n),
            "days_since_last_fuite": np.random.uniform(0, 7300, n),
            "has_recent_fuite": np.random.choice([0, 1], n),
            "ratio_age_median": np.random.uniform(0.3, 2.0, n),
            "overdue_years": np.random.uniform(0, 30, n),
            "rank": range(1, n + 1),
        }
    )


@pytest.fixture
def default_params():
    return ScenarioParams(
        freeze_date="2024-01-01",
        horizon_years=1,
        mode="baseline",
        baseline_mode="top_k_pct",
        top_k_pct=0.10,
        budget_total=1_000_000,
        cost_per_km=200_000,
    )


class TestKPICoherence:
    """Tests de cohérence des KPIs."""

    def test_risk_total_equals_treated_plus_untreated(
        self, sample_scored_df, default_params
    ):
        """risk_total = risk_treated + risk_untreated"""
        result = run_baseline_scenario(sample_scored_df, default_params)
        assert abs(result.risk_total - (result.risk_treated + result.risk_untreated)) < 1e-6

    def test_risk_treated_leq_risk_total(self, sample_scored_df, default_params):
        """Le risque traité ne peut pas dépasser le risque total."""
        result = run_baseline_scenario(sample_scored_df, default_params)
        assert result.risk_treated <= result.risk_total + 1e-6

    def test_n_selected_leq_n_total(self, sample_scored_df, default_params):
        """Le nombre de sélectionnés <= total."""
        result = run_baseline_scenario(sample_scored_df, default_params)
        assert result.n_selected <= result.n_total

    def test_coverage_pct_between_0_and_100(self, sample_scored_df, default_params):
        """Coverage percentage doit être entre 0 et 100."""
        result = run_baseline_scenario(sample_scored_df, default_params)
        assert 0 <= result.coverage_length_pct <= 100

    def test_baseline_top_k_selects_correct_count(
        self, sample_scored_df, default_params
    ):
        """Le mode top_k_pct sélectionne le bon nombre de tronçons."""
        result = run_baseline_scenario(sample_scored_df, default_params)
        expected_n = max(1, int(np.ceil(len(sample_scored_df) * 0.10)))
        # May differ slightly due to exclusions, but should be close
        assert abs(result.n_selected - expected_n) <= 1

    def test_baseline_selects_highest_scores(self, sample_scored_df, default_params):
        """Le baseline doit sélectionner les scores les plus élevés."""
        result = run_baseline_scenario(sample_scored_df, default_params)
        if result.n_selected > 0 and result.n_total > result.n_selected:
            min_selected_score = result.df_selected["risk_score"].min()
            max_unselected = result.df_scored[
                ~result.df_scored["GID"].isin(set(result.df_selected["GID"]))
            ]["risk_score"].max()
            assert min_selected_score >= max_unselected - 1e-6

    def test_cost_computation_positive(self, sample_scored_df, default_params):
        """Les coûts doivent être positifs."""
        costs = compute_costs(sample_scored_df, default_params)
        assert (costs >= 0).all()

    def test_excluded_materials_not_in_result(self, sample_scored_df):
        """Les matériaux exclus ne doivent pas apparaître dans la sélection."""
        params = ScenarioParams(
            freeze_date="2024-01-01",
            excluded_materials=["FT", "PVC"],
            budget_total=10_000_000,
            cost_per_km=200_000,
        )
        result = run_baseline_scenario(sample_scored_df, params)
        selected_mats = set(result.df_selected["MAT"].unique())
        assert not selected_mats.intersection({"FT", "PVC"})


class TestGreedySolver:
    """Tests du solveur greedy."""

    def test_greedy_respects_budget(self, sample_scored_df, default_params):
        """Le solveur greedy doit respecter le budget."""
        sample_scored_df["cost"] = sample_scored_df["longueur_km"] * 200_000

        result = solve_greedy(
            df=sample_scored_df,
            budget_total=100_000,
            weight_mode="length",
        )

        selected = sample_scored_df[
            sample_scored_df["GID"].isin(result["selected_gids"])
        ]
        total_cost = selected["cost"].sum()
        assert total_cost <= 100_000 + 1e-6

    def test_greedy_respects_lineaire_max(self, sample_scored_df, default_params):
        """Le solveur greedy doit respecter la contrainte linéaire."""
        sample_scored_df["cost"] = sample_scored_df["longueur_km"] * 200_000

        result = solve_greedy(
            df=sample_scored_df,
            budget_total=10_000_000,
            lineaire_max_km=1.0,
            weight_mode="length",
        )

        selected = sample_scored_df[
            sample_scored_df["GID"].isin(result["selected_gids"])
        ]
        total_length = selected["longueur_km"].sum()
        assert total_length <= 1.0 + 1e-6

    def test_greedy_returns_valid_status(self, sample_scored_df, default_params):
        """Le solveur doit retourner un statut valide."""
        sample_scored_df["cost"] = sample_scored_df["longueur_km"] * 200_000

        result = solve_greedy(
            df=sample_scored_df,
            budget_total=1_000_000,
        )

        assert result["status"] in ("OPTIMAL", "NO_SOLUTION")
        assert isinstance(result["selected_gids"], list)
        assert result["solve_time_s"] >= 0

    def test_greedy_excluded_gids(self, sample_scored_df, default_params):
        """Les GIDs exclus ne doivent pas être sélectionnés."""
        sample_scored_df["cost"] = sample_scored_df["longueur_km"] * 200_000
        excluded = [1, 2, 3]

        result = solve_greedy(
            df=sample_scored_df,
            budget_total=10_000_000,
            excluded_gids=excluded,
        )

        assert not set(result["selected_gids"]).intersection(set(excluded))
