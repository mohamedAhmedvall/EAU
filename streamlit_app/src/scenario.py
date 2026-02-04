"""
Structures de données pour les scénarios de plan de renouvellement.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import date


@dataclass
class ScenarioParams:
    """Paramètres d'un scénario de renouvellement."""

    # Dates
    freeze_date: str  # YYYY-MM-DD
    horizon_years: int = 1

    # Plan
    plan_years: int = 1  # 1 = annuel, N = multi-annuel

    # Sélection baseline
    mode: str = "baseline"  # "baseline" ou "optimized"
    baseline_mode: str = "top_k_pct"  # "top_k_pct", "top_n", "top_length"
    top_k_pct: float = 0.10
    top_n: int = 1000
    top_length_km: float = 50.0

    # Budget et contraintes
    budget_total: float = 10_000_000.0  # euros
    budget_annual: Optional[float] = None
    lineaire_max_km: Optional[float] = None
    cost_per_km: float = 200_000.0  # euros/km

    # Coûts par matériau (multiplicateur)
    cost_factor_mat: Dict[str, float] = field(default_factory=dict)
    # Coûts par diamètre (multiplicateur)
    cost_factor_diam: Dict[str, float] = field(default_factory=dict)

    # Contraintes de groupe
    max_pct_budget_per_zone: Optional[float] = None  # ex: 0.30 = max 30% du budget par zone
    zone_column: Optional[str] = None

    # Exclusions
    excluded_gids: List[int] = field(default_factory=list)
    excluded_materials: List[str] = field(default_factory=list)

    # Pondération
    weight_mode: str = "length"  # "length" ou "uniform"


@dataclass
class ScenarioResult:
    """Résultat d'un scénario de renouvellement."""

    params: ScenarioParams
    df_scored: pd.DataFrame  # Tous les tronçons scorés
    df_selected: pd.DataFrame  # Tronçons sélectionnés pour renouvellement

    # KPIs
    risk_total: float = 0.0
    risk_treated: float = 0.0
    risk_untreated: float = 0.0
    coverage_length_km: float = 0.0
    coverage_length_pct: float = 0.0
    budget_used: float = 0.0
    n_selected: int = 0
    n_total: int = 0

    # Solver info
    solver_status: str = ""
    solver_time_s: float = 0.0

    def compute_kpis(self):
        """Calcule les KPIs métier à partir des données scorées et sélectionnées."""
        df = self.df_scored
        sel = self.df_selected

        weight_col = "longueur_km"
        if weight_col not in df.columns:
            df["longueur_km"] = df["LNG"] / 1000.0
        if weight_col not in sel.columns and len(sel) > 0:
            sel["longueur_km"] = sel["LNG"] / 1000.0

        # Exposure = longueur_km
        self.risk_total = (df["risk_score"] * df["longueur_km"]).sum()

        if len(sel) > 0:
            self.risk_treated = (sel["risk_score"] * sel["longueur_km"]).sum()
            self.coverage_length_km = sel["longueur_km"].sum()
            self.budget_used = sel["cost"].sum() if "cost" in sel.columns else 0.0
            self.n_selected = len(sel)
        else:
            self.risk_treated = 0.0
            self.coverage_length_km = 0.0
            self.budget_used = 0.0
            self.n_selected = 0

        self.risk_untreated = self.risk_total - self.risk_treated
        total_length = df["longueur_km"].sum()
        self.coverage_length_pct = (
            self.coverage_length_km / total_length * 100 if total_length > 0 else 0.0
        )
        self.n_total = len(df)


def compute_costs(df: pd.DataFrame, params: ScenarioParams) -> pd.Series:
    """
    Calcule le coût proxy pour chaque tronçon.

    cost = longueur_km * cost_per_km * factor_mat * factor_diam
    """
    lng_km = df["LNG"] / 1000.0 if "longueur_km" not in df.columns else df["longueur_km"]

    cost = lng_km * params.cost_per_km

    # Facteur matériau
    if params.cost_factor_mat:
        mat_factor = df["MAT"].map(params.cost_factor_mat).fillna(1.0)
        cost = cost * mat_factor

    # Facteur diamètre
    if params.cost_factor_diam:
        diam_factor = df["DIAMETRE"].map(params.cost_factor_diam).fillna(1.0)
        cost = cost * diam_factor

    return cost


def run_baseline_scenario(
    df_scored: pd.DataFrame, params: ScenarioParams
) -> ScenarioResult:
    """
    Exécute un scénario baseline (simple sélection top-K).
    """
    df = df_scored.copy()

    # Ajouter longueur_km et coûts
    if "longueur_km" not in df.columns:
        df["longueur_km"] = df["LNG"] / 1000.0
    df["cost"] = compute_costs(df, params)

    # Appliquer les exclusions
    if params.excluded_gids:
        df = df[~df["GID"].isin(params.excluded_gids)]
    if params.excluded_materials:
        df = df[~df["MAT"].isin(params.excluded_materials)]

    # Trier par score décroissant
    df = df.sort_values("risk_score", ascending=False).reset_index(drop=True)

    # Sélection baseline
    if params.baseline_mode == "top_k_pct":
        n_select = max(1, int(np.ceil(len(df) * params.top_k_pct)))
        selected = df.head(n_select)
    elif params.baseline_mode == "top_n":
        selected = df.head(min(params.top_n, len(df)))
    elif params.baseline_mode == "top_length":
        cum_length = df["longueur_km"].cumsum()
        mask = cum_length <= params.top_length_km
        if not mask.any():
            selected = df.head(1)
        else:
            selected = df[mask]
    else:
        n_select = max(1, int(np.ceil(len(df) * params.top_k_pct)))
        selected = df.head(n_select)

    result = ScenarioResult(
        params=params,
        df_scored=df,
        df_selected=selected.copy(),
        solver_status="baseline",
    )
    result.compute_kpis()
    return result
