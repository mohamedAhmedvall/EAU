"""
Solveur OR-Tools (programmation linéaire en nombres entiers) pour le plan de renouvellement.

Problème:
    max  sum( score_i * weight_i * x_i )
    s.c. sum( cost_i * x_i )      <= budget_total
         sum( longueur_i * x_i )   <= lineaire_max  (optionnel)
         contraintes de zone                        (optionnel)
         x_i ∈ {0, 1}
"""
import pandas as pd
import numpy as np
import time
from typing import Optional, List, Dict

try:
    from ortools.linear_solver import pywraplp

    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False


def is_available() -> bool:
    return ORTOOLS_AVAILABLE


def solve_milp(
    df: pd.DataFrame,
    budget_total: float,
    lineaire_max_km: Optional[float] = None,
    weight_mode: str = "length",
    excluded_gids: Optional[List[int]] = None,
    excluded_materials: Optional[List[str]] = None,
    max_pct_budget_per_zone: Optional[float] = None,
    zone_column: Optional[str] = None,
    time_limit_s: float = 60.0,
) -> Dict:
    """
    Résout le problème de sélection optimale via MILP (OR-Tools).

    Parameters
    ----------
    df : pd.DataFrame
        Doit contenir : GID, risk_score, longueur_km, cost
    budget_total : float
        Budget maximum en euros
    lineaire_max_km : float, optional
        Linéaire maximum en km
    weight_mode : str
        "length" = pondérer par longueur_km, "uniform" = poids 1
    excluded_gids : list, optional
    excluded_materials : list, optional
    max_pct_budget_per_zone : float, optional
    zone_column : str, optional
    time_limit_s : float
        Temps max pour le solveur

    Returns
    -------
    dict: selected_gids, status, solve_time_s, objective_value, logs
    """
    if not ORTOOLS_AVAILABLE:
        return {
            "selected_gids": [],
            "status": "ORTOOLS_NOT_AVAILABLE",
            "solve_time_s": 0,
            "objective_value": 0,
            "logs": ["OR-Tools non installé. Utiliser le solveur greedy."],
        }

    t0 = time.time()
    logs = []

    df_work = df.copy()

    # Exclusions
    if excluded_gids:
        df_work = df_work[~df_work["GID"].isin(excluded_gids)]
        logs.append(f"Exclusion de {len(excluded_gids)} GIDs")

    if excluded_materials:
        before = len(df_work)
        df_work = df_work[~df_work["MAT"].isin(excluded_materials)]
        logs.append(f"Exclusion matériaux {excluded_materials}: {before - len(df_work)} retirés")

    df_work = df_work.reset_index(drop=True)
    n = len(df_work)
    logs.append(f"Nombre de tronçons candidats: {n}")

    # Poids
    if weight_mode == "length":
        weights = df_work["longueur_km"].values
    else:
        weights = np.ones(n)

    scores = df_work["risk_score"].values
    costs = df_work["cost"].values
    lengths = df_work["longueur_km"].values
    gids = df_work["GID"].values

    # Valeurs objectif (score * weight, multipliées par 1e6 pour la précision)
    values = scores * weights * 1_000_000

    # Créer le solveur
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        solver = pywraplp.Solver.CreateSolver("CBC")

    if solver is None:
        return {
            "selected_gids": [],
            "status": "NO_SOLVER_BACKEND",
            "solve_time_s": time.time() - t0,
            "objective_value": 0,
            "logs": logs + ["Aucun backend de solveur disponible (SCIP/CBC)."],
        }

    solver.SetTimeLimit(int(time_limit_s * 1000))

    # Variables de décision x_i ∈ {0, 1}
    x = [solver.IntVar(0, 1, f"x_{i}") for i in range(n)]

    # Contrainte budget
    budget_constraint = solver.Constraint(0, budget_total, "budget")
    for i in range(n):
        budget_constraint.SetCoefficient(x[i], costs[i])
    logs.append(f"Contrainte budget: <= {budget_total:,.0f} €")

    # Contrainte linéaire (optionnel)
    if lineaire_max_km is not None:
        length_constraint = solver.Constraint(0, lineaire_max_km, "lineaire")
        for i in range(n):
            length_constraint.SetCoefficient(x[i], lengths[i])
        logs.append(f"Contrainte linéaire: <= {lineaire_max_km:,.1f} km")

    # Contraintes de zone (optionnel)
    if max_pct_budget_per_zone and zone_column and zone_column in df_work.columns:
        max_zone_budget = budget_total * max_pct_budget_per_zone
        zones = df_work[zone_column].unique()
        for zone in zones:
            zone_mask = df_work[zone_column] == zone
            zone_indices = df_work.index[zone_mask].tolist()
            if zone_indices:
                zc = solver.Constraint(0, max_zone_budget, f"zone_{zone}")
                for i in zone_indices:
                    zc.SetCoefficient(x[i], costs[i])
        logs.append(
            f"Contraintes zones ({len(zones)} zones): max {max_pct_budget_per_zone*100:.0f}% "
            f"du budget par zone ({max_zone_budget:,.0f} €)"
        )

    # Objectif : maximiser la valeur
    objective = solver.Objective()
    for i in range(n):
        objective.SetCoefficient(x[i], values[i])
    objective.SetMaximization()

    # Résoudre
    logs.append("Résolution en cours...")
    status = solver.Solve()

    solve_time = time.time() - t0

    status_map = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }
    status_str = status_map.get(status, f"UNKNOWN_{status}")

    selected_gids = []
    obj_val = 0.0

    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        for i in range(n):
            if x[i].solution_value() > 0.5:
                selected_gids.append(gids[i])
        obj_val = solver.Objective().Value() / 1_000_000

    logs.append(f"Statut: {status_str}")
    logs.append(f"Temps: {solve_time:.2f}s")
    logs.append(f"Tronçons sélectionnés: {len(selected_gids)}")
    logs.append(f"Valeur objectif: {obj_val:,.4f}")

    if selected_gids:
        sel_df = df_work[df_work["GID"].isin(selected_gids)]
        logs.append(f"Budget utilisé: {sel_df['cost'].sum():,.0f} / {budget_total:,.0f} €")
        logs.append(f"Linéaire sélectionné: {sel_df['longueur_km'].sum():,.1f} km")

    return {
        "selected_gids": selected_gids,
        "status": status_str,
        "solve_time_s": solve_time,
        "objective_value": obj_val,
        "logs": logs,
    }
