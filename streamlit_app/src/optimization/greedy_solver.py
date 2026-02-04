"""
Solveur greedy pour le plan de renouvellement.
Fallback si OR-Tools n'est pas disponible.

Algorithme : greedy par ratio score*weight / cost décroissant,
sous contraintes de budget et linéaire max.
"""
import pandas as pd
import numpy as np
import time
from typing import Optional, List, Dict


def solve_greedy(
    df: pd.DataFrame,
    budget_total: float,
    lineaire_max_km: Optional[float] = None,
    weight_mode: str = "length",
    excluded_gids: Optional[List[int]] = None,
    excluded_materials: Optional[List[str]] = None,
    max_pct_budget_per_zone: Optional[float] = None,
    zone_column: Optional[str] = None,
) -> Dict:
    """
    Solveur greedy : sélectionne les tronçons par ratio valeur/coût décroissant.

    Parameters
    ----------
    df : pd.DataFrame
        Doit contenir : GID, risk_score, longueur_km, cost
    budget_total : float
        Budget maximum
    lineaire_max_km : float, optional
        Linéaire maximum (km)
    weight_mode : str
        "length" = pondérer par longueur, "uniform" = poids 1
    excluded_gids : list, optional
    excluded_materials : list, optional
    max_pct_budget_per_zone : float, optional
    zone_column : str, optional

    Returns
    -------
    dict avec keys: selected_gids, status, solve_time_s, logs
    """
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
        logs.append(f"Exclusion matériaux {excluded_materials}: {before - len(df_work)} tronçons retirés")

    # Poids
    if weight_mode == "length":
        df_work["weight"] = df_work["longueur_km"]
    else:
        df_work["weight"] = 1.0

    # Valeur = score * weight
    df_work["value"] = df_work["risk_score"] * df_work["weight"]

    # Ratio valeur / coût
    df_work["ratio"] = df_work["value"] / np.maximum(df_work["cost"], 1.0)

    # Trier par ratio décroissant
    df_work = df_work.sort_values("ratio", ascending=False).reset_index(drop=True)

    selected = []
    budget_remaining = budget_total
    length_remaining = lineaire_max_km if lineaire_max_km else float("inf")
    zone_budgets = {}  # zone -> budget utilisé

    for _, row in df_work.iterrows():
        cost_i = row["cost"]
        length_i = row["longueur_km"]

        # Vérifier budget
        if cost_i > budget_remaining:
            continue

        # Vérifier linéaire
        if length_i > length_remaining:
            continue

        # Vérifier contrainte zone
        if max_pct_budget_per_zone and zone_column and zone_column in row.index:
            zone = row[zone_column]
            zone_used = zone_budgets.get(zone, 0.0)
            max_zone_budget = budget_total * max_pct_budget_per_zone
            if zone_used + cost_i > max_zone_budget:
                continue

        # Sélectionner
        selected.append(row["GID"])
        budget_remaining -= cost_i
        length_remaining -= length_i

        if max_pct_budget_per_zone and zone_column and zone_column in row.index:
            zone = row[zone_column]
            zone_budgets[zone] = zone_budgets.get(zone, 0.0) + cost_i

    solve_time = time.time() - t0
    logs.append(f"Greedy: {len(selected)} tronçons sélectionnés en {solve_time:.2f}s")
    logs.append(f"Budget utilisé: {budget_total - budget_remaining:,.0f} / {budget_total:,.0f}")
    if lineaire_max_km:
        logs.append(
            f"Linéaire: {lineaire_max_km - length_remaining:,.1f} / {lineaire_max_km:,.1f} km"
        )

    return {
        "selected_gids": selected,
        "status": "OPTIMAL" if selected else "NO_SOLUTION",
        "solve_time_s": solve_time,
        "logs": logs,
    }
