"""
Moteur d'optimisation MILP pour la planification des renouvellements
"""
from typing import List, Optional, Tuple
import numpy as np
from pulp import (
    LpProblem, LpMaximize, LpVariable, LpBinary,
    lpSum, LpStatus, PULP_CBC_CMD
)

from models import (
    Troncon, TronconPlanifie, ResultatOptimisation,
    KpiAnnee, Scenario
)


def calculer_score_composite(troncon: Troncon, alpha: float) -> float:
    """Calcule le score composite d'un tronçon."""
    return alpha * troncon.score_ml + (1 - alpha) * troncon.score_opportunite


def optimiser_annee(
    troncons: List[Troncon],
    budget_max: float,
    lineaire_min: float,  # en mètres
    lineaire_max: float,  # en mètres
    alpha: float = 1.0,
    time_limit: int = 60,
) -> Tuple[List[Troncon], str]:
    """
    Optimise la sélection des tronçons pour une année.

    Args:
        troncons: Liste des tronçons candidats
        budget_max: Budget maximum en euros
        lineaire_min: Linéaire minimum à renouveler en mètres
        lineaire_max: Linéaire maximum à renouveler en mètres
        alpha: Coefficient de pondération (1=100% risque ML, 0=100% opportunité)
        time_limit: Limite de temps en secondes

    Returns:
        Tuple (liste des tronçons sélectionnés, message de statut)
    """
    if len(troncons) == 0:
        return [], "Aucun tronçon candidat"

    # Vérifier cohérence des contraintes
    if lineaire_min > lineaire_max:
        return [], f"Erreur: linéaire min ({lineaire_min/1000:.1f} km) > linéaire max ({lineaire_max/1000:.1f} km)"

    # Créer le problème
    prob = LpProblem("Renouvellement_Annuel", LpMaximize)

    # Variables binaires: x_i = 1 si le tronçon i est sélectionné
    x = {t.gid: LpVariable(f"x_{t.gid}", cat=LpBinary) for t in troncons}

    # Précalculer les scores composites
    scores = {t.gid: calculer_score_composite(t, alpha) for t in troncons}

    # Fonction objectif: maximiser Σ (score_i × longueur_i × x_i)
    prob += lpSum([
        scores[t.gid] * t.longueur * x[t.gid]
        for t in troncons
    ]), "Valeur_Totale"

    # Contrainte 1: Budget
    prob += lpSum([
        t.cout_total * x[t.gid]
        for t in troncons
    ]) <= budget_max, "Budget_Max"

    # Contrainte 2: Linéaire minimum
    prob += lpSum([
        t.longueur * x[t.gid]
        for t in troncons
    ]) >= lineaire_min, "Lineaire_Min"

    # Contrainte 3: Linéaire maximum
    prob += lpSum([
        t.longueur * x[t.gid]
        for t in troncons
    ]) <= lineaire_max, "Lineaire_Max"

    # Résoudre
    solver = PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    prob.solve(solver)

    # Analyser le résultat
    status = LpStatus[prob.status]

    if status == "Infeasible":
        # Diagnostiquer le problème
        cout_total_dispo = sum(t.cout_total for t in troncons)
        lineaire_total_dispo = sum(t.longueur for t in troncons)

        if lineaire_total_dispo < lineaire_min:
            return [], f"Infaisable: linéaire disponible ({lineaire_total_dispo/1000:.1f} km) < minimum requis ({lineaire_min/1000:.1f} km)"

        # Estimer le budget nécessaire pour le linéaire min
        troncons_par_ratio = sorted(troncons, key=lambda t: t.cout_total / t.longueur)
        budget_estime = 0
        lineaire_cumul = 0
        for t in troncons_par_ratio:
            if lineaire_cumul >= lineaire_min:
                break
            budget_estime += t.cout_total
            lineaire_cumul += t.longueur

        if budget_estime > budget_max:
            return [], f"Infaisable: budget ({budget_max/1e6:.2f} M€) insuffisant pour le linéaire min. Besoin estimé: {budget_estime/1e6:.2f} M€"

        return [], f"Infaisable: contraintes incompatibles (statut: {status})"

    if status != "Optimal":
        return [], f"Optimisation non résolue (statut: {status})"

    # Extraire les tronçons sélectionnés
    selection = [t for t in troncons if x[t.gid].value() == 1]

    return selection, "OK"


def optimiser_horizon(
    troncons: List[Troncon],
    scenario: Scenario,
    lineaire_total_km: float,
) -> ResultatOptimisation:
    """
    Optimise sur tout l'horizon temporel.

    Args:
        troncons: Liste de tous les tronçons
        scenario: Paramètres du scénario
        lineaire_total_km: Linéaire total du réseau en km

    Returns:
        ResultatOptimisation avec tous les résultats
    """
    # Convertir linéaire en mètres
    lineaire_total_m = lineaire_total_km * 1000
    lineaire_min_m = lineaire_total_m * (scenario.lineaire_min_pct / 100)
    lineaire_max_m = lineaire_total_m * (scenario.lineaire_max_pct / 100)

    # Calculer risque initial (moyenne pondérée par longueur)
    total_longueur = sum(t.longueur for t in troncons)
    risque_initial = sum(t.score_ml * t.longueur for t in troncons) / total_longueur if total_longueur > 0 else 0

    # Liste des tronçons restants à traiter
    troncons_restants = list(troncons)
    troncons_planifies = []
    kpis_par_annee = []
    budget_total = 0
    lineaire_total = 0

    annee_debut = 2024  # Année de départ

    for annee_idx in range(scenario.horizon_ans):
        annee = annee_debut + annee_idx

        if len(troncons_restants) == 0:
            # Plus de tronçons à renouveler
            kpis_par_annee.append(KpiAnnee(
                annee=annee,
                budget_consomme=0,
                lineaire_km=0,
                nb_troncons=0,
                risque_moyen=0
            ))
            continue

        # Optimiser cette année
        selection, message = optimiser_annee(
            troncons=troncons_restants,
            budget_max=scenario.budget_annuel,
            lineaire_min=lineaire_min_m,
            lineaire_max=lineaire_max_m,
            alpha=scenario.alpha,
        )

        if message != "OK" and annee_idx == 0:
            # Échec dès la première année
            return ResultatOptimisation(
                troncons_planifies=[],
                budget_total_consomme=0,
                lineaire_total_km=0,
                risque_initial=risque_initial,
                risque_residuel=risque_initial,
                kpis_par_annee=[],
                message=message,
                succes=False
            )

        # Si on ne peut pas atteindre le minimum, on prend ce qu'on peut
        if len(selection) == 0 and message != "OK":
            # Essayer sans contrainte de minimum
            selection, _ = optimiser_annee(
                troncons=troncons_restants,
                budget_max=scenario.budget_annuel,
                lineaire_min=0,
                lineaire_max=lineaire_max_m,
                alpha=scenario.alpha,
            )

        # Calculer les KPIs de l'année
        budget_annee = sum(t.cout_total for t in selection)
        lineaire_annee = sum(t.longueur for t in selection) / 1000  # en km
        risque_annee = np.mean([t.score_ml for t in selection]) if selection else 0

        kpis_par_annee.append(KpiAnnee(
            annee=annee,
            budget_consomme=budget_annee,
            lineaire_km=lineaire_annee,
            nb_troncons=len(selection),
            risque_moyen=risque_annee
        ))

        # Ajouter aux planifiés
        for t in selection:
            score_comp = calculer_score_composite(t, scenario.alpha)
            troncons_planifies.append(TronconPlanifie(
                troncon=t,
                annee_renouvellement=annee,
                score_composite=score_comp
            ))

        budget_total += budget_annee
        lineaire_total += lineaire_annee

        # Retirer les tronçons sélectionnés des restants
        gids_selectionnes = {t.gid for t in selection}
        troncons_restants = [t for t in troncons_restants if t.gid not in gids_selectionnes]

    # Calculer risque résiduel
    if troncons_restants:
        longueur_restante = sum(t.longueur for t in troncons_restants)
        risque_residuel = sum(t.score_ml * t.longueur for t in troncons_restants) / longueur_restante if longueur_restante > 0 else 0
    else:
        risque_residuel = 0

    return ResultatOptimisation(
        troncons_planifies=troncons_planifies,
        budget_total_consomme=budget_total,
        lineaire_total_km=lineaire_total,
        risque_initial=risque_initial,
        risque_residuel=risque_residuel,
        kpis_par_annee=kpis_par_annee,
        message=f"Optimisation réussie: {len(troncons_planifies)} tronçons planifiés sur {scenario.horizon_ans} ans",
        succes=True
    )


def simuler_scenario(
    troncons: List[Troncon],
    scenario: Scenario,
) -> ResultatOptimisation:
    """
    Point d'entrée principal pour simuler un scénario.

    Args:
        troncons: Liste de tous les tronçons du réseau
        scenario: Paramètres du scénario à simuler

    Returns:
        ResultatOptimisation
    """
    # Filtrer les tronçons en service
    troncons_actifs = [t for t in troncons if t.statut == "EN SERVICE"]

    if len(troncons_actifs) == 0:
        return ResultatOptimisation(
            troncons_planifies=[],
            budget_total_consomme=0,
            lineaire_total_km=0,
            risque_initial=0,
            risque_residuel=0,
            kpis_par_annee=[],
            message="Aucun tronçon en service",
            succes=False
        )

    # Calculer linéaire total
    lineaire_total_km = sum(t.longueur for t in troncons_actifs) / 1000

    # Lancer l'optimisation
    return optimiser_horizon(
        troncons=troncons_actifs,
        scenario=scenario,
        lineaire_total_km=lineaire_total_km
    )
