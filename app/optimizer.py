"""
Moteur d'optimisation sous contrainte budgetaire
================================================

Optimise la selection des conduites a renouveler pour maximiser
la reduction du risque sous contrainte de budget.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class OptimizationResult:
    """Resultat d'une optimisation."""
    selected_ids: List[str]
    total_cost: float
    total_length: float
    risk_before: float
    risk_after: float
    n_failures_avoided: float
    budget_used_pct: float
    details: pd.DataFrame


class BudgetOptimizer:
    """
    Optimiseur de renouvellement sous contrainte budgetaire.

    Utilise une approche gloutonne (greedy) basee sur le ratio
    risque_reduit / cout pour maximiser l'efficacite budgetaire.
    """

    # Cout moyen de renouvellement par metre lineaire (euros/m)
    DEFAULT_COST_PER_METER = 500  # €/m

    # Cout par diametre (€/m)
    COST_BY_DIAMETER = {
        50: 300,
        80: 350,
        100: 400,
        150: 500,
        200: 600,
        250: 750,
        300: 900,
        400: 1200,
        500: 1500,
    }

    def __init__(self, df: pd.DataFrame, scores: np.ndarray):
        """
        Args:
            df: DataFrame avec les conduites (doit avoir 'GID', 'longueur', 'diametre')
            scores: Scores de risque calibres (probabilites)
        """
        self.df = df.copy()
        self.df['risk_score'] = scores
        self._compute_costs()

    def _compute_costs(self):
        """Calcule le cout de renouvellement de chaque conduite."""
        def get_cost_per_meter(diam):
            # Trouver le diametre le plus proche
            diams = list(self.COST_BY_DIAMETER.keys())
            closest = min(diams, key=lambda x: abs(x - diam))
            return self.COST_BY_DIAMETER[closest]

        if 'diametre' in self.df.columns:
            self.df['cost_per_m'] = self.df['diametre'].apply(get_cost_per_meter)
        else:
            self.df['cost_per_m'] = self.DEFAULT_COST_PER_METER

        self.df['renewal_cost'] = self.df['longueur'] * self.df['cost_per_m']

    def optimize(
        self,
        budget: float,
        min_diameter: int = 0,
        max_diameter: int = 9999,
        materials: List[str] = None,
        min_age: int = 0,
        max_length: float = None,
    ) -> OptimizationResult:
        """
        Optimise la selection des conduites sous contrainte budgetaire.

        Args:
            budget: Budget total disponible (euros)
            min_diameter: Diametre minimum a considerer
            max_diameter: Diametre maximum a considerer
            materials: Liste des materiaux a considerer (None = tous)
            min_age: Age minimum des conduites
            max_length: Longueur max par conduite (None = pas de limite)

        Returns:
            OptimizationResult avec les conduites selectionnees
        """
        # Filtrer les conduites eligibles
        mask = pd.Series(True, index=self.df.index)

        if 'diametre' in self.df.columns:
            mask &= (self.df['diametre'] >= min_diameter) & (self.df['diametre'] <= max_diameter)

        if materials and 'materiau' in self.df.columns:
            mask &= self.df['materiau'].isin(materials)

        if 'age_at_freeze' in self.df.columns:
            mask &= self.df['age_at_freeze'] >= min_age

        if max_length and 'longueur' in self.df.columns:
            mask &= self.df['longueur'] <= max_length

        df_eligible = self.df[mask].copy()

        if len(df_eligible) == 0:
            return OptimizationResult(
                selected_ids=[],
                total_cost=0,
                total_length=0,
                risk_before=self.df['risk_score'].sum(),
                risk_after=self.df['risk_score'].sum(),
                n_failures_avoided=0,
                budget_used_pct=0,
                details=pd.DataFrame()
            )

        # Calculer l'efficacite: risque reduit par euro depense
        # On suppose que renouveler une conduite reduit son risque a ~0
        df_eligible['risk_reduction'] = df_eligible['risk_score']
        df_eligible['efficiency'] = df_eligible['risk_reduction'] / df_eligible['renewal_cost']

        # Trier par efficacite decroissante (greedy)
        df_sorted = df_eligible.sort_values('efficiency', ascending=False)

        # Selection gloutonne
        selected = []
        total_cost = 0

        for idx, row in df_sorted.iterrows():
            if total_cost + row['renewal_cost'] <= budget:
                selected.append(idx)
                total_cost += row['renewal_cost']

        # Calculer les metriques
        df_selected = df_sorted.loc[selected] if selected else pd.DataFrame()

        risk_before = self.df['risk_score'].sum()
        risk_reduced = df_selected['risk_score'].sum() if len(df_selected) > 0 else 0
        risk_after = risk_before - risk_reduced

        total_length = df_selected['longueur'].sum() if len(df_selected) > 0 else 0

        # Preparer le detail
        if len(df_selected) > 0:
            details = df_selected[['GID', 'longueur', 'diametre', 'risk_score',
                                   'renewal_cost', 'efficiency']].copy()
            details = details.rename(columns={
                'longueur': 'Longueur (m)',
                'diametre': 'Diametre (mm)',
                'risk_score': 'Score Risque',
                'renewal_cost': 'Cout (€)',
                'efficiency': 'Efficacite'
            })
        else:
            details = pd.DataFrame()

        return OptimizationResult(
            selected_ids=[self.df.loc[i, 'GID'] for i in selected] if selected else [],
            total_cost=total_cost,
            total_length=total_length,
            risk_before=risk_before,
            risk_after=risk_after,
            n_failures_avoided=risk_reduced,
            budget_used_pct=(total_cost / budget * 100) if budget > 0 else 0,
            details=details
        )

    def compare_scenarios(
        self,
        budgets: List[float],
        **kwargs
    ) -> pd.DataFrame:
        """
        Compare plusieurs scenarios budgetaires.

        Args:
            budgets: Liste des budgets a comparer
            **kwargs: Arguments supplementaires pour optimize()

        Returns:
            DataFrame comparatif
        """
        results = []

        for budget in budgets:
            result = self.optimize(budget, **kwargs)
            results.append({
                'Budget (€)': budget,
                'Budget (M€)': budget / 1_000_000,
                'Conduites': len(result.selected_ids),
                'Lineaire (km)': result.total_length / 1000,
                'Risque Initial': result.risk_before,
                'Risque Residuel': result.risk_after,
                'Reduction (%)': (1 - result.risk_after / result.risk_before) * 100 if result.risk_before > 0 else 0,
                'Defaillances Evitees': result.n_failures_avoided,
                'Budget Utilise (%)': result.budget_used_pct,
            })

        return pd.DataFrame(results)

    def what_if_analysis(
        self,
        base_budget: float,
        variations: List[float] = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3],
        **kwargs
    ) -> pd.DataFrame:
        """
        Analyse what-if: impact des variations budgetaires.

        Args:
            base_budget: Budget de reference
            variations: Liste des variations (ex: -0.1 = -10%)

        Returns:
            DataFrame avec les resultats
        """
        budgets = [base_budget * (1 + v) for v in variations]
        labels = [f"{v*100:+.0f}%" for v in variations]

        results = self.compare_scenarios(budgets, **kwargs)
        results['Variation'] = labels

        return results


def compute_risk_score(value: float) -> int:
    """Convertit une probabilite en score sur 100."""
    # Normalisation: 0-0.1 -> 0-100
    return min(100, int(value * 1000))
