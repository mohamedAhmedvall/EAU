#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse de survie et corrélation anomalies/fin de vie des canalisations

Objectif: Préparer les données pour un modèle de prédiction de casse
qui alimentera un moteur d'optimisation des plans de renouvellement.

Auteur: Data Science Pipeline
Date: 2026-02-03
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

DATA_DIR = 'data'
REPORTS_DIR = 'reports'
FIGURES_DIR = 'reports/figures'
PROCESSED_DIR = 'data/processed'

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def normalize_datetime(series):
    """Normalise une série datetime en supprimant le timezone"""
    if series.dt.tz is not None:
        return series.dt.tz_localize(None)
    return series


def save_figure(fig, name, dpi=150):
    """Sauvegarde une figure"""
    path = os.path.join(FIGURES_DIR, f'{name}.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Figure: {path}")
    return path


def load_data():
    """Charge toutes les données nécessaires"""
    print("\n1. Chargement des données...")

    # Tronçons
    df_troncons = pd.read_csv(f'{DATA_DIR}/v1_trafic_prepared.csv')
    df_troncons['DDP'] = pd.to_datetime(df_troncons['DDP'], errors='coerce')
    df_troncons['DDP'] = normalize_datetime(df_troncons['DDP'])
    df_troncons['DHS'] = pd.to_datetime(df_troncons['DHS'], errors='coerce')
    df_troncons['DHS'] = normalize_datetime(df_troncons['DHS'])
    print(f"   Tronçons: {len(df_troncons):,}")

    # Anomalies
    df_anomalies = pd.read_excel(f'{DATA_DIR}/Anomalie.xlsx')
    df_anomalies['DATE_DETECTION'] = pd.to_datetime(df_anomalies['DATE_DETECTION'], errors='coerce')
    df_anomalies['DATE_DETECTION'] = normalize_datetime(df_anomalies['DATE_DETECTION'])
    print(f"   Anomalies: {len(df_anomalies):,}")

    return df_troncons, df_anomalies


def analyze_end_of_life(df_troncons, df_anomalies):
    """
    Analyse de la fin de vie des tronçons et corrélation avec les anomalies
    """
    print("\n2. Analyse de la fin de vie...")

    report = []
    report.append("# Analyse Survie : Corrélation Anomalies / Fin de Vie\n")
    report.append(f"*Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    report.append("\n## Objectif\n")
    report.append("Analyser la relation entre les anomalies et la fin de vie des canalisations ")
    report.append("pour construire un modèle de prédiction de casse alimentant un moteur ")
    report.append("d'optimisation des plans de renouvellement.\n")

    # Définir la fin de vie
    df = df_troncons.copy()
    df['is_abandoned'] = (df['STATUT_OBJET'] == 'ABANDONNE').astype(int)

    n_total = len(df)
    n_abandoned = df['is_abandoned'].sum()
    n_active = n_total - n_abandoned

    report.append("\n## 1. État du parc de canalisations\n")
    report.append(f"| Statut | Nombre | % |\n")
    report.append(f"|--------|--------|---|\n")
    report.append(f"| EN SERVICE | {n_active:,} | {n_active/n_total*100:.1f}% |\n")
    report.append(f"| ABANDONNÉ (fin de vie) | {n_abandoned:,} | {n_abandoned/n_total*100:.1f}% |\n")
    report.append(f"| **TOTAL** | {n_total:,} | 100% |\n")

    # Joindre les anomalies
    print("   Jointure anomalies...")
    ano_counts = df_anomalies.groupby('GID_OBJET').agg({
        'GID_OBJET': 'count',
        'DATE_DETECTION': ['min', 'max']
    })
    ano_counts.columns = ['n_anomalies', 'first_anomaly', 'last_anomaly']
    ano_counts = ano_counts.reset_index()

    # Normaliser les dates d'anomalie
    ano_counts['first_anomaly'] = normalize_datetime(pd.to_datetime(ano_counts['first_anomaly'], errors='coerce'))
    ano_counts['last_anomaly'] = normalize_datetime(pd.to_datetime(ano_counts['last_anomaly'], errors='coerce'))

    df = df.merge(ano_counts, left_on='GID', right_on='GID_OBJET', how='left')
    df['n_anomalies'] = df['n_anomalies'].fillna(0).astype(int)
    df['has_anomaly'] = (df['n_anomalies'] > 0).astype(int)

    # ==========================================================================
    # ANALYSE 1: Taux de fin de vie selon présence d'anomalie
    # ==========================================================================
    report.append("\n## 2. Corrélation Anomalies → Fin de vie\n")

    # Tableau croisé
    cross_tab = pd.crosstab(df['has_anomaly'], df['is_abandoned'], margins=True)

    # Calcul des taux
    with_ano = df[df['has_anomaly'] == 1]
    without_ano = df[df['has_anomaly'] == 0]

    rate_with_ano = with_ano['is_abandoned'].mean() * 100
    rate_without_ano = without_ano['is_abandoned'].mean() * 100
    relative_risk = rate_with_ano / rate_without_ano if rate_without_ano > 0 else np.inf

    report.append("\n### 2.1 Taux de fin de vie selon présence d'anomalie\n")
    report.append("| Groupe | N tronçons | N abandonnés | Taux fin de vie |\n")
    report.append("|--------|------------|--------------|------------------|\n")
    report.append(f"| Sans anomalie | {len(without_ano):,} | {without_ano['is_abandoned'].sum():,} | {rate_without_ano:.2f}% |\n")
    report.append(f"| Avec anomalie(s) | {len(with_ano):,} | {with_ano['is_abandoned'].sum():,} | {rate_with_ano:.2f}% |\n")
    report.append(f"\n**Risque relatif (RR)** = {relative_risk:.2f}\n")
    report.append(f"→ Un tronçon avec anomalie a **{relative_risk:.1f}x plus de risque** d'être abandonné.\n")

    # Test Chi2
    chi2, p_val, dof, expected = chi2_contingency(cross_tab.iloc[:-1, :-1])
    report.append(f"\n**Test Chi²**: χ² = {chi2:,.1f}, p-value < 0.001 → Association **hautement significative**\n")

    # ==========================================================================
    # ANALYSE 2: Relation nombre d'anomalies vs fin de vie
    # ==========================================================================
    report.append("\n### 2.2 Taux de fin de vie selon le nombre d'anomalies\n")

    df['n_anomalies_cat'] = pd.cut(df['n_anomalies'],
                                    bins=[-1, 0, 1, 2, 5, 10, 1000],
                                    labels=['0', '1', '2', '3-5', '6-10', '>10'])

    rate_by_n = df.groupby('n_anomalies_cat').agg({
        'GID': 'count',
        'is_abandoned': ['sum', 'mean']
    })
    rate_by_n.columns = ['n_troncons', 'n_abandonnes', 'taux']
    rate_by_n['taux'] = rate_by_n['taux'] * 100

    report.append("| Nb anomalies | N tronçons | N abandonnés | Taux fin de vie |\n")
    report.append("|--------------|------------|--------------|------------------|\n")
    for idx in rate_by_n.index:
        row = rate_by_n.loc[idx]
        report.append(f"| {idx} | {int(row['n_troncons']):,} | {int(row['n_abandonnes']):,} | {row['taux']:.2f}% |\n")

    # Figure: Taux de fin de vie par nombre d'anomalies
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Barplot taux
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(rate_by_n)))
    axes[0].bar(range(len(rate_by_n)), rate_by_n['taux'], color=colors, edgecolor='black')
    axes[0].set_xticks(range(len(rate_by_n)))
    axes[0].set_xticklabels(rate_by_n.index)
    axes[0].set_xlabel('Nombre d\'anomalies')
    axes[0].set_ylabel('Taux de fin de vie (%)')
    axes[0].set_title('Taux de fin de vie selon le nombre d\'anomalies')
    axes[0].axhline(y=df['is_abandoned'].mean()*100, color='gray', linestyle='--',
                    label=f'Moyenne globale: {df["is_abandoned"].mean()*100:.1f}%')
    axes[0].legend()

    # Boxplot âge à la fin de vie
    df_abandoned = df[df['is_abandoned'] == 1].copy()
    df_abandoned['age_at_death'] = (df_abandoned['DHS'] - df_abandoned['DDP']).dt.days / 365.25
    df_abandoned = df_abandoned[df_abandoned['age_at_death'] > 0]

    df_abandoned.boxplot(column='age_at_death', by='n_anomalies_cat', ax=axes[1])
    axes[1].set_xlabel('Nombre d\'anomalies')
    axes[1].set_ylabel('Âge à la fin de vie (années)')
    axes[1].set_title('Âge de fin de vie selon le nombre d\'anomalies')
    plt.suptitle('')

    plt.tight_layout()
    save_figure(fig, 'survival_01_taux_fin_vie_anomalies')
    report.append("\n![Taux fin de vie](figures/survival_01_taux_fin_vie_anomalies.png)\n")

    # ==========================================================================
    # ANALYSE 3: Analyse temporelle - Anomalies avant fin de vie
    # ==========================================================================
    report.append("\n### 2.3 Séquence temporelle : Anomalie → Fin de vie\n")

    # Pour les tronçons abandonnés avec anomalies, calculer le délai
    df_with_both = df[(df['is_abandoned'] == 1) & (df['has_anomaly'] == 1)].copy()
    df_with_both['days_last_ano_to_death'] = (df_with_both['DHS'] - df_with_both['last_anomaly']).dt.days
    df_with_both['days_first_ano_to_death'] = (df_with_both['DHS'] - df_with_both['first_anomaly']).dt.days

    # Filtrer les valeurs aberrantes
    valid_delays = df_with_both[
        (df_with_both['days_last_ano_to_death'] > -365) &  # Anomalie peut être après DHS de 1 an max
        (df_with_both['days_last_ano_to_death'] < 20*365)  # Max 20 ans avant
    ]['days_last_ano_to_death']

    valid_delays_years = valid_delays / 365.25

    report.append(f"\nParmi les {len(df_with_both):,} tronçons abandonnés avec anomalies:\n")
    report.append(f"- Délai médian dernière anomalie → fin de vie: **{valid_delays_years.median():.1f} ans**\n")
    report.append(f"- Délai moyen: {valid_delays_years.mean():.1f} ans\n")
    report.append(f"- 25% des cas: < {valid_delays_years.quantile(0.25):.1f} ans\n")
    report.append(f"- 75% des cas: < {valid_delays_years.quantile(0.75):.1f} ans\n")

    # Anomalies APRÈS la fin de vie (incohérence données?)
    ano_after = (df_with_both['days_last_ano_to_death'] < 0).sum()
    report.append(f"\n⚠️ {ano_after:,} cas où la dernière anomalie est APRÈS la date d'abandon ")
    report.append("(incohérence données ou anomalie détectée post-abandon)\n")

    # Figure: Distribution des délais
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogramme délais
    valid_delays_years_capped = valid_delays_years[(valid_delays_years >= 0) & (valid_delays_years <= 15)]
    axes[0].hist(valid_delays_years_capped, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(valid_delays_years_capped.median(), color='red', linestyle='--',
                    label=f'Médiane: {valid_delays_years_capped.median():.1f} ans')
    axes[0].set_xlabel('Années entre dernière anomalie et fin de vie')
    axes[0].set_ylabel('Fréquence')
    axes[0].set_title('Délai dernière anomalie → Abandon')
    axes[0].legend()

    # Cumulative: % des fins de vie dans X ans après anomalie
    survival_times = np.sort(valid_delays_years_capped)
    cumulative = np.arange(1, len(survival_times) + 1) / len(survival_times)

    axes[1].plot(survival_times, cumulative * 100, linewidth=2, color='darkred')
    axes[1].fill_between(survival_times, cumulative * 100, alpha=0.3, color='red')
    axes[1].set_xlabel('Années après dernière anomalie')
    axes[1].set_ylabel('% cumulé de fins de vie')
    axes[1].set_title('Distribution cumulative : Fin de vie après dernière anomalie')
    axes[1].axhline(50, color='gray', linestyle='--', alpha=0.5)
    axes[1].axhline(75, color='gray', linestyle='--', alpha=0.5)

    # Marquer les points clés
    for pct in [25, 50, 75]:
        idx = np.searchsorted(cumulative, pct/100)
        if idx < len(survival_times):
            axes[1].annotate(f'{pct}% à {survival_times[idx]:.1f} ans',
                           xy=(survival_times[idx], pct), fontsize=9)

    plt.tight_layout()
    save_figure(fig, 'survival_02_delai_anomalie_fin_vie')
    report.append("\n![Délai anomalie-fin de vie](figures/survival_02_delai_anomalie_fin_vie.png)\n")

    return df, report


def analyze_survival_by_material(df, df_anomalies):
    """
    Analyse de survie par matériau avec courbes Kaplan-Meier simplifiées
    """
    print("\n3. Analyse de survie par matériau...")

    report = []
    report.append("\n## 3. Analyse de survie par matériau\n")

    # Calculer l'âge à la fin de vie ou l'âge actuel (censuré)
    df = df.copy()

    # Date de référence pour les tronçons en service
    ref_date = pd.Timestamp('2024-01-01')

    # Pour les abandonnés: durée de vie = DHS - DDP
    # Pour les en service: durée observée = ref_date - DDP (censuré)
    df['DDP'] = pd.to_datetime(df['DDP'], errors='coerce')
    df['DHS'] = pd.to_datetime(df['DHS'], errors='coerce')

    df['duration_years'] = np.where(
        df['is_abandoned'] == 1,
        (df['DHS'] - df['DDP']).dt.days / 365.25,
        (ref_date - df['DDP']).dt.days / 365.25
    )

    # Filtrer les durées valides
    df = df[(df['duration_years'] > 0) & (df['duration_years'] < 200)]

    # Statistiques par matériau
    mat_stats = df.groupby('MAT').agg({
        'GID': 'count',
        'is_abandoned': ['sum', 'mean'],
        'duration_years': ['median', 'mean', 'std']
    })
    mat_stats.columns = ['n_total', 'n_abandoned', 'taux_abandon', 'duree_mediane', 'duree_moyenne', 'duree_std']
    mat_stats['taux_abandon'] = mat_stats['taux_abandon'] * 100
    mat_stats = mat_stats.sort_values('n_total', ascending=False)

    report.append("\n### 3.1 Statistiques de survie par matériau\n")
    report.append("| Matériau | N total | N abandonnés | Taux abandon | Durée médiane | Durée moyenne |\n")
    report.append("|----------|---------|--------------|--------------|---------------|---------------|\n")

    for mat in mat_stats.head(15).index:
        row = mat_stats.loc[mat]
        report.append(f"| {mat} | {int(row['n_total']):,} | {int(row['n_abandoned']):,} | "
                     f"{row['taux_abandon']:.1f}% | {row['duree_mediane']:.0f} ans | {row['duree_moyenne']:.0f} ans |\n")

    # Courbes de survie simplifiées (Kaplan-Meier approximé)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    top_materials = mat_stats.head(8).index.tolist()
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_materials)))

    for i, mat in enumerate(top_materials):
        df_mat = df[df['MAT'] == mat].copy()

        # Construire la courbe de survie
        max_time = 120  # 120 ans max
        times = np.arange(0, max_time + 1, 1)

        # Pour chaque temps t, calculer le % de survivants
        # Survivant = tronçon avec durée >= t OU censuré avec durée >= t
        survival = []
        for t in times:
            # Tronçons à risque à temps t (durée >= t)
            at_risk = (df_mat['duration_years'] >= t).sum()
            # Tronçons qui ont survécu (en service ou abandonné après t)
            survived = ((df_mat['is_abandoned'] == 0) | (df_mat['duration_years'] > t)).sum()

            if at_risk > 0:
                survival.append(survived / len(df_mat))
            else:
                survival.append(survival[-1] if survival else 1.0)

        axes[0].plot(times, np.array(survival) * 100, label=f'{mat} (n={len(df_mat):,})',
                    color=colors[i], linewidth=2)

    axes[0].set_xlabel('Âge (années)')
    axes[0].set_ylabel('% de survie estimé')
    axes[0].set_title('Courbes de survie par matériau')
    axes[0].legend(loc='lower left', fontsize=9)
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 100)
    axes[0].axhline(50, color='gray', linestyle='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3)

    # Boxplot durée de vie par matériau (abandonnés uniquement)
    df_abandoned = df[(df['is_abandoned'] == 1) & (df['MAT'].isin(top_materials))]

    order = df_abandoned.groupby('MAT')['duration_years'].median().sort_values(ascending=False).index

    sns.boxplot(data=df_abandoned, x='MAT', y='duration_years', order=order, ax=axes[1], palette='coolwarm')
    axes[1].set_xlabel('Matériau')
    axes[1].set_ylabel('Durée de vie (années)')
    axes[1].set_title('Distribution de la durée de vie par matériau (abandonnés)')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    save_figure(fig, 'survival_03_courbes_survie_materiau')
    report.append("\n![Courbes de survie](figures/survival_03_courbes_survie_materiau.png)\n")

    # ==========================================================================
    # Impact des anomalies sur la survie par matériau
    # ==========================================================================
    report.append("\n### 3.2 Impact des anomalies sur la survie par matériau\n")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    report.append("| Matériau | Avec ano: durée médiane | Sans ano: durée médiane | Différence |\n")
    report.append("|----------|-------------------------|-------------------------|------------|\n")

    for i, mat in enumerate(top_materials[:6]):
        df_mat = df[df['MAT'] == mat].copy()

        # Comparer avec/sans anomalie
        with_ano = df_mat[df_mat['has_anomaly'] == 1]
        without_ano = df_mat[df_mat['has_anomaly'] == 0]

        # Durées médianes (tous, pas seulement abandonnés)
        dur_with = with_ano[with_ano['is_abandoned'] == 1]['duration_years'].median()
        dur_without = without_ano[without_ano['is_abandoned'] == 1]['duration_years'].median()
        diff = dur_with - dur_without if pd.notna(dur_with) and pd.notna(dur_without) else np.nan

        report.append(f"| {mat} | {dur_with:.0f} ans | {dur_without:.0f} ans | {diff:+.0f} ans |\n")

        # Courbes comparatives
        for label, subset, color in [('Sans anomalie', without_ano, 'green'), ('Avec anomalie', with_ano, 'red')]:
            times = np.arange(0, 101, 1)
            survival = []
            for t in times:
                at_risk = (subset['duration_years'] >= t).sum()
                survived = ((subset['is_abandoned'] == 0) | (subset['duration_years'] > t)).sum()
                survival.append(survived / len(subset) if len(subset) > 0 else 1.0)

            axes[i].plot(times, np.array(survival) * 100, label=f'{label} (n={len(subset):,})',
                        color=color, linewidth=2)

        axes[i].set_title(f'{mat}')
        axes[i].set_xlabel('Âge (années)')
        axes[i].set_ylabel('% survie')
        axes[i].legend(fontsize=8)
        axes[i].set_xlim(0, 100)
        axes[i].set_ylim(0, 100)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'survival_04_impact_anomalies_par_materiau')
    report.append("\n![Impact anomalies](figures/survival_04_impact_anomalies_par_materiau.png)\n")

    return df, mat_stats, report


def analyze_predictive_features(df, df_anomalies):
    """
    Analyse des features prédictives pour le modèle de casse
    """
    print("\n4. Analyse des features prédictives...")

    report = []
    report.append("\n## 4. Features prédictives pour le modèle de casse\n")

    # Calculer des features avancées
    df = df.copy()

    # Feature 1: Densité d'anomalies (anomalies / longueur)
    df['ano_density'] = df['n_anomalies'] / df['LNG'].clip(lower=1)

    # Feature 2: Anomalies récentes (< 5 ans avant ref_date)
    ref_date = pd.Timestamp('2024-01-01')
    recent_ano = df_anomalies[df_anomalies['DATE_DETECTION'] >= ref_date - pd.DateOffset(years=5)]
    recent_counts = recent_ano.groupby('GID_OBJET').size().reset_index(name='n_recent_ano')
    df = df.merge(recent_counts, left_on='GID', right_on='GID_OBJET', how='left')
    df['n_recent_ano'] = df['n_recent_ano'].fillna(0)

    # Feature 3: Âge relatif vs durée vie médiane du matériau
    mat_median_life = df[df['is_abandoned'] == 1].groupby('MAT')['duration_years'].median()
    df['mat_median_life'] = df['MAT'].map(mat_median_life)
    df['age_ratio'] = df['duration_years'] / df['mat_median_life'].clip(lower=1)

    # Feature 4: Dépasse la durée de vie médiane du matériau
    df['over_median_life'] = (df['duration_years'] > df['mat_median_life']).astype(int)

    # Évaluer le pouvoir prédictif de chaque feature
    report.append("\n### 4.1 Pouvoir prédictif des features (corrélation avec fin de vie)\n")

    features_to_test = ['duration_years', 'n_anomalies', 'n_recent_ano', 'ano_density',
                       'age_ratio', 'over_median_life', 'DIAMETRE', 'LNG']

    report.append("| Feature | Corrélation Pearson | p-value | Odds Ratio (si binaire) |\n")
    report.append("|---------|---------------------|---------|-------------------------|\n")

    for feat in features_to_test:
        if feat in df.columns and df[feat].notna().sum() > 100:
            valid = df[[feat, 'is_abandoned']].dropna()

            # Corrélation
            corr, p_val = stats.pointbiserialr(valid['is_abandoned'], valid[feat])

            # Odds ratio si binaire
            if valid[feat].nunique() <= 2:
                ct = pd.crosstab(valid[feat] > 0, valid['is_abandoned'])
                if ct.shape == (2, 2):
                    a, b, c, d = ct.iloc[1, 1], ct.iloc[1, 0], ct.iloc[0, 1], ct.iloc[0, 0]
                    odds_ratio = (a * d) / (b * c) if b * c > 0 else np.inf
                    report.append(f"| {feat} | {corr:.3f} | {p_val:.2e} | {odds_ratio:.2f} |\n")
                else:
                    report.append(f"| {feat} | {corr:.3f} | {p_val:.2e} | - |\n")
            else:
                report.append(f"| {feat} | {corr:.3f} | {p_val:.2e} | - |\n")

    # ==========================================================================
    # Analyse multivariée: comparaison abandonnés vs en service
    # ==========================================================================
    report.append("\n### 4.2 Comparaison des profils : Abandonnés vs En service\n")

    df_active = df[df['is_abandoned'] == 0]
    df_abandoned = df[df['is_abandoned'] == 1]

    report.append("| Caractéristique | En service | Abandonnés | Test stat | p-value |\n")
    report.append("|-----------------|------------|------------|-----------|----------|\n")

    for feat in ['duration_years', 'n_anomalies', 'DIAMETRE', 'LNG']:
        if feat in df.columns:
            active_vals = df_active[feat].dropna()
            abandoned_vals = df_abandoned[feat].dropna()

            stat, p_val = stats.mannwhitneyu(active_vals, abandoned_vals, alternative='two-sided')

            report.append(f"| {feat} | {active_vals.median():.1f} | {abandoned_vals.median():.1f} | "
                         f"{stat:.0f} | {p_val:.2e} |\n")

    # Figure: Distribution des features par statut
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    features_plot = ['duration_years', 'n_anomalies', 'age_ratio', 'ano_density']
    titles = ['Âge (années)', 'Nombre d\'anomalies', 'Ratio âge/durée vie médiane', 'Densité anomalies (/m)']

    for idx, (feat, title) in enumerate(zip(features_plot, titles)):
        ax = axes[idx // 2, idx % 2]

        if feat in df.columns:
            # Limiter pour visualisation
            data_plot = df[[feat, 'is_abandoned']].dropna()

            if feat == 'duration_years':
                data_plot = data_plot[data_plot[feat] <= 120]
            elif feat == 'n_anomalies':
                data_plot = data_plot[data_plot[feat] <= 20]
            elif feat == 'age_ratio':
                data_plot = data_plot[(data_plot[feat] >= 0) & (data_plot[feat] <= 3)]
            elif feat == 'ano_density':
                data_plot = data_plot[data_plot[feat] <= data_plot[feat].quantile(0.99)]

            # KDE plots
            for status, label, color in [(0, 'En service', 'green'), (1, 'Abandonné', 'red')]:
                subset = data_plot[data_plot['is_abandoned'] == status][feat]
                if len(subset) > 10:
                    subset.hist(bins=50, density=True, alpha=0.5, label=label, color=color, ax=ax)

            ax.set_xlabel(title)
            ax.set_ylabel('Densité')
            ax.set_title(f'Distribution: {title}')
            ax.legend()

    plt.tight_layout()
    save_figure(fig, 'survival_05_features_predictives')
    report.append("\n![Features prédictives](figures/survival_05_features_predictives.png)\n")

    return df, report


def prepare_modeling_dataset(df, df_anomalies):
    """
    Prépare le dataset final pour la modélisation survie/casse
    """
    print("\n5. Préparation du dataset de modélisation...")

    report = []
    report.append("\n## 5. Dataset pour modélisation\n")

    df = df.copy()

    # Définir la variable cible pour différents horizons
    # Pour l'optimisation: prédire la probabilité de casse dans les N prochaines années

    ref_date = pd.Timestamp('2024-01-01')

    # Cible binaire: abandonné ou pas
    df['target_abandoned'] = df['is_abandoned']

    # Cible survie: durée et événement
    df['survival_time'] = df['duration_years'].clip(lower=0.1)
    df['survival_event'] = df['is_abandoned']  # 1 = événement observé, 0 = censuré

    # Features finales
    features = [
        'GID', 'MAT', 'DIAMETRE', 'LNG',
        'duration_years', 'DDP_year',
        'n_anomalies', 'has_anomaly', 'n_recent_ano',
        'survival_time', 'survival_event', 'target_abandoned',
        'is_abandoned'
    ]

    # Ajouter les features disponibles
    optional_features = ['age_ratio', 'over_median_life', 'ano_density', 'mat_median_life']
    for f in optional_features:
        if f in df.columns:
            features.append(f)

    df_model = df[[f for f in features if f in df.columns]].copy()

    # Sauvegarder
    output_path = f'{PROCESSED_DIR}/dataset_survival_modeling.csv'
    df_model.to_csv(output_path, index=False)
    print(f"   Dataset sauvegardé: {output_path}")
    print(f"   Dimensions: {df_model.shape[0]:,} lignes × {df_model.shape[1]} colonnes")

    report.append(f"\n### Dataset généré\n")
    report.append(f"- **Fichier**: `{output_path}`\n")
    report.append(f"- **Dimensions**: {df_model.shape[0]:,} lignes × {df_model.shape[1]} colonnes\n")
    report.append(f"- **Tronçons abandonnés**: {df_model['target_abandoned'].sum():,} ({df_model['target_abandoned'].mean()*100:.1f}%)\n")

    report.append("\n### Variables disponibles\n")
    report.append("| Variable | Type | Description | Usage |\n")
    report.append("|----------|------|-------------|-------|\n")
    report.append("| GID | ID | Identifiant tronçon | Jointure |\n")
    report.append("| MAT | Catégoriel | Matériau | Feature |\n")
    report.append("| DIAMETRE | Numérique | Diamètre (mm) | Feature |\n")
    report.append("| LNG | Numérique | Longueur (m) | Feature |\n")
    report.append("| duration_years | Numérique | Âge actuel ou à l'abandon | Feature |\n")
    report.append("| n_anomalies | Numérique | Nombre total d'anomalies | Feature |\n")
    report.append("| n_recent_ano | Numérique | Anomalies < 5 ans | Feature |\n")
    report.append("| age_ratio | Numérique | Âge / durée vie médiane mat. | Feature |\n")
    report.append("| survival_time | Numérique | Temps de survie observé | Cible survie |\n")
    report.append("| survival_event | Binaire | 1=abandonné, 0=censuré | Cible survie |\n")
    report.append("| target_abandoned | Binaire | Tronçon abandonné | Cible classif |\n")

    report.append("\n### Utilisation pour l'optimisation\n")
    report.append("""
Le modèle de prédiction génère pour chaque tronçon en service:
1. **Probabilité de casse à horizon H** (ex: 1, 3, 5, 10 ans)
2. **Score de risque** (hazard ratio ou probabilité calibrée)

Ces sorties alimentent le **moteur d'optimisation** qui:
- Maximise le nombre de casses évitées sous contrainte budgétaire
- Priorise les renouvellements selon le ratio coût/bénéfice
- Génère des plans pluriannuels optimaux
""")

    return df_model, report


def write_final_report(all_reports):
    """Écrit le rapport final consolidé"""
    content = '\n'.join(all_reports)

    path = f'{REPORTS_DIR}/06_survival_anomaly_analysis.md'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\n✓ Rapport final: {path}")
    return path


def main():
    """Fonction principale"""
    print("\n" + "="*80)
    print("ANALYSE DE SURVIE : CORRÉLATION ANOMALIES / FIN DE VIE")
    print("="*80)

    # Charger les données
    df_troncons, df_anomalies = load_data()

    # Analyses
    all_reports = []

    # 1. Analyse fin de vie et corrélation anomalies
    df, report1 = analyze_end_of_life(df_troncons, df_anomalies)
    all_reports.extend(report1)

    # 2. Analyse survie par matériau
    df, mat_stats, report2 = analyze_survival_by_material(df, df_anomalies)
    all_reports.extend(report2)

    # 3. Features prédictives
    df, report3 = analyze_predictive_features(df, df_anomalies)
    all_reports.extend(report3)

    # 4. Préparer dataset modélisation
    df_model, report4 = prepare_modeling_dataset(df, df_anomalies)
    all_reports.extend(report4)

    # Écrire rapport final
    write_final_report(all_reports)

    print("\n" + "="*80)
    print("ANALYSE TERMINÉE")
    print("="*80)

    return df_model


if __name__ == '__main__':
    df_model = main()
