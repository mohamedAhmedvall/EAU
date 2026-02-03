#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse des données contextuelles pour la prédiction des défaillances

Ce script analyse les variables contextuelles disponibles :
- Trafic routier (DT_FLUX_CIRCULATION)
- Nombre de logements et abonnés
- Commune et zone géographique
- Données des anomalies (profondeur, cause, etc.)

Auteur: Data Science Pipeline
Date: 2026-02-03
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, spearmanr
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

DATA_DIR = 'data'
REPORTS_DIR = 'reports'
FIGURES_DIR = 'reports/figures'

os.makedirs(FIGURES_DIR, exist_ok=True)


def normalize_datetime(series):
    """Normalise une série datetime"""
    if hasattr(series.dt, 'tz') and series.dt.tz is not None:
        return series.dt.tz_localize(None)
    return series


def save_figure(fig, name, dpi=150):
    """Sauvegarde une figure"""
    path = os.path.join(FIGURES_DIR, f'{name}.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Figure: {path}")
    return path


def load_all_data():
    """Charge et joint toutes les données disponibles"""
    print("\n1. Chargement de toutes les données...")

    # Tronçons de base
    df = pd.read_csv(f'{DATA_DIR}/v1_trafic_prepared.csv')
    df['DDP'] = pd.to_datetime(df['DDP'], errors='coerce')
    df['DHS'] = pd.to_datetime(df['DHS'], errors='coerce')
    df['is_abandoned'] = (df['STATUT_OBJET'] == 'ABANDONNE').astype(int)
    print(f"   Tronçons: {len(df):,}")

    # TronconDT (trafic, logements, abonnés)
    df_dt = pd.read_excel(f'{DATA_DIR}/TronconDT.xlsx')
    print(f"   TronconDT: {len(df_dt):,}")

    # Commune
    df_commune = pd.read_excel(f'{DATA_DIR}/commune.xlsx')
    print(f"   Communes: {len(df_commune):,}")

    # Anomalies
    df_anomalies = pd.read_excel(f'{DATA_DIR}/Anomalie.xlsx')
    df_anomalies['DATE_DETECTION'] = pd.to_datetime(df_anomalies['DATE_DETECTION'], errors='coerce')
    df_anomalies['DATE_DETECTION'] = normalize_datetime(df_anomalies['DATE_DETECTION'])
    print(f"   Anomalies: {len(df_anomalies):,}")

    # Branchements
    df_branch = pd.read_excel(f'{DATA_DIR}/branchement.xlsx')
    print(f"   Branchements: {len(df_branch):,}")

    # Jointures
    print("\n   Jointures...")

    # Joindre TronconDT
    df = df.merge(df_dt, left_on='GID', right_on='DT_GID_TRONCON', how='left')
    print(f"   + TronconDT: {df['DT_FLUX_CIRCULATION'].notna().sum():,} tronçons avec données")

    # Joindre Commune
    df = df.merge(df_commune, left_on='GID', right_on='GID_TRONCON', how='left')
    print(f"   + Commune: {df['GID_COMMUNE'].notna().sum():,} tronçons avec données")

    # Compter anomalies par tronçon
    ano_stats = df_anomalies.groupby('GID_OBJET').agg({
        'GID_OBJET': 'count',
        'PROFONDEUR_FUITE': 'mean',
        'DIAMETRE_FUITE': 'mean',
        'CAUSE_FUITE': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
    })
    ano_stats.columns = ['n_anomalies', 'profondeur_moy', 'diametre_fuite_moy', 'cause_principale']
    ano_stats = ano_stats.reset_index()

    df = df.merge(ano_stats, left_on='GID', right_on='GID_OBJET', how='left')
    df['n_anomalies'] = df['n_anomalies'].fillna(0).astype(int)
    df['has_anomaly'] = (df['n_anomalies'] > 0).astype(int)

    # Compter branchements par tronçon amont
    branch_counts = df_branch.groupby('GID_AMONT').size().reset_index(name='n_branchements')
    df = df.merge(branch_counts, left_on='GID', right_on='GID_AMONT', how='left')
    df['n_branchements'] = df['n_branchements'].fillna(0).astype(int)

    print(f"\n   Dataset final: {len(df):,} lignes × {len(df.columns)} colonnes")

    return df, df_anomalies, df_branch


def analyze_traffic(df):
    """Analyse de l'impact du trafic routier"""
    print("\n2. Analyse du trafic routier...")

    report = []
    report.append("\n## 2. Impact du trafic routier (DT_FLUX_CIRCULATION)\n")

    # Statistiques du trafic
    traffic_stats = df['DT_FLUX_CIRCULATION'].value_counts().sort_index()

    report.append("\n### 2.1 Distribution des niveaux de trafic\n")
    report.append("| Niveau trafic | N tronçons | % |\n")
    report.append("|---------------|------------|---|\n")

    for level in traffic_stats.index:
        count = traffic_stats[level]
        pct = count / len(df) * 100
        report.append(f"| {level} | {count:,} | {pct:.1f}% |\n")

    # Taux d'abandon par niveau de trafic
    traffic_abandon = df.groupby('DT_FLUX_CIRCULATION').agg({
        'GID': 'count',
        'is_abandoned': ['sum', 'mean'],
        'n_anomalies': 'mean'
    })
    traffic_abandon.columns = ['n_total', 'n_abandoned', 'taux_abandon', 'moy_anomalies']
    traffic_abandon['taux_abandon'] = traffic_abandon['taux_abandon'] * 100

    report.append("\n### 2.2 Corrélation trafic → Fin de vie\n")
    report.append("| Niveau trafic | N tronçons | Taux abandon | Moy. anomalies |\n")
    report.append("|---------------|------------|--------------|----------------|\n")

    for level in traffic_abandon.index:
        row = traffic_abandon.loc[level]
        report.append(f"| {level} | {int(row['n_total']):,} | {row['taux_abandon']:.1f}% | {row['moy_anomalies']:.2f} |\n")

    # Test de corrélation
    valid = df[['DT_FLUX_CIRCULATION', 'is_abandoned']].dropna()
    if len(valid) > 100:
        corr, p_val = spearmanr(valid['DT_FLUX_CIRCULATION'], valid['is_abandoned'])
        report.append(f"\n**Corrélation Spearman**: ρ = {corr:.3f}, p-value = {p_val:.2e}\n")

        if p_val < 0.05:
            if corr > 0:
                report.append("→ Plus le trafic est élevé, plus le risque d'abandon augmente.\n")
            else:
                report.append("→ Plus le trafic est élevé, moins le risque d'abandon.\n")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Distribution du trafic
    traffic_stats.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
    axes[0].set_title('Distribution des niveaux de trafic')
    axes[0].set_xlabel('Niveau de trafic')
    axes[0].set_ylabel('Nombre de tronçons')

    # Taux d'abandon par trafic
    traffic_abandon['taux_abandon'].plot(kind='bar', ax=axes[1], color='coral', edgecolor='black')
    axes[1].set_title('Taux d\'abandon par niveau de trafic')
    axes[1].set_xlabel('Niveau de trafic')
    axes[1].set_ylabel('Taux d\'abandon (%)')
    axes[1].axhline(df['is_abandoned'].mean()*100, color='gray', linestyle='--', label='Moyenne')
    axes[1].legend()

    # Moyenne anomalies par trafic
    traffic_abandon['moy_anomalies'].plot(kind='bar', ax=axes[2], color='teal', edgecolor='black')
    axes[2].set_title('Moyenne d\'anomalies par niveau de trafic')
    axes[2].set_xlabel('Niveau de trafic')
    axes[2].set_ylabel('Moyenne anomalies')

    plt.tight_layout()
    save_figure(fig, 'context_01_trafic_analyse')
    report.append("\n![Analyse trafic](figures/context_01_trafic_analyse.png)\n")

    return report


def analyze_housing(df):
    """Analyse de l'impact du nombre de logements et abonnés"""
    print("\n3. Analyse logements et abonnés...")

    report = []
    report.append("\n## 3. Impact du nombre de logements et abonnés\n")

    # Statistiques
    report.append("\n### 3.1 Statistiques descriptives\n")
    report.append("| Variable | Min | Médiane | Moyenne | Max | % renseigné |\n")
    report.append("|----------|-----|---------|---------|-----|-------------|\n")

    for col, label in [('DT_NB_LOGEMENT', 'Nb logements'), ('DT_NB_ABONNE', 'Nb abonnés')]:
        if col in df.columns:
            valid = df[col].dropna()
            pct = len(valid) / len(df) * 100
            report.append(f"| {label} | {valid.min():.0f} | {valid.median():.0f} | "
                         f"{valid.mean():.1f} | {valid.max():.0f} | {pct:.1f}% |\n")

    # Catégoriser le nombre de logements
    df['logements_cat'] = pd.cut(df['DT_NB_LOGEMENT'],
                                  bins=[-1, 0, 5, 20, 100, 10000],
                                  labels=['0', '1-5', '6-20', '21-100', '>100'])

    housing_stats = df.groupby('logements_cat').agg({
        'GID': 'count',
        'is_abandoned': 'mean',
        'n_anomalies': 'mean'
    })
    housing_stats.columns = ['n_troncons', 'taux_abandon', 'moy_anomalies']
    housing_stats['taux_abandon'] = housing_stats['taux_abandon'] * 100

    report.append("\n### 3.2 Impact du nombre de logements desservis\n")
    report.append("| Nb logements | N tronçons | Taux abandon | Moy. anomalies |\n")
    report.append("|--------------|------------|--------------|----------------|\n")

    for cat in housing_stats.index:
        row = housing_stats.loc[cat]
        report.append(f"| {cat} | {int(row['n_troncons']):,} | {row['taux_abandon']:.1f}% | "
                     f"{row['moy_anomalies']:.2f} |\n")

    # Corrélation
    valid = df[['DT_NB_LOGEMENT', 'is_abandoned']].dropna()
    if len(valid) > 100:
        corr, p_val = spearmanr(valid['DT_NB_LOGEMENT'], valid['is_abandoned'])
        report.append(f"\n**Corrélation Spearman (logements vs abandon)**: ρ = {corr:.3f}, p = {p_val:.2e}\n")

    valid = df[['DT_NB_ABONNE', 'is_abandoned']].dropna()
    if len(valid) > 100:
        corr, p_val = spearmanr(valid['DT_NB_ABONNE'], valid['is_abandoned'])
        report.append(f"**Corrélation Spearman (abonnés vs abandon)**: ρ = {corr:.3f}, p = {p_val:.2e}\n")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Taux d'abandon par nb logements
    housing_stats['taux_abandon'].plot(kind='bar', ax=axes[0], color='mediumpurple', edgecolor='black')
    axes[0].set_title('Taux d\'abandon par nombre de logements desservis')
    axes[0].set_xlabel('Nombre de logements')
    axes[0].set_ylabel('Taux d\'abandon (%)')
    axes[0].axhline(df['is_abandoned'].mean()*100, color='gray', linestyle='--')
    axes[0].tick_params(axis='x', rotation=0)

    # Scatter logements vs anomalies
    sample = df[['DT_NB_LOGEMENT', 'n_anomalies']].dropna()
    sample = sample[sample['DT_NB_LOGEMENT'] <= 200]  # Limiter pour visualisation
    sample = sample[sample['n_anomalies'] <= 20]

    axes[1].scatter(sample['DT_NB_LOGEMENT'], sample['n_anomalies'], alpha=0.3, s=10)
    axes[1].set_xlabel('Nombre de logements')
    axes[1].set_ylabel('Nombre d\'anomalies')
    axes[1].set_title('Relation logements vs anomalies')

    # Ligne de tendance
    z = np.polyfit(sample['DT_NB_LOGEMENT'], sample['n_anomalies'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 200, 100)
    axes[1].plot(x_line, p(x_line), 'r--', alpha=0.8, label=f'Tendance')
    axes[1].legend()

    plt.tight_layout()
    save_figure(fig, 'context_02_logements_analyse')
    report.append("\n![Analyse logements](figures/context_02_logements_analyse.png)\n")

    return report


def analyze_communes(df):
    """Analyse géographique par commune"""
    print("\n4. Analyse par commune...")

    report = []
    report.append("\n## 4. Analyse géographique par commune\n")

    # Stats par commune
    commune_stats = df.groupby('GID_COMMUNE').agg({
        'GID': 'count',
        'is_abandoned': ['sum', 'mean'],
        'n_anomalies': ['sum', 'mean'],
        'LNG': 'sum'
    })
    commune_stats.columns = ['n_troncons', 'n_abandoned', 'taux_abandon',
                              'n_anomalies', 'moy_anomalies', 'longueur_totale']
    commune_stats['taux_abandon'] = commune_stats['taux_abandon'] * 100
    commune_stats['densite_ano'] = commune_stats['n_anomalies'] / commune_stats['longueur_totale'] * 1000  # /km
    commune_stats = commune_stats.sort_values('n_troncons', ascending=False)

    report.append("\n### 4.1 Top 20 communes par nombre de tronçons\n")
    report.append("| GID Commune | N tronçons | Taux abandon | Densité ano (/km) |\n")
    report.append("|-------------|------------|--------------|-------------------|\n")

    for commune in commune_stats.head(20).index:
        row = commune_stats.loc[commune]
        report.append(f"| {commune} | {int(row['n_troncons']):,} | {row['taux_abandon']:.1f}% | "
                     f"{row['densite_ano']:.2f} |\n")

    # Communes à risque (taux abandon élevé)
    communes_large = commune_stats[commune_stats['n_troncons'] >= 500]
    top_risk = communes_large.nlargest(10, 'taux_abandon')

    report.append("\n### 4.2 Communes à risque (≥500 tronçons, taux abandon élevé)\n")
    report.append("| GID Commune | N tronçons | Taux abandon | Densité ano (/km) |\n")
    report.append("|-------------|------------|--------------|-------------------|\n")

    for commune in top_risk.index:
        row = top_risk.loc[commune]
        report.append(f"| {commune} | {int(row['n_troncons']):,} | {row['taux_abandon']:.1f}% | "
                     f"{row['densite_ano']:.2f} |\n")

    # Variabilité entre communes
    report.append("\n### 4.3 Variabilité inter-communes\n")
    report.append(f"- Nombre de communes: {len(commune_stats):,}\n")
    report.append(f"- Taux abandon min: {commune_stats['taux_abandon'].min():.1f}%\n")
    report.append(f"- Taux abandon max: {commune_stats['taux_abandon'].max():.1f}%\n")
    report.append(f"- Taux abandon médian: {commune_stats['taux_abandon'].median():.1f}%\n")
    report.append(f"- Écart-type: {commune_stats['taux_abandon'].std():.1f}%\n")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution des taux d'abandon par commune
    communes_large['taux_abandon'].hist(bins=30, ax=axes[0], color='indianred', edgecolor='black')
    axes[0].axvline(communes_large['taux_abandon'].median(), color='black', linestyle='--',
                    label=f'Médiane: {communes_large["taux_abandon"].median():.1f}%')
    axes[0].set_title('Distribution des taux d\'abandon par commune (≥500 tronçons)')
    axes[0].set_xlabel('Taux d\'abandon (%)')
    axes[0].set_ylabel('Nombre de communes')
    axes[0].legend()

    # Scatter taux abandon vs densité anomalies
    axes[1].scatter(communes_large['densite_ano'], communes_large['taux_abandon'],
                    s=communes_large['n_troncons']/50, alpha=0.6)
    axes[1].set_xlabel('Densité d\'anomalies (/km)')
    axes[1].set_ylabel('Taux d\'abandon (%)')
    axes[1].set_title('Taux abandon vs Densité anomalies par commune')

    plt.tight_layout()
    save_figure(fig, 'context_03_communes_analyse')
    report.append("\n![Analyse communes](figures/context_03_communes_analyse.png)\n")

    return report


def analyze_anomaly_details(df_anomalies):
    """Analyse détaillée des caractéristiques des anomalies"""
    print("\n5. Analyse détaillée des anomalies...")

    report = []
    report.append("\n## 5. Caractéristiques détaillées des anomalies\n")

    # Profondeur de fuite
    report.append("\n### 5.1 Profondeur des fuites\n")
    profondeur = df_anomalies['PROFONDEUR_FUITE'].dropna()
    report.append(f"- Données disponibles: {len(profondeur):,} ({len(profondeur)/len(df_anomalies)*100:.1f}%)\n")

    if len(profondeur) > 10:
        report.append(f"- Min: {profondeur.min():.1f} m\n")
        report.append(f"- Médiane: {profondeur.median():.1f} m\n")
        report.append(f"- Moyenne: {profondeur.mean():.1f} m\n")
        report.append(f"- Max: {profondeur.max():.1f} m\n")

    # Cause de fuite
    report.append("\n### 5.2 Causes des fuites\n")
    cause = df_anomalies['CAUSE_FUITE'].dropna()
    report.append(f"- Données disponibles: {len(cause):,} ({len(cause)/len(df_anomalies)*100:.1f}%)\n")

    if len(cause) > 0:
        cause_counts = cause.value_counts()
        report.append("\n| Cause | Nombre | % |\n")
        report.append("|-------|--------|---|\n")
        for c in cause_counts.head(10).index:
            count = cause_counts[c]
            pct = count / len(cause) * 100
            report.append(f"| {c} | {count:,} | {pct:.1f}% |\n")

    # Diamètre de fuite
    report.append("\n### 5.3 Diamètre des fuites\n")
    diametre = df_anomalies['DIAMETRE_FUITE'].dropna()
    report.append(f"- Données disponibles: {len(diametre):,} ({len(diametre)/len(df_anomalies)*100:.1f}%)\n")

    if len(diametre) > 10:
        report.append(f"- Min: {diametre.min():.0f} mm\n")
        report.append(f"- Médiane: {diametre.median():.0f} mm\n")
        report.append(f"- Moyenne: {diametre.mean():.0f} mm\n")
        report.append(f"- Max: {diametre.max():.0f} mm\n")

    # Type d'anomalie
    report.append("\n### 5.4 Types d'anomalies\n")
    type_counts = df_anomalies['TYPE_ANOMALIE'].value_counts()

    report.append("| Type | Nombre | % |\n")
    report.append("|------|--------|---|\n")
    for t in type_counts.index:
        count = type_counts[t]
        pct = count / len(df_anomalies) * 100
        report.append(f"| {t} | {count:,} | {pct:.1f}% |\n")

    # Caractérisation
    report.append("\n### 5.5 Caractérisation des anomalies\n")
    caract_counts = df_anomalies['CARACTERISATION'].value_counts()

    report.append("| Caractérisation | Nombre | % |\n")
    report.append("|-----------------|--------|---|\n")
    for c in caract_counts.head(15).index:
        count = caract_counts[c]
        pct = count / len(df_anomalies) * 100
        report.append(f"| {c} | {count:,} | {pct:.1f}% |\n")

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Types d'anomalies
    type_counts.plot(kind='bar', ax=axes[0, 0], color='steelblue', edgecolor='black')
    axes[0, 0].set_title('Types d\'anomalies')
    axes[0, 0].set_xlabel('')
    axes[0, 0].set_ylabel('Nombre')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Caractérisations
    caract_counts.head(10).plot(kind='bar', ax=axes[0, 1], color='teal', edgecolor='black')
    axes[0, 1].set_title('Top 10 caractérisations')
    axes[0, 1].set_xlabel('')
    axes[0, 1].set_ylabel('Nombre')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Profondeur si disponible
    if len(profondeur) > 10:
        profondeur_capped = profondeur[profondeur <= profondeur.quantile(0.99)]
        axes[1, 0].hist(profondeur_capped, bins=30, color='coral', edgecolor='black')
        axes[1, 0].axvline(profondeur.median(), color='red', linestyle='--',
                          label=f'Médiane: {profondeur.median():.1f}m')
        axes[1, 0].set_title('Distribution de la profondeur des fuites')
        axes[1, 0].set_xlabel('Profondeur (m)')
        axes[1, 0].set_ylabel('Fréquence')
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'Données de profondeur\ninsuffisantes',
                       ha='center', va='center', fontsize=14)
        axes[1, 0].set_title('Profondeur des fuites')

    # Diamètre si disponible
    if len(diametre) > 10:
        diametre_capped = diametre[diametre <= diametre.quantile(0.99)]
        axes[1, 1].hist(diametre_capped, bins=30, color='mediumpurple', edgecolor='black')
        axes[1, 1].axvline(diametre.median(), color='red', linestyle='--',
                          label=f'Médiane: {diametre.median():.0f}mm')
        axes[1, 1].set_title('Distribution du diamètre des fuites')
        axes[1, 1].set_xlabel('Diamètre (mm)')
        axes[1, 1].set_ylabel('Fréquence')
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'Données de diamètre\ninsuffisantes',
                       ha='center', va='center', fontsize=14)
        axes[1, 1].set_title('Diamètre des fuites')

    plt.tight_layout()
    save_figure(fig, 'context_04_anomalies_details')
    report.append("\n![Détails anomalies](figures/context_04_anomalies_details.png)\n")

    return report


def analyze_branchements(df, df_branch):
    """Analyse de l'impact des branchements"""
    print("\n6. Analyse des branchements...")

    report = []
    report.append("\n## 6. Impact des branchements\n")

    # Statistiques sur les branchements
    report.append("\n### 6.1 Statistiques des branchements\n")
    report.append(f"- Nombre total de branchements: {len(df_branch):,}\n")
    report.append(f"- Tronçons avec branchements: {(df['n_branchements'] > 0).sum():,}\n")

    branch_stats = df['n_branchements'].describe()
    report.append(f"- Moyenne branchements/tronçon: {branch_stats['mean']:.1f}\n")
    report.append(f"- Médiane: {branch_stats['50%']:.0f}\n")
    report.append(f"- Max: {branch_stats['max']:.0f}\n")

    # Impact sur les anomalies
    df['branch_cat'] = pd.cut(df['n_branchements'],
                               bins=[-1, 0, 2, 5, 10, 1000],
                               labels=['0', '1-2', '3-5', '6-10', '>10'])

    branch_impact = df.groupby('branch_cat').agg({
        'GID': 'count',
        'is_abandoned': 'mean',
        'n_anomalies': 'mean'
    })
    branch_impact.columns = ['n_troncons', 'taux_abandon', 'moy_anomalies']
    branch_impact['taux_abandon'] = branch_impact['taux_abandon'] * 100

    report.append("\n### 6.2 Impact du nombre de branchements\n")
    report.append("| Nb branchements | N tronçons | Taux abandon | Moy. anomalies |\n")
    report.append("|-----------------|------------|--------------|----------------|\n")

    for cat in branch_impact.index:
        row = branch_impact.loc[cat]
        report.append(f"| {cat} | {int(row['n_troncons']):,} | {row['taux_abandon']:.1f}% | "
                     f"{row['moy_anomalies']:.2f} |\n")

    # Corrélation
    valid = df[['n_branchements', 'n_anomalies']].dropna()
    if len(valid) > 100:
        corr, p_val = spearmanr(valid['n_branchements'], valid['n_anomalies'])
        report.append(f"\n**Corrélation Spearman (branchements vs anomalies)**: ρ = {corr:.3f}, p = {p_val:.2e}\n")

    # Matériaux des branchements
    report.append("\n### 6.3 Matériaux des branchements\n")
    mat_counts = df_branch['MATERIAU'].value_counts()

    report.append("| Matériau | Nombre | % |\n")
    report.append("|----------|--------|---|\n")
    for mat in mat_counts.head(10).index:
        count = mat_counts[mat]
        pct = count / len(df_branch) * 100
        report.append(f"| {mat} | {count:,} | {pct:.1f}% |\n")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Impact branchements
    branch_impact['taux_abandon'].plot(kind='bar', ax=axes[0], color='darkorange', edgecolor='black')
    axes[0].set_title('Taux d\'abandon par nombre de branchements')
    axes[0].set_xlabel('Nombre de branchements')
    axes[0].set_ylabel('Taux d\'abandon (%)')
    axes[0].axhline(df['is_abandoned'].mean()*100, color='gray', linestyle='--')
    axes[0].tick_params(axis='x', rotation=0)

    # Matériaux branchements
    mat_counts.head(8).plot(kind='bar', ax=axes[1], color='forestgreen', edgecolor='black')
    axes[1].set_title('Distribution des matériaux de branchements')
    axes[1].set_xlabel('Matériau')
    axes[1].set_ylabel('Nombre')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    save_figure(fig, 'context_05_branchements_analyse')
    report.append("\n![Analyse branchements](figures/context_05_branchements_analyse.png)\n")

    return report


def create_summary(df):
    """Crée un résumé des données manquantes et recommandations"""
    print("\n7. Synthèse et recommandations...")

    report = []
    report.append("\n## 7. Synthèse : Données disponibles vs manquantes\n")

    report.append("\n### 7.1 Variables contextuelles disponibles\n")
    report.append("| Variable | Source | % renseigné | Impact observé |\n")
    report.append("|----------|--------|-------------|----------------|\n")

    variables = [
        ('DT_FLUX_CIRCULATION', 'TronconDT', df['DT_FLUX_CIRCULATION'].notna().mean()*100, 'Modéré'),
        ('DT_NB_LOGEMENT', 'TronconDT', df['DT_NB_LOGEMENT'].notna().mean()*100, 'Faible'),
        ('DT_NB_ABONNE', 'TronconDT', df['DT_NB_ABONNE'].notna().mean()*100, 'Faible'),
        ('GID_COMMUNE', 'commune', df['GID_COMMUNE'].notna().mean()*100, 'Élevé (effet zone)'),
        ('n_branchements', 'branchement', (df['n_branchements'] > 0).mean()*100, 'Modéré'),
    ]

    for var, source, pct, impact in variables:
        report.append(f"| {var} | {source} | {pct:.1f}% | {impact} |\n")

    report.append("\n### 7.2 Variables manquantes (non disponibles dans les données)\n")
    report.append("| Variable | Importance pour prédiction | Source typique |\n")
    report.append("|----------|---------------------------|----------------|\n")
    report.append("| Pression réseau | Élevée | Capteurs de pression |\n")
    report.append("| Vitesse d'écoulement | Élevée | Modèle hydraulique |\n")
    report.append("| Type de sol/terrain | Élevée | SIG géologique |\n")
    report.append("| Profondeur enfouissement | Modérée | Relevé terrain |\n")
    report.append("| Corrosivité du sol | Élevée | Analyse géotechnique |\n")
    report.append("| Température sol | Modérée | Capteurs/modèle |\n")
    report.append("| Mouvements de terrain | Élevée | SIG risques naturels |\n")

    report.append("\n### 7.3 Recommandations pour enrichir les données\n")
    report.append("""
1. **Pression et vitesse** : Intégrer les données du modèle hydraulique (EPANET ou similaire)
2. **Sol et terrain** : Croiser avec les données géologiques du BRGM
3. **Risques naturels** : Ajouter les zones de retrait-gonflement des argiles
4. **Historique interventions** : Ajouter les données de maintenance préventive
5. **Qualité de l'eau** : Corrosivité, pH, chlorures peuvent impacter les matériaux
""")

    return report


def write_report(all_reports):
    """Écrit le rapport final"""
    content = '\n'.join(all_reports)

    path = f'{REPORTS_DIR}/07_contextual_analysis.md'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\n✓ Rapport: {path}")


def main():
    """Fonction principale"""
    print("\n" + "="*80)
    print("ANALYSE DES DONNÉES CONTEXTUELLES")
    print("="*80)

    # Charger les données
    df, df_anomalies, df_branch = load_all_data()

    # Analyses
    all_reports = []

    all_reports.append("# Analyse des Données Contextuelles\n")
    all_reports.append(f"*Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    all_reports.append("\n## 1. Objectif\n")
    all_reports.append("Analyser les variables contextuelles disponibles (trafic, logements, communes, ")
    all_reports.append("branchements) et identifier les données manquantes pour améliorer le modèle.\n")

    # Analyses détaillées
    all_reports.extend(analyze_traffic(df))
    all_reports.extend(analyze_housing(df))
    all_reports.extend(analyze_communes(df))
    all_reports.extend(analyze_anomaly_details(df_anomalies))
    all_reports.extend(analyze_branchements(df, df_branch))
    all_reports.extend(create_summary(df))

    # Écrire le rapport
    write_report(all_reports)

    print("\n" + "="*80)
    print("ANALYSE TERMINÉE")
    print("="*80)


if __name__ == '__main__':
    main()
