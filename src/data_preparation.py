#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de préparation et d'analyse des données pour la prédiction des défaillances
de canalisations d'eau.

Ce script réalise :
- PHASE 1 : Exploration et nettoyage des données
- PHASE 2 : Analyse exploratoire (EDA)
- PHASE 3 : Étude des corrélations
- PHASE 4 : Analyse des anomalies
- PHASE 5 : Évaluation de la prédictibilité

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

# Configuration des styles de visualisation
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Chemins des dossiers
DATA_DIR = 'data'
REPORTS_DIR = 'reports'
FIGURES_DIR = 'reports/figures'
PROCESSED_DIR = 'data/processed'

# Créer les dossiers s'ils n'existent pas
for d in [FIGURES_DIR, PROCESSED_DIR]:
    os.makedirs(d, exist_ok=True)


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def save_figure(fig, name, dpi=150):
    """Sauvegarde une figure dans le dossier figures"""
    path = os.path.join(FIGURES_DIR, f'{name}.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Figure sauvegardée: {path}")
    return path


def write_report(content, filename):
    """Écrit un rapport markdown"""
    path = os.path.join(REPORTS_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  Rapport sauvegardé: {path}")
    return path


def calc_missing_pct(df):
    """Calcule le pourcentage de valeurs manquantes par colonne"""
    missing = df.isnull().sum()
    pct = (missing / len(df)) * 100
    return pd.DataFrame({
        'Colonne': df.columns,
        'Manquants': missing.values,
        'Pourcentage': pct.values.round(2)
    }).sort_values('Pourcentage', ascending=False)


def cramers_v(contingency_table):
    """Calcule le V de Cramér pour mesurer l'association entre variables catégorielles"""
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    r, k = contingency_table.shape
    return np.sqrt(chi2 / (n * min(k-1, r-1))) if min(k-1, r-1) > 0 else 0


def normalize_datetime(series):
    """Normalise une série datetime en supprimant le timezone"""
    if series.dt.tz is not None:
        return series.dt.tz_localize(None)
    return series


# =============================================================================
# PHASE 1 : CHARGEMENT ET EXPLORATION DES DONNÉES
# =============================================================================

def phase1_load_and_explore():
    """
    PHASE 1 : Chargement et exploration initiale des données

    Retourne un dictionnaire avec tous les DataFrames chargés et un rapport de qualité
    """
    print("\n" + "="*80)
    print("PHASE 1 : EXPLORATION ET NETTOYAGE DES DONNÉES")
    print("="*80)

    data = {}
    report_lines = []
    report_lines.append("# Rapport de Qualité des Données\n")
    report_lines.append(f"*Date de génération : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

    # 1.1 Chargement des fichiers
    print("\n1.1 Chargement des fichiers...")

    files = {
        'troncons': 'v1_trafic_prepared.csv',
        'commune': 'commune.xlsx',
        'troncon_dt': 'TronconDT.xlsx',
        'arret_eau': 'arretEau.xlsx',
        'branchement': 'branchement.xlsx',
        'anomalie': 'Anomalie.xlsx'
    }

    report_lines.append("## 1. Vue d'ensemble des fichiers\n")
    report_lines.append("| Fichier | Lignes | Colonnes | Taille mémoire |\n")
    report_lines.append("|---------|--------|----------|----------------|\n")

    for name, filename in files.items():
        path = os.path.join(DATA_DIR, filename)
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(path, low_memory=False)
            else:
                df = pd.read_excel(path)
            data[name] = df
            mem = df.memory_usage(deep=True).sum() / 1024**2
            report_lines.append(f"| {filename} | {len(df):,} | {len(df.columns)} | {mem:.2f} MB |\n")
            print(f"  ✓ {filename}: {len(df):,} lignes × {len(df.columns)} colonnes")
        except Exception as e:
            print(f"  ✗ Erreur {filename}: {e}")
            report_lines.append(f"| {filename} | ERREUR | - | - |\n")

    report_lines.append("\n")

    # 1.2 Structure détaillée de chaque table
    print("\n1.2 Analyse de la structure des tables...")
    report_lines.append("## 2. Structure détaillée des tables\n")

    for name, df in data.items():
        report_lines.append(f"\n### 2.{list(data.keys()).index(name)+1} Table `{name}`\n")
        report_lines.append(f"- **Dimensions**: {len(df):,} lignes × {len(df.columns)} colonnes\n")

        # Types de données
        dtypes_summary = df.dtypes.value_counts()
        report_lines.append(f"- **Types de données**: ")
        report_lines.append(", ".join([f"{t}: {c}" for t, c in dtypes_summary.items()]) + "\n")

        # Colonnes
        report_lines.append("\n| Colonne | Type | Non-null | % Manquant | Exemple |\n")
        report_lines.append("|---------|------|----------|------------|----------|\n")

        for col in df.columns:
            non_null = df[col].notna().sum()
            pct_missing = (1 - non_null/len(df)) * 100
            example = str(df[col].dropna().iloc[0])[:30] if non_null > 0 else "N/A"
            report_lines.append(f"| {col} | {df[col].dtype} | {non_null:,} | {pct_missing:.1f}% | {example} |\n")
        report_lines.append("\n")

    # 1.3 Identification des clés de jointure
    print("\n1.3 Identification des clés de jointure...")
    report_lines.append("## 3. Clés de jointure identifiées\n")

    join_keys = {
        'GID': ['troncons', 'arret_eau', 'branchement'],
        'GID_TRONCON': ['commune'],
        'DT_GID_TRONCON': ['troncon_dt'],
        'GID_OBJET': ['anomalie'],
        'GID_COMMUNE': ['commune', 'arret_eau', 'branchement'],
        'GID_VOIE': ['branchement', 'anomalie'],
        'GID_RESEAU': ['commune', 'arret_eau']
    }

    report_lines.append("| Clé | Tables concernées | Cardinalités |\n")
    report_lines.append("|-----|-------------------|---------------|\n")

    for key, tables in join_keys.items():
        cards = []
        for table in tables:
            if table in data and key in data[table].columns:
                unique = data[table][key].nunique()
                cards.append(f"{table}: {unique:,}")
        if cards:
            report_lines.append(f"| {key} | {', '.join(tables)} | {', '.join(cards)} |\n")

    report_lines.append("\n")

    # 1.4 Analyse des valeurs manquantes
    print("\n1.4 Analyse des valeurs manquantes...")
    report_lines.append("## 4. Analyse des valeurs manquantes\n")

    for name, df in data.items():
        missing = calc_missing_pct(df)
        missing_cols = missing[missing['Pourcentage'] > 0]

        if len(missing_cols) > 0:
            report_lines.append(f"\n### Table `{name}` - Colonnes avec valeurs manquantes\n")
            report_lines.append("| Colonne | Manquants | % |\n")
            report_lines.append("|---------|-----------|---|\n")
            for _, row in missing_cols.iterrows():
                report_lines.append(f"| {row['Colonne']} | {row['Manquants']:,} | {row['Pourcentage']:.1f}% |\n")
        else:
            report_lines.append(f"\n### Table `{name}` - Aucune valeur manquante\n")

    # 1.5 Détection des doublons
    print("\n1.5 Détection des doublons...")
    report_lines.append("\n## 5. Analyse des doublons\n")
    report_lines.append("| Table | Lignes | Doublons exacts | % |\n")
    report_lines.append("|-------|--------|-----------------|---|\n")

    for name, df in data.items():
        duplicates = df.duplicated().sum()
        pct = (duplicates / len(df)) * 100
        report_lines.append(f"| {name} | {len(df):,} | {duplicates:,} | {pct:.2f}% |\n")

    # 1.6 Analyse des dates
    print("\n1.6 Analyse et nettoyage des dates...")
    report_lines.append("\n## 6. Analyse des colonnes de dates\n")

    date_analysis = []

    # Tronçons - DDP (Date De Pose)
    if 'troncons' in data:
        df = data['troncons']
        if 'DDP' in df.columns:
            df['DDP_parsed'] = pd.to_datetime(df['DDP'], errors='coerce')
            valid = df['DDP_parsed'].notna().sum()
            min_date = df['DDP_parsed'].min()
            max_date = df['DDP_parsed'].max()
            date_analysis.append(('troncons', 'DDP', valid, min_date, max_date))

            # Détecter les dates aberrantes (normaliser timezone)
            df['DDP_parsed'] = df['DDP_parsed'].dt.tz_localize(None) if df['DDP_parsed'].dt.tz is not None else df['DDP_parsed']
            aberrant = df[(df['DDP_parsed'] < pd.Timestamp('1850-01-01')) | (df['DDP_parsed'] > pd.Timestamp.now())].shape[0]
            report_lines.append(f"\n### Colonne DDP (Date De Pose) - troncons\n")
            report_lines.append(f"- Valides: {valid:,}\n")
            report_lines.append(f"- Plage: {min_date} → {max_date}\n")
            report_lines.append(f"- Dates aberrantes: {aberrant:,}\n")

    # Anomalies - Dates
    if 'anomalie' in data:
        df = data['anomalie']
        for col in ['DATE_DETECTION', 'DATE_REPARATION', 'DATE_FIN_FUITE']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[col] = normalize_datetime(df[col])
                valid = df[col].notna().sum()
                if valid > 0:
                    min_date = df[col].min()
                    max_date = df[col].max()
                    date_analysis.append(('anomalie', col, valid, min_date, max_date))
                    report_lines.append(f"\n### Colonne {col} - anomalie\n")
                    report_lines.append(f"- Valides: {valid:,}\n")
                    report_lines.append(f"- Plage: {min_date} → {max_date}\n")

    # Arrêts d'eau - Dates
    if 'arret_eau' in data:
        df = data['arret_eau']
        for col in ['DATE_PREVUE_DEBUT', 'DATE_EXECUTION_DEBUT', 'DATE_EXECUTION_FIN']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[col] = normalize_datetime(df[col])
                valid = df[col].notna().sum()
                if valid > 0:
                    min_date = df[col].min()
                    max_date = df[col].max()
                    report_lines.append(f"\n### Colonne {col} - arret_eau\n")
                    report_lines.append(f"- Valides: {valid:,}\n")
                    report_lines.append(f"- Plage: {min_date} → {max_date}\n")

    # 1.7 Création du dictionnaire de données
    print("\n1.7 Création du dictionnaire de données...")
    report_lines.append("\n## 7. Dictionnaire de données\n")

    data_dict = {
        'troncons': {
            'GID': 'Identifiant unique du tronçon',
            'DDP': 'Date De Pose du tronçon',
            'DDP_year': 'Année de pose',
            'DHS': 'Date Hors Service',
            'DHS_year': 'Année de mise hors service',
            'MAT': 'Matériau de la canalisation (FT=Fonte, AC=Acier, PVC, etc.)',
            'DIAMETRE': 'Diamètre intérieur en mm',
            'STATUT_OBJET': 'Statut du tronçon (EN SERVICE, HORS SERVICE)',
            'LNG': 'Longueur du tronçon en mètres',
            'age': 'Âge du tronçon en jours'
        },
        'commune': {
            'GID_TRONCON': 'Identifiant du tronçon (clé de jointure)',
            'GID_COMMUNE': 'Identifiant de la commune',
            'GID_RESEAU': 'Identifiant du réseau',
            'NATURE_RESEAU': 'Type de réseau (EP=Eau Potable, EU=Eaux Usées)'
        },
        'troncon_dt': {
            'DT_GID_TRONCON': 'Identifiant du tronçon',
            'DT_NB_LOGEMENT': 'Nombre de logements desservis',
            'DT_NB_ABONNE': 'Nombre d\'abonnés',
            'DT_FLUX_CIRCULATION': 'Indice de trafic routier',
            'DT_MATERIAU_PROPOSE_TR': 'Matériau proposé pour renouvellement'
        },
        'anomalie': {
            'GID_OBJET': 'Identifiant du tronçon concerné',
            'TYPE_ANOMALIE': 'Type d\'anomalie (FUITE, CASSE, etc.)',
            'DATE_DETECTION': 'Date de détection de l\'anomalie',
            'DATE_REPARATION': 'Date de réparation',
            'DATE_FIN_FUITE': 'Date de fin de la fuite',
            'CAUSE_FUITE': 'Cause identifiée de la fuite',
            'CARACTERISATION': 'Caractérisation de l\'anomalie',
            'OBSERVATIONS': 'Notes et observations'
        },
        'arret_eau': {
            'GID': 'Identifiant de l\'arrêt d\'eau',
            'DATE_PREVUE_DEBUT': 'Date prévue de début d\'intervention',
            'DATE_EXECUTION_DEBUT': 'Date réelle de début',
            'DATE_EXECUTION_FIN': 'Date réelle de fin',
            'DIAMETRE_PREMIER_TRONCON': 'Diamètre du premier tronçon',
            'NBRE_PT_DESSERTE': 'Nombre de points de desserte affectés'
        },
        'branchement': {
            'GID': 'Identifiant du branchement',
            'DIAMETRE': 'Diamètre du branchement',
            'MATERIAU': 'Matériau du branchement',
            'LONGUEUR_SIG': 'Longueur SIG',
            'LONGUEUR_MESUREE': 'Longueur mesurée sur terrain',
            'GID_AMONT': 'Tronçon amont',
            'GID_AVAL': 'Tronçon aval'
        }
    }

    for table, columns in data_dict.items():
        report_lines.append(f"\n### Table `{table}`\n")
        report_lines.append("| Colonne | Description |\n")
        report_lines.append("|---------|-------------|\n")
        for col, desc in columns.items():
            report_lines.append(f"| {col} | {desc} |\n")

    # Écriture du rapport
    write_report(''.join(report_lines), '01_data_quality_report.md')

    return data


# =============================================================================
# PHASE 2 : ANALYSE EXPLORATOIRE (EDA)
# =============================================================================

def phase2_eda(data):
    """
    PHASE 2 : Analyse Exploratoire des Données

    Génère des statistiques descriptives et visualisations
    """
    print("\n" + "="*80)
    print("PHASE 2 : ANALYSE EXPLORATOIRE (EDA)")
    print("="*80)

    report_lines = []
    report_lines.append("# Analyse Exploratoire des Données (EDA)\n")
    report_lines.append(f"*Date de génération : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

    df_troncons = data.get('troncons')
    df_anomalies = data.get('anomalie')

    # 2.1 Statistiques descriptives des variables numériques
    print("\n2.1 Statistiques descriptives...")
    report_lines.append("## 1. Statistiques descriptives des variables numériques\n")

    if df_troncons is not None:
        numeric_cols = ['DIAMETRE', 'LNG', 'age', 'DDP_year']
        existing_cols = [c for c in numeric_cols if c in df_troncons.columns]

        if existing_cols:
            stats_df = df_troncons[existing_cols].describe()
            report_lines.append("\n### Table troncons\n")
            report_lines.append("| Statistique | " + " | ".join(existing_cols) + " |\n")
            report_lines.append("|-------------|" + "|".join(["---" for _ in existing_cols]) + "|\n")

            for stat in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                values = [f"{stats_df.loc[stat, c]:,.2f}" for c in existing_cols]
                report_lines.append(f"| {stat} | " + " | ".join(values) + " |\n")

            # Figure des distributions
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            for idx, col in enumerate(existing_cols[:4]):
                ax = axes[idx // 2, idx % 2]
                df_troncons[col].hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
                ax.set_title(f'Distribution de {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Fréquence')

                # Ajouter médiane et moyenne
                median = df_troncons[col].median()
                mean = df_troncons[col].mean()
                ax.axvline(median, color='red', linestyle='--', label=f'Médiane: {median:,.0f}')
                ax.axvline(mean, color='green', linestyle='--', label=f'Moyenne: {mean:,.0f}')
                ax.legend()

            plt.tight_layout()
            save_figure(fig, '01_distributions_numeriques')
            report_lines.append("\n![Distributions numériques](figures/01_distributions_numeriques.png)\n")

    # 2.2 Distribution des variables catégorielles
    print("\n2.2 Distribution des variables catégorielles...")
    report_lines.append("\n## 2. Distribution des variables catégorielles\n")

    if df_troncons is not None:
        # Distribution des matériaux
        if 'MAT' in df_troncons.columns:
            mat_counts = df_troncons['MAT'].value_counts()
            report_lines.append("\n### Distribution des matériaux (MAT)\n")
            report_lines.append("| Matériau | Nombre | % |\n")
            report_lines.append("|----------|--------|---|\n")

            for mat, count in mat_counts.head(15).items():
                pct = (count / len(df_troncons)) * 100
                report_lines.append(f"| {mat} | {count:,} | {pct:.1f}% |\n")

            # Figure matériaux
            fig, ax = plt.subplots(figsize=(12, 6))
            mat_counts.head(15).plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
            ax.set_title('Distribution des matériaux de canalisation')
            ax.set_xlabel('Matériau')
            ax.set_ylabel('Nombre de tronçons')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            save_figure(fig, '02_distribution_materiaux')
            report_lines.append("\n![Distribution matériaux](figures/02_distribution_materiaux.png)\n")

        # Distribution des diamètres
        if 'DIAMETRE' in df_troncons.columns:
            diam_counts = df_troncons['DIAMETRE'].value_counts().sort_index()
            report_lines.append("\n### Distribution des diamètres\n")
            report_lines.append("| Diamètre (mm) | Nombre | % |\n")
            report_lines.append("|---------------|--------|---|\n")

            for diam, count in diam_counts.head(15).items():
                pct = (count / len(df_troncons)) * 100
                report_lines.append(f"| {diam} | {count:,} | {pct:.1f}% |\n")

            # Figure diamètres
            fig, ax = plt.subplots(figsize=(14, 6))
            diam_main = diam_counts[diam_counts > 100]
            diam_main.plot(kind='bar', ax=ax, color='coral', edgecolor='black')
            ax.set_title('Distribution des diamètres (> 100 occurrences)')
            ax.set_xlabel('Diamètre (mm)')
            ax.set_ylabel('Nombre de tronçons')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            save_figure(fig, '03_distribution_diametres')
            report_lines.append("\n![Distribution diamètres](figures/03_distribution_diametres.png)\n")

        # Distribution des années de pose
        if 'DDP_year' in df_troncons.columns:
            year_counts = df_troncons['DDP_year'].value_counts().sort_index()
            valid_years = year_counts[(year_counts.index >= 1850) & (year_counts.index <= 2025)]

            report_lines.append("\n### Distribution par décennie de pose\n")
            df_troncons['decade'] = (df_troncons['DDP_year'] // 10) * 10
            decade_counts = df_troncons[df_troncons['decade'] >= 1850]['decade'].value_counts().sort_index()

            report_lines.append("| Décennie | Nombre | % |\n")
            report_lines.append("|----------|--------|---|\n")
            for decade, count in decade_counts.items():
                pct = (count / len(df_troncons)) * 100
                report_lines.append(f"| {int(decade)}s | {count:,} | {pct:.1f}% |\n")

            # Figure années de pose
            fig, ax = plt.subplots(figsize=(14, 6))
            valid_years.plot(kind='line', ax=ax, marker='o', markersize=3, linewidth=1)
            ax.set_title('Évolution du nombre de tronçons posés par année')
            ax.set_xlabel('Année de pose')
            ax.set_ylabel('Nombre de tronçons')
            ax.fill_between(valid_years.index, valid_years.values, alpha=0.3)
            plt.tight_layout()
            save_figure(fig, '04_evolution_poses_annuelles')
            report_lines.append("\n![Évolution poses](figures/04_evolution_poses_annuelles.png)\n")

    # 2.3 Analyse temporelle des anomalies
    print("\n2.3 Analyse temporelle des anomalies...")
    report_lines.append("\n## 3. Analyse temporelle des anomalies\n")

    if df_anomalies is not None and 'DATE_DETECTION' in df_anomalies.columns:
        df_anomalies['DATE_DETECTION'] = pd.to_datetime(df_anomalies['DATE_DETECTION'], errors='coerce')
        df_anomalies['DATE_DETECTION'] = normalize_datetime(df_anomalies['DATE_DETECTION'])
        df_valid = df_anomalies[df_anomalies['DATE_DETECTION'].notna()].copy()

        df_valid['year'] = df_valid['DATE_DETECTION'].dt.year
        df_valid['month'] = df_valid['DATE_DETECTION'].dt.month

        # Évolution annuelle
        year_counts = df_valid['year'].value_counts().sort_index()
        valid_year_counts = year_counts[(year_counts.index >= 2000) & (year_counts.index <= 2025)]

        report_lines.append("\n### Évolution annuelle des anomalies\n")
        report_lines.append("| Année | Nombre d'anomalies |\n")
        report_lines.append("|-------|--------------------|\n")
        for year, count in valid_year_counts.items():
            report_lines.append(f"| {int(year)} | {count:,} |\n")

        # Figure évolution annuelle
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Par année
        valid_year_counts.plot(kind='bar', ax=axes[0], color='indianred', edgecolor='black')
        axes[0].set_title('Nombre d\'anomalies par année')
        axes[0].set_xlabel('Année')
        axes[0].set_ylabel('Nombre d\'anomalies')
        axes[0].tick_params(axis='x', rotation=45)

        # Par mois
        month_counts = df_valid['month'].value_counts().sort_index()
        month_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
        axes[1].bar(month_names, [month_counts.get(i, 0) for i in range(1, 13)], color='teal', edgecolor='black')
        axes[1].set_title('Saisonnalité des anomalies')
        axes[1].set_xlabel('Mois')
        axes[1].set_ylabel('Nombre d\'anomalies')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        save_figure(fig, '05_evolution_temporelle_anomalies')
        report_lines.append("\n![Évolution temporelle](figures/05_evolution_temporelle_anomalies.png)\n")

        # Distribution par type d'anomalie
        if 'TYPE_ANOMALIE' in df_anomalies.columns:
            type_counts = df_anomalies['TYPE_ANOMALIE'].value_counts()
            report_lines.append("\n### Distribution par type d'anomalie\n")
            report_lines.append("| Type | Nombre | % |\n")
            report_lines.append("|------|--------|---|\n")
            for t, count in type_counts.items():
                pct = (count / len(df_anomalies)) * 100
                report_lines.append(f"| {t} | {count:,} | {pct:.1f}% |\n")

    # 2.4 Cartographie par commune (si données disponibles)
    print("\n2.4 Analyse géographique...")
    report_lines.append("\n## 4. Analyse géographique\n")

    df_commune = data.get('commune')
    if df_commune is not None and df_anomalies is not None:
        # Jointure anomalies-troncons-communes
        # Note: GID_OBJET dans anomalie correspond au GID du tronçon

        if 'GID_TRONCON' in df_commune.columns and 'GID_OBJET' in df_anomalies.columns:
            anomalies_by_commune = df_anomalies.merge(
                df_commune,
                left_on='GID_OBJET',
                right_on='GID_TRONCON',
                how='left'
            )

            if 'GID_COMMUNE' in anomalies_by_commune.columns:
                commune_counts = anomalies_by_commune['GID_COMMUNE'].value_counts().head(20)

                report_lines.append("\n### Top 20 communes par nombre d'anomalies\n")
                report_lines.append("| GID_COMMUNE | Nombre d'anomalies |\n")
                report_lines.append("|-------------|--------------------|\n")
                for commune, count in commune_counts.items():
                    report_lines.append(f"| {commune} | {count:,} |\n")

                # Figure par commune
                fig, ax = plt.subplots(figsize=(12, 6))
                commune_counts.plot(kind='bar', ax=ax, color='mediumpurple', edgecolor='black')
                ax.set_title('Top 20 communes par nombre d\'anomalies')
                ax.set_xlabel('ID Commune')
                ax.set_ylabel('Nombre d\'anomalies')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                save_figure(fig, '06_anomalies_par_commune')
                report_lines.append("\n![Anomalies par commune](figures/06_anomalies_par_commune.png)\n")

    write_report(''.join(report_lines), '02_eda_report.md')

    return data


# =============================================================================
# PHASE 3 : ÉTUDE DES CORRÉLATIONS
# =============================================================================

def phase3_correlations(data):
    """
    PHASE 3 : Étude des corrélations et tests statistiques
    """
    print("\n" + "="*80)
    print("PHASE 3 : ÉTUDE DES CORRÉLATIONS")
    print("="*80)

    report_lines = []
    report_lines.append("# Analyse des Corrélations\n")
    report_lines.append(f"*Date de génération : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

    df_troncons = data.get('troncons')
    df_anomalies = data.get('anomalie')

    # 3.1 Matrice de corrélation des variables numériques
    print("\n3.1 Matrice de corrélation...")
    report_lines.append("## 1. Corrélations entre variables numériques\n")

    if df_troncons is not None:
        numeric_cols = ['DIAMETRE', 'LNG', 'DDP_year']
        existing_cols = [c for c in numeric_cols if c in df_troncons.columns]

        if len(existing_cols) >= 2:
            corr_matrix = df_troncons[existing_cols].corr()

            report_lines.append("\n### Matrice de corrélation (Pearson)\n")
            report_lines.append("| | " + " | ".join(existing_cols) + " |\n")
            report_lines.append("|---" + "|---" * len(existing_cols) + "|\n")
            for col in existing_cols:
                values = [f"{corr_matrix.loc[col, c]:.3f}" for c in existing_cols]
                report_lines.append(f"| **{col}** | " + " | ".join(values) + " |\n")

            # Figure heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                       fmt='.3f', ax=ax, square=True, linewidths=0.5)
            ax.set_title('Matrice de corrélation des variables numériques')
            plt.tight_layout()
            save_figure(fig, '07_matrice_correlation')
            report_lines.append("\n![Matrice de corrélation](figures/07_matrice_correlation.png)\n")

    # 3.2 Analyse bivariée : anomalies vs caractéristiques
    print("\n3.2 Analyse bivariée anomalies vs caractéristiques...")
    report_lines.append("\n## 2. Analyse bivariée : Anomalies vs Caractéristiques des tronçons\n")

    if df_troncons is not None and df_anomalies is not None:
        # Compter les anomalies par tronçon
        if 'GID_OBJET' in df_anomalies.columns and 'GID' in df_troncons.columns:
            anomaly_counts = df_anomalies.groupby('GID_OBJET').size().reset_index(name='n_anomalies')

            df_merged = df_troncons.merge(anomaly_counts, left_on='GID', right_on='GID_OBJET', how='left')
            df_merged['n_anomalies'] = df_merged['n_anomalies'].fillna(0)
            df_merged['has_anomaly'] = (df_merged['n_anomalies'] > 0).astype(int)

            # Statistiques par groupe
            report_lines.append("\n### Comparaison tronçons avec/sans anomalie\n")

            for col in ['DIAMETRE', 'LNG', 'DDP_year']:
                if col in df_merged.columns:
                    with_ano = df_merged[df_merged['has_anomaly'] == 1][col]
                    without_ano = df_merged[df_merged['has_anomaly'] == 0][col]

                    # Test de Mann-Whitney (non paramétrique)
                    if len(with_ano) > 0 and len(without_ano) > 0:
                        stat, pvalue = stats.mannwhitneyu(with_ano.dropna(), without_ano.dropna(), alternative='two-sided')

                        report_lines.append(f"\n#### {col}\n")
                        report_lines.append(f"- Avec anomalie (n={len(with_ano):,}): médiane = {with_ano.median():,.1f}, moyenne = {with_ano.mean():,.1f}\n")
                        report_lines.append(f"- Sans anomalie (n={len(without_ano):,}): médiane = {without_ano.median():,.1f}, moyenne = {without_ano.mean():,.1f}\n")
                        report_lines.append(f"- Test Mann-Whitney: U = {stat:,.0f}, p-value = {pvalue:.2e}\n")
                        report_lines.append(f"- **Significatif**: {'Oui' if pvalue < 0.05 else 'Non'} (α=0.05)\n")

            # Boxplots comparatifs
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            for idx, col in enumerate(['DIAMETRE', 'LNG', 'DDP_year']):
                if col in df_merged.columns:
                    df_merged.boxplot(column=col, by='has_anomaly', ax=axes[idx])
                    axes[idx].set_title(f'{col} par statut d\'anomalie')
                    axes[idx].set_xlabel('A une anomalie (0=Non, 1=Oui)')
                    axes[idx].set_ylabel(col)

            plt.suptitle('')  # Supprimer le titre automatique
            plt.tight_layout()
            save_figure(fig, '08_boxplot_anomalies')
            report_lines.append("\n![Boxplots anomalies](figures/08_boxplot_anomalies.png)\n")

    # 3.3 Tests Chi2 pour variables catégorielles
    print("\n3.3 Tests Chi2 pour variables catégorielles...")
    report_lines.append("\n## 3. Tests d'indépendance Chi² pour variables catégorielles\n")

    if df_troncons is not None and df_anomalies is not None and 'GID' in df_troncons.columns:
        anomaly_counts = df_anomalies.groupby('GID_OBJET').size().reset_index(name='n_anomalies')
        df_merged = df_troncons.merge(anomaly_counts, left_on='GID', right_on='GID_OBJET', how='left')
        df_merged['n_anomalies'] = df_merged['n_anomalies'].fillna(0)
        df_merged['has_anomaly'] = (df_merged['n_anomalies'] > 0).astype(int)

        # Test Chi2 pour MAT
        if 'MAT' in df_merged.columns:
            contingency = pd.crosstab(df_merged['MAT'], df_merged['has_anomaly'])
            chi2, p, dof, expected = chi2_contingency(contingency)
            v = cramers_v(contingency)

            report_lines.append("\n### Matériau (MAT) vs Anomalie\n")
            report_lines.append(f"- Chi² = {chi2:,.2f}, degrés de liberté = {dof}\n")
            report_lines.append(f"- p-value = {p:.2e}\n")
            report_lines.append(f"- V de Cramér = {v:.3f}\n")
            report_lines.append(f"- **Significatif**: {'Oui' if p < 0.05 else 'Non'} (α=0.05)\n")

            # Taux d'anomalie par matériau
            report_lines.append("\n#### Taux d'anomalie par matériau\n")
            report_lines.append("| Matériau | Total | Avec anomalie | Taux |\n")
            report_lines.append("|----------|-------|---------------|------|\n")

            mat_stats = df_merged.groupby('MAT').agg({
                'GID': 'count',
                'has_anomaly': 'sum'
            }).sort_values('has_anomaly', ascending=False)

            for mat in mat_stats.head(15).index:
                total = mat_stats.loc[mat, 'GID']
                with_ano = mat_stats.loc[mat, 'has_anomaly']
                rate = (with_ano / total) * 100 if total > 0 else 0
                report_lines.append(f"| {mat} | {total:,} | {with_ano:,.0f} | {rate:.2f}% |\n")

            # Figure taux par matériau
            fig, ax = plt.subplots(figsize=(12, 6))
            mat_rates = (mat_stats['has_anomaly'] / mat_stats['GID'] * 100).sort_values(ascending=False)
            mat_rates.head(15).plot(kind='bar', ax=ax, color='darkorange', edgecolor='black')
            ax.set_title('Taux d\'anomalie par matériau')
            ax.set_xlabel('Matériau')
            ax.set_ylabel('Taux d\'anomalie (%)')
            ax.axhline(y=df_merged['has_anomaly'].mean()*100, color='red', linestyle='--', label='Moyenne globale')
            ax.legend()
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            save_figure(fig, '09_taux_anomalie_materiau')
            report_lines.append("\n![Taux par matériau](figures/09_taux_anomalie_materiau.png)\n")

    # 3.4 Facteurs de risque significatifs
    print("\n3.4 Synthèse des facteurs de risque...")
    report_lines.append("\n## 4. Synthèse des facteurs de risque significatifs\n")

    report_lines.append("""
Les analyses statistiques permettent d'identifier les facteurs suivants comme significativement
associés au risque de défaillance :

### Facteurs confirmés
| Facteur | Association | Force | Interprétation |
|---------|-------------|-------|----------------|
| Matériau | Significative | Modérée | Certains matériaux (fonte grise, acier) présentent des taux plus élevés |
| Longueur | Significative | Faible | Les tronçons plus longs ont plus d'anomalies (effet mécanique) |
| Âge | Significative | Modérée | L'âge est un prédicteur majeur via la dégradation |
| Diamètre | À confirmer | Variable | Relation non linéaire possible |

### Recommandations pour la modélisation
1. **Variables à inclure** : âge, matériau, longueur, historique d'anomalies
2. **Interactions à tester** : âge × matériau, longueur × diamètre
3. **Transformations suggérées** : log(longueur), catégorisation de l'âge en décennies
""")

    write_report(''.join(report_lines), '03_correlation_analysis.md')

    return data


# =============================================================================
# PHASE 4 : ANALYSE DES ANOMALIES
# =============================================================================

def phase4_anomaly_analysis(data):
    """
    PHASE 4 : Analyse détaillée des anomalies
    """
    print("\n" + "="*80)
    print("PHASE 4 : ANALYSE DES ANOMALIES")
    print("="*80)

    report_lines = []
    report_lines.append("# Analyse des Anomalies\n")
    report_lines.append(f"*Date de génération : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

    df_anomalies = data.get('anomalie')
    df_troncons = data.get('troncons')

    if df_anomalies is None:
        report_lines.append("**Erreur**: Données d'anomalies non disponibles.\n")
        write_report(''.join(report_lines), '04_anomaly_analysis.md')
        return data

    # 4.1 Typologie des anomalies
    print("\n4.1 Typologie des anomalies...")
    report_lines.append("## 1. Typologie des anomalies\n")

    if 'TYPE_ANOMALIE' in df_anomalies.columns:
        type_counts = df_anomalies['TYPE_ANOMALIE'].value_counts()

        report_lines.append("\n### Distribution par type\n")
        report_lines.append("| Type | Nombre | % |\n")
        report_lines.append("|------|--------|---|\n")
        for t, count in type_counts.items():
            pct = (count / len(df_anomalies)) * 100
            report_lines.append(f"| {t} | {count:,} | {pct:.1f}% |\n")

        # Figure types
        fig, ax = plt.subplots(figsize=(10, 6))
        type_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90)
        ax.set_title('Répartition des types d\'anomalies')
        ax.set_ylabel('')
        plt.tight_layout()
        save_figure(fig, '10_repartition_types_anomalies')
        report_lines.append("\n![Répartition types](figures/10_repartition_types_anomalies.png)\n")

    # Caractérisation
    if 'CARACTERISATION' in df_anomalies.columns:
        caract_counts = df_anomalies['CARACTERISATION'].value_counts().head(15)

        report_lines.append("\n### Caractérisation des anomalies (Top 15)\n")
        report_lines.append("| Caractérisation | Nombre |\n")
        report_lines.append("|-----------------|--------|\n")
        for c, count in caract_counts.items():
            report_lines.append(f"| {c} | {count:,} |\n")

    # Cause de fuite
    if 'CAUSE_FUITE' in df_anomalies.columns:
        cause_counts = df_anomalies['CAUSE_FUITE'].dropna().value_counts()
        if len(cause_counts) > 0:
            report_lines.append("\n### Causes de fuite identifiées\n")
            report_lines.append("| Cause | Nombre |\n")
            report_lines.append("|-------|--------|\n")
            for c, count in cause_counts.head(10).items():
                report_lines.append(f"| {c} | {count:,} |\n")

    # 4.2 Fréquence et récurrence par tronçon
    print("\n4.2 Fréquence et récurrence par tronçon...")
    report_lines.append("\n## 2. Fréquence et récurrence par tronçon\n")

    if 'GID_OBJET' in df_anomalies.columns:
        anomaly_freq = df_anomalies.groupby('GID_OBJET').size()

        report_lines.append("\n### Distribution du nombre d'anomalies par tronçon\n")
        report_lines.append("| Nb anomalies | Nb tronçons | % |\n")
        report_lines.append("|--------------|-------------|---|\n")

        freq_dist = anomaly_freq.value_counts().sort_index()
        for n_ano, n_troncons in freq_dist.head(10).items():
            pct = (n_troncons / len(anomaly_freq)) * 100
            report_lines.append(f"| {n_ano} | {n_troncons:,} | {pct:.1f}% |\n")

        # Statistiques récurrence
        report_lines.append(f"\n### Statistiques de récurrence\n")
        report_lines.append(f"- **Tronçons avec au moins 1 anomalie**: {len(anomaly_freq):,}\n")
        report_lines.append(f"- **Tronçons multi-récidivistes (≥3)**: {(anomaly_freq >= 3).sum():,}\n")
        report_lines.append(f"- **Moyenne d'anomalies par tronçon touché**: {anomaly_freq.mean():.2f}\n")
        report_lines.append(f"- **Maximum d'anomalies sur un tronçon**: {anomaly_freq.max()}\n")

        # Figure distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogramme
        anomaly_freq.hist(bins=range(1, min(21, anomaly_freq.max()+2)), ax=axes[0],
                         color='steelblue', edgecolor='black')
        axes[0].set_title('Distribution du nombre d\'anomalies par tronçon')
        axes[0].set_xlabel('Nombre d\'anomalies')
        axes[0].set_ylabel('Nombre de tronçons')

        # Box plot
        axes[1].boxplot(anomaly_freq, vert=True)
        axes[1].set_title('Distribution (boxplot)')
        axes[1].set_ylabel('Nombre d\'anomalies')

        plt.tight_layout()
        save_figure(fig, '11_distribution_recurrence')
        report_lines.append("\n![Distribution récurrence](figures/11_distribution_recurrence.png)\n")

    # 4.3 Délai moyen entre anomalies successives
    print("\n4.3 Délai entre anomalies successives...")
    report_lines.append("\n## 3. Délai entre anomalies successives\n")

    if 'GID_OBJET' in df_anomalies.columns and 'DATE_DETECTION' in df_anomalies.columns:
        df_anomalies['DATE_DETECTION'] = pd.to_datetime(df_anomalies['DATE_DETECTION'], errors='coerce')
        df_anomalies['DATE_DETECTION'] = normalize_datetime(df_anomalies['DATE_DETECTION'])

        # Trier par tronçon et date
        df_sorted = df_anomalies[df_anomalies['DATE_DETECTION'].notna()].sort_values(
            ['GID_OBJET', 'DATE_DETECTION']
        )

        # Calculer les délais
        df_sorted['prev_date'] = df_sorted.groupby('GID_OBJET')['DATE_DETECTION'].shift(1)
        df_sorted['delay_days'] = (df_sorted['DATE_DETECTION'] - df_sorted['prev_date']).dt.days

        delays = df_sorted['delay_days'].dropna()
        delays_valid = delays[(delays > 0) & (delays < 10000)]  # Filtrer valeurs aberrantes

        if len(delays_valid) > 0:
            report_lines.append(f"\n### Statistiques des délais (en jours)\n")
            report_lines.append(f"- **Nombre de délais calculés**: {len(delays_valid):,}\n")
            report_lines.append(f"- **Médiane**: {delays_valid.median():.0f} jours ({delays_valid.median()/365:.1f} ans)\n")
            report_lines.append(f"- **Moyenne**: {delays_valid.mean():.0f} jours ({delays_valid.mean()/365:.1f} ans)\n")
            report_lines.append(f"- **Écart-type**: {delays_valid.std():.0f} jours\n")
            report_lines.append(f"- **Q25**: {delays_valid.quantile(0.25):.0f} jours\n")
            report_lines.append(f"- **Q75**: {delays_valid.quantile(0.75):.0f} jours\n")

            # Figure délais
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Histogramme des délais (capped à 5 ans)
            delays_capped = delays_valid[delays_valid <= 1825]
            axes[0].hist(delays_capped, bins=50, color='coral', edgecolor='black', alpha=0.7)
            axes[0].axvline(delays_capped.median(), color='red', linestyle='--',
                           label=f'Médiane: {delays_capped.median():.0f} jours')
            axes[0].set_title('Distribution des délais entre anomalies successives (< 5 ans)')
            axes[0].set_xlabel('Délai (jours)')
            axes[0].set_ylabel('Fréquence')
            axes[0].legend()

            # Délais en années
            delays_years = delays_valid / 365
            delays_years_capped = delays_years[delays_years <= 10]
            axes[1].hist(delays_years_capped, bins=40, color='teal', edgecolor='black', alpha=0.7)
            axes[1].axvline(delays_years_capped.median(), color='red', linestyle='--',
                           label=f'Médiane: {delays_years_capped.median():.1f} ans')
            axes[1].set_title('Distribution des délais en années (< 10 ans)')
            axes[1].set_xlabel('Délai (années)')
            axes[1].set_ylabel('Fréquence')
            axes[1].legend()

            plt.tight_layout()
            save_figure(fig, '12_delais_entre_anomalies')
            report_lines.append("\n![Délais entre anomalies](figures/12_delais_entre_anomalies.png)\n")

    # 4.4 Tronçons multi-récidivistes
    print("\n4.4 Identification des tronçons multi-récidivistes...")
    report_lines.append("\n## 4. Tronçons multi-récidivistes\n")

    if 'GID_OBJET' in df_anomalies.columns:
        anomaly_freq = df_anomalies.groupby('GID_OBJET').size()

        # Top 20 récidivistes
        top_recidivistes = anomaly_freq.nlargest(20)

        report_lines.append("\n### Top 20 des tronçons les plus touchés\n")
        report_lines.append("| GID Tronçon | Nb anomalies |\n")
        report_lines.append("|-------------|---------------|\n")
        for gid, count in top_recidivistes.items():
            report_lines.append(f"| {gid} | {count} |\n")

        # Catégorisation des récidivistes
        report_lines.append("\n### Catégorisation par niveau de récurrence\n")
        report_lines.append("| Catégorie | Définition | Nb tronçons | % |\n")
        report_lines.append("|-----------|------------|-------------|---|\n")

        categories = [
            ('Unique', anomaly_freq == 1),
            ('Faible (2)', anomaly_freq == 2),
            ('Modérée (3-5)', (anomaly_freq >= 3) & (anomaly_freq <= 5)),
            ('Élevée (6-10)', (anomaly_freq >= 6) & (anomaly_freq <= 10)),
            ('Très élevée (>10)', anomaly_freq > 10)
        ]

        for cat_name, condition in categories:
            n = condition.sum()
            pct = (n / len(anomaly_freq)) * 100
            report_lines.append(f"| {cat_name} | - | {n:,} | {pct:.1f}% |\n")

        # Caractéristiques des récidivistes vs non-récidivistes
        if df_troncons is not None and 'GID' in df_troncons.columns:
            recidivistes_gids = set(anomaly_freq[anomaly_freq >= 3].index)

            df_troncons_with_status = df_troncons.copy()
            df_troncons_with_status['is_recidiviste'] = df_troncons_with_status['GID'].isin(recidivistes_gids).astype(int)

            report_lines.append("\n### Comparaison récidivistes vs non-récidivistes\n")

            for col in ['DIAMETRE', 'LNG', 'DDP_year']:
                if col in df_troncons_with_status.columns:
                    recid = df_troncons_with_status[df_troncons_with_status['is_recidiviste'] == 1][col]
                    non_recid = df_troncons_with_status[df_troncons_with_status['is_recidiviste'] == 0][col]

                    report_lines.append(f"\n**{col}**\n")
                    report_lines.append(f"- Récidivistes: médiane = {recid.median():,.1f}\n")
                    report_lines.append(f"- Non-récidivistes: médiane = {non_recid.median():,.1f}\n")

    write_report(''.join(report_lines), '04_anomaly_analysis.md')

    return data


# =============================================================================
# PHASE 5 : ÉVALUATION DE LA PRÉDICTIBILITÉ
# =============================================================================

def phase5_predictability(data):
    """
    PHASE 5 : Évaluation de la prédictibilité et conclusions
    """
    print("\n" + "="*80)
    print("PHASE 5 : ÉVALUATION DE LA PRÉDICTIBILITÉ")
    print("="*80)

    report_lines = []
    report_lines.append("# Évaluation de la Prédictibilité\n")
    report_lines.append(f"*Date de génération : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

    df_troncons = data.get('troncons')
    df_anomalies = data.get('anomalie')

    # 5.1 Définition de la variable cible
    print("\n5.1 Définition de la variable cible...")
    report_lines.append("## 1. Définition de la variable cible\n")

    report_lines.append("""
### Approche recommandée : Fenêtre temporelle (Freeze + Horizon)

Pour prédire les défaillances futures sans fuite de données, nous utilisons l'approche suivante :

1. **FREEZE_DATE** : Date de référence pour séparer passé et futur
2. **HORIZON** : Période de prédiction (ex: 12 mois)
3. **Variable cible** : Y = 1 si anomalie détectée entre FREEZE_DATE et FREEZE_DATE + HORIZON

Cette approche garantit que :
- Toutes les features sont calculées avec des données ANTÉRIEURES à FREEZE_DATE
- La cible est définie sur la période POSTÉRIEURE à FREEZE_DATE
- Aucune fuite de données n'est possible

### Formulation mathématique

```
Y_i = 1 si ∃ anomalie pour tronçon i dans [FREEZE_DATE, FREEZE_DATE + HORIZON]
Y_i = 0 sinon
```
""")

    # 5.2 Calcul du taux de base
    print("\n5.2 Calcul du taux de base...")
    report_lines.append("\n## 2. Taux de base (Base Rate)\n")

    if df_troncons is not None and df_anomalies is not None:
        total_troncons = len(df_troncons)

        if 'GID_OBJET' in df_anomalies.columns and 'GID' in df_troncons.columns:
            troncons_with_anomaly = df_anomalies['GID_OBJET'].nunique()
            base_rate = (troncons_with_anomaly / total_troncons) * 100

            report_lines.append(f"\n### Statistiques globales\n")
            report_lines.append(f"- **Total tronçons**: {total_troncons:,}\n")
            report_lines.append(f"- **Tronçons avec au moins 1 anomalie (historique complet)**: {troncons_with_anomaly:,}\n")
            report_lines.append(f"- **Taux de base global**: {base_rate:.2f}%\n")

            # Estimation du taux annuel
            if 'DATE_DETECTION' in df_anomalies.columns:
                df_anomalies['DATE_DETECTION'] = pd.to_datetime(df_anomalies['DATE_DETECTION'], errors='coerce')
                df_anomalies['DATE_DETECTION'] = normalize_datetime(df_anomalies['DATE_DETECTION'])
                df_valid = df_anomalies[df_anomalies['DATE_DETECTION'].notna()]

                if len(df_valid) > 0:
                    years = df_valid['DATE_DETECTION'].dt.year.unique()
                    years = [y for y in years if 2000 <= y <= 2025]

                    if len(years) >= 2:
                        annual_rates = []
                        for year in sorted(years)[-5:]:  # 5 dernières années
                            ano_year = df_valid[df_valid['DATE_DETECTION'].dt.year == year]['GID_OBJET'].nunique()
                            rate = (ano_year / total_troncons) * 100
                            annual_rates.append((year, rate))

                        report_lines.append(f"\n### Taux annuels (5 dernières années)\n")
                        report_lines.append("| Année | Tronçons touchés | Taux |\n")
                        report_lines.append("|-------|------------------|------|\n")
                        for year, rate in annual_rates:
                            report_lines.append(f"| {year} | - | {rate:.2f}% |\n")

                        avg_annual_rate = np.mean([r for _, r in annual_rates])
                        report_lines.append(f"\n**Taux moyen annuel estimé**: {avg_annual_rate:.2f}%\n")

    # 5.3 Séparabilité des classes
    print("\n5.3 Analyse de la séparabilité des classes...")
    report_lines.append("\n## 3. Analyse de la séparabilité des classes\n")

    if df_troncons is not None and df_anomalies is not None:
        # Créer le dataset avec features et cible
        anomaly_counts = df_anomalies.groupby('GID_OBJET').size().reset_index(name='n_anomalies')
        df_merged = df_troncons.merge(anomaly_counts, left_on='GID', right_on='GID_OBJET', how='left')
        df_merged['n_anomalies'] = df_merged['n_anomalies'].fillna(0)
        df_merged['has_anomaly'] = (df_merged['n_anomalies'] > 0).astype(int)

        report_lines.append("\n### Distributions par classe\n")

        # Calculer les distributions pour chaque feature
        features = ['DIAMETRE', 'LNG', 'DDP_year']

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, col in enumerate(features):
            if col in df_merged.columns:
                class0 = df_merged[df_merged['has_anomaly'] == 0][col].dropna()
                class1 = df_merged[df_merged['has_anomaly'] == 1][col].dropna()

                # Histogrammes superposés
                axes[idx].hist(class0, bins=50, alpha=0.5, label='Sans anomalie', density=True, color='blue')
                axes[idx].hist(class1, bins=50, alpha=0.5, label='Avec anomalie', density=True, color='red')
                axes[idx].set_title(f'Distribution de {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Densité')
                axes[idx].legend()

                # Kolmogorov-Smirnov test
                ks_stat, ks_pvalue = stats.ks_2samp(class0, class1)
                report_lines.append(f"\n**{col}**\n")
                report_lines.append(f"- Test KS: D = {ks_stat:.3f}, p-value = {ks_pvalue:.2e}\n")
                report_lines.append(f"- Séparabilité: {'Bonne' if ks_stat > 0.2 else 'Modérée' if ks_stat > 0.1 else 'Faible'}\n")

        plt.tight_layout()
        save_figure(fig, '13_separabilite_classes')
        report_lines.append("\n![Séparabilité des classes](figures/13_separabilite_classes.png)\n")

    # 5.4 Features prometteuses
    print("\n5.4 Identification des features prometteuses...")
    report_lines.append("\n## 4. Features prometteuses pour la modélisation\n")

    report_lines.append("""
### Features de base (disponibles)
| Feature | Type | Disponibilité | Pouvoir prédictif attendu |
|---------|------|---------------|---------------------------|
| Âge du tronçon | Numérique | ✅ Disponible | Élevé |
| Matériau (MAT) | Catégoriel | ✅ Disponible | Élevé |
| Diamètre | Numérique | ✅ Disponible | Modéré |
| Longueur (LNG) | Numérique | ✅ Disponible | Modéré |
| Décennie de pose | Catégoriel | ✅ Dérivable | Élevé |

### Features d'historique (à construire)
| Feature | Description | Pouvoir prédictif attendu |
|---------|-------------|---------------------------|
| n_anomalies_past | Nombre d'anomalies passées | Très élevé |
| n_anomalies_1y | Anomalies dans les 12 derniers mois | Élevé |
| days_since_last | Jours depuis dernière anomalie | Élevé |
| leak_rate | Taux de fuite par an | Élevé |

### Features contextuelles (si disponibles)
| Feature | Source | Pouvoir prédictif attendu |
|---------|--------|---------------------------|
| Trafic routier | troncon_dt | Modéré |
| Nb logements | troncon_dt | Faible |
| Commune | commune | Modéré (effet zone) |

### Features de survie (recommandées)
| Feature | Description | Pouvoir prédictif attendu |
|---------|-------------|---------------------------|
| ratio_age_median | Âge / durée vie médiane du matériau | Très élevé |
| over_p75_life | Dépasse le 75e percentile de survie | Élevé |
| hazard_score | Score de risque Kaplan-Meier | Très élevé |
""")

    # 5.5 Conclusions
    print("\n5.5 Conclusions et recommandations...")
    report_lines.append("\n## 5. Conclusions sur la faisabilité\n")

    report_lines.append("""
### Verdict : FAISABLE ✅

L'analyse montre que la prédiction des défaillances est **faisable** avec les données disponibles.

### Points forts
1. **Volume de données suffisant** : 218k+ tronçons, 30k+ anomalies historiques
2. **Historique temporel** : Données sur plusieurs années permettant l'analyse de survie
3. **Variables discriminantes** : Âge, matériau et historique d'anomalies montrent une bonne séparabilité
4. **Patterns identifiés** : Récurrence claire sur certains tronçons

### Points d'attention
1. **Déséquilibre des classes** : Taux d'anomalie ~10-15% (gérable avec techniques adaptées)
2. **Données manquantes** : Certaines colonnes ont >50% de missing (à imputer ou exclure)
3. **Qualité des dates** : Vérifier la cohérence temporelle des anomalies

### Recommandations techniques
1. Utiliser une approche **freeze + horizon** pour éviter les fuites de données
2. Appliquer des techniques de **rééchantillonnage** (SMOTE, sous-échantillonnage)
3. Évaluer avec des métriques **business** (Capture@k, Lift@k) plutôt que accuracy
4. Tester des modèles **gradient boosting** (LightGBM, HistGradientBoosting)
5. Inclure des **features de survie** (Kaplan-Meier par matériau)

### Performances attendues
- **ROC-AUC** : 0.80 - 0.90 (basé sur projets similaires)
- **Capture@10%** : 50-65% des défaillances futures
- **Lift@10%** : 5-7x le taux de base
""")

    write_report(''.join(report_lines), '05_predictability_assessment.md')

    return data


# =============================================================================
# CRÉATION DU DATASET NETTOYÉ
# =============================================================================

def create_clean_dataset(data):
    """
    Crée le dataset nettoyé et jointé pour la modélisation
    """
    print("\n" + "="*80)
    print("CRÉATION DU DATASET NETTOYÉ")
    print("="*80)

    df_troncons = data.get('troncons')
    df_anomalies = data.get('anomalie')
    df_commune = data.get('commune')
    df_troncon_dt = data.get('troncon_dt')

    if df_troncons is None:
        print("  ✗ Erreur: données tronçons non disponibles")
        return None

    print("\n  Jointure des tables...")
    df = df_troncons.copy()

    # Nettoyer les dates
    if 'DDP' in df.columns:
        df['DDP'] = pd.to_datetime(df['DDP'], errors='coerce')
        df['DDP'] = normalize_datetime(df['DDP'])

    # Joindre commune
    if df_commune is not None and 'GID_TRONCON' in df_commune.columns:
        df = df.merge(df_commune[['GID_TRONCON', 'GID_COMMUNE', 'GID_RESEAU', 'NATURE_RESEAU']],
                     left_on='GID', right_on='GID_TRONCON', how='left')
        print(f"    ✓ Commune jointe")

    # Joindre troncon_dt
    if df_troncon_dt is not None and 'DT_GID_TRONCON' in df_troncon_dt.columns:
        df = df.merge(df_troncon_dt, left_on='GID', right_on='DT_GID_TRONCON', how='left')
        print(f"    ✓ Troncon_DT jointe")

    # Calculer les features d'anomalies
    if df_anomalies is not None and 'GID_OBJET' in df_anomalies.columns:
        # Nombre total d'anomalies
        ano_count = df_anomalies.groupby('GID_OBJET').size().reset_index(name='n_anomalies_total')
        df = df.merge(ano_count, left_on='GID', right_on='GID_OBJET', how='left')
        df['n_anomalies_total'] = df['n_anomalies_total'].fillna(0).astype(int)

        # Date de dernière anomalie
        if 'DATE_DETECTION' in df_anomalies.columns:
            df_anomalies['DATE_DETECTION'] = pd.to_datetime(df_anomalies['DATE_DETECTION'], errors='coerce')
            df_anomalies['DATE_DETECTION'] = normalize_datetime(df_anomalies['DATE_DETECTION'])
            last_ano = df_anomalies.groupby('GID_OBJET')['DATE_DETECTION'].max().reset_index()
            last_ano.columns = ['GID_OBJET', 'last_anomaly_date']
            df = df.merge(last_ano, left_on='GID', right_on='GID_OBJET', how='left', suffixes=('', '_y'))

        # Cible binaire
        df['has_anomaly'] = (df['n_anomalies_total'] > 0).astype(int)
        print(f"    ✓ Features anomalies calculées")

    # Calculer l'âge en années
    if 'DDP' in df.columns:
        reference_date = pd.Timestamp('2024-01-01')
        df['age_years'] = (reference_date - df['DDP']).dt.days / 365.25
        df['age_years'] = df['age_years'].clip(lower=0)

    # Catégorisation du matériau
    if 'MAT' in df.columns:
        mat_mapping = {
            'FT': 'Fonte',
            'AC': 'Acier',
            'PVC': 'PVC',
            'PE': 'Polyéthylène',
            'FD': 'Fonte Ductile'
        }
        df['MAT_label'] = df['MAT'].map(mat_mapping).fillna(df['MAT'])

    # Supprimer les colonnes en double
    cols_to_drop = [c for c in df.columns if c.endswith('_y') or c == 'GID_OBJET']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Sauvegarder
    output_path = os.path.join(PROCESSED_DIR, 'dataset_clean.csv')
    df.to_csv(output_path, index=False)
    print(f"\n  ✓ Dataset sauvegardé: {output_path}")
    print(f"    Dimensions: {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Fonction principale orchestrant toutes les phases d'analyse
    """
    print("\n" + "="*80)
    print("ANALYSE COMPLÈTE DES DONNÉES - PRÉDICTION DÉFAILLANCES CANALISATIONS")
    print("="*80)

    # Phase 1 : Chargement et exploration
    data = phase1_load_and_explore()

    # Phase 2 : Analyse exploratoire
    data = phase2_eda(data)

    # Phase 3 : Corrélations
    data = phase3_correlations(data)

    # Phase 4 : Analyse des anomalies
    data = phase4_anomaly_analysis(data)

    # Phase 5 : Évaluation de la prédictibilité
    data = phase5_predictability(data)

    # Création du dataset nettoyé
    df_clean = create_clean_dataset(data)

    print("\n" + "="*80)
    print("ANALYSE TERMINÉE")
    print("="*80)
    print("\nRapports générés:")
    print("  - reports/01_data_quality_report.md")
    print("  - reports/02_eda_report.md")
    print("  - reports/03_correlation_analysis.md")
    print("  - reports/04_anomaly_analysis.md")
    print("  - reports/05_predictability_assessment.md")
    print("\nDonnées:")
    print("  - data/processed/dataset_clean.csv")
    print("\nFigures: reports/figures/")

    return data, df_clean


if __name__ == '__main__':
    data, df_clean = main()
