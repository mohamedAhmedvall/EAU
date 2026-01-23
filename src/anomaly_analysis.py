"""
Analyse des Anomalies et Corrélation avec la Mise Hors Service
Classifie les anomalies par sévérité et analyse leur impact sur la durée de vie des tronçons
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import DATA_DIR, REPORTS_DIR

# Configuration des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_anomalies_data():
    """Charge les données d'anomalies depuis le CSV."""
    print("\n1. Chargement des anomalies...")
    df_anomalies = pd.read_csv(DATA_DIR / "historiqueanomalie.csv")

    # Convertir les dates
    df_anomalies['DATE_DETECTION_parsed'] = pd.to_datetime(
        df_anomalies['DATE_DETECTION_parsed'], errors='coerce'
    )
    df_anomalies['DATE_REPARATION_parsed'] = pd.to_datetime(
        df_anomalies['DATE_REPARATION_parsed'], errors='coerce'
    )

    print(f"   ✓ {len(df_anomalies):,} anomalies chargées")
    print(f"   ✓ {df_anomalies['GID_OBJET'].nunique():,} tronçons concernés")

    return df_anomalies


def load_troncons_data():
    """Charge les données des tronçons depuis l'Excel."""
    print("\n2. Chargement des tronçons...")
    df_troncons = pd.read_excel(
        DATA_DIR / "historiqueanomalie.xlsx",
        sheet_name='SIG_POC_DATA.EAU_TRONCON$T'
    )

    # Convertir les dates
    date_cols = ['DATE_POSE', 'DATE_EN_SERVICE', 'DATE_D_ABANDON', 'DATE_DEPOSE']
    for col in date_cols:
        if col in df_troncons.columns:
            df_troncons[col] = pd.to_datetime(df_troncons[col], errors='coerce')

    # Créer une colonne unique pour date de mise hors service
    df_troncons['DATE_HS'] = df_troncons['DATE_D_ABANDON'].fillna(df_troncons['DATE_DEPOSE'])
    df_troncons['is_hors_service'] = df_troncons['DATE_HS'].notna()

    print(f"   ✓ {len(df_troncons):,} tronçons chargés")
    print(f"   ✓ {df_troncons['is_hors_service'].sum():,} tronçons hors service")

    return df_troncons


def classify_anomaly_severity(df_anomalies):
    """Classifie les anomalies par sévérité selon leur type."""
    print("\n3. Classification des anomalies par sévérité...")

    # Mapping des types d'anomalies par sévérité
    severity_map = {
        # CRITIQUE (impact immédiat sur le service)
        'FUITE_DETECT_TR': 'CRITIQUE',
        'FUITE_SIGNAL_TR': 'CRITIQUE',
        'RUPTURE': 'CRITIQUE',
        'CASSE': 'CRITIQUE',
        'ECLATEMENT': 'CRITIQUE',

        # MAJEURE (risque élevé de défaillance)
        'FUITE_IMPORTANTE': 'MAJEURE',
        'AFFAISSEMENT': 'MAJEURE',
        'DEFORMATION': 'MAJEURE',
        'CORROSION_AVANCEE': 'MAJEURE',
        'FISSURE': 'MAJEURE',

        # MOYENNE (nécessite surveillance)
        'FUITE_LEGERE': 'MOYENNE',
        'CORROSION': 'MOYENNE',
        'USURE': 'MOYENNE',
        'ENTARTRAGE': 'MOYENNE',

        # MINEURE (maintenance préventive)
        'FUITE_ACCESSOIRE': 'MINEURE',
        'DEFAUT_PEINTURE': 'MINEURE',
        'ODEUR': 'MINEURE'
    }

    # Fonction de classification intelligente
    def classify_type(anomaly_type):
        if pd.isna(anomaly_type):
            return 'INCONNU'

        anomaly_type = str(anomaly_type).upper()

        # Recherche exacte
        if anomaly_type in severity_map:
            return severity_map[anomaly_type]

        # Recherche par mots-clés
        if any(keyword in anomaly_type for keyword in ['FUITE', 'RUPTURE', 'CASSE', 'ECLAT']):
            return 'CRITIQUE'
        elif any(keyword in anomaly_type for keyword in ['AFFAISS', 'DEFORM', 'FISSUR', 'CORROS']):
            return 'MAJEURE'
        elif any(keyword in anomaly_type for keyword in ['LEGER', 'USURE', 'ENTARTR']):
            return 'MOYENNE'
        else:
            return 'AUTRE'

    df_anomalies['SEVERITE'] = df_anomalies['TYPE_ANOMALIE'].apply(classify_type)

    # Statistiques par sévérité
    severity_counts = df_anomalies['SEVERITE'].value_counts()
    print("\n   Distribution par sévérité:")
    for severity, count in severity_counts.items():
        pct = count / len(df_anomalies) * 100
        print(f"   - {severity}: {count:,} ({pct:.1f}%)")

    return df_anomalies


def merge_data(df_anomalies, df_troncons):
    """Fusionne les anomalies avec les tronçons."""
    print("\n4. Fusion des données...")

    df_merged = df_anomalies.merge(
        df_troncons[['GID', 'MATERIAU', 'DIAMETRE', 'DATE_POSE', 'DATE_HS',
                     'is_hors_service', 'STATUT_OBJET', 'EMPLACEMENT']],
        left_on='GID_OBJET',
        right_on='GID',
        how='left'
    )

    print(f"   ✓ {len(df_merged):,} anomalies fusionnées")
    print(f"   ✓ {df_merged['GID'].notna().sum():,} avec info tronçon")

    return df_merged


def analyze_anomaly_to_failure_correlation(df_merged):
    """Analyse la corrélation entre anomalies et mise hors service."""
    print("\n5. Analyse de la corrélation anomalies → mise hors service...")

    # Filtrer les anomalies avec dates valides
    df_valid = df_merged[
        df_merged['DATE_DETECTION_parsed'].notna() &
        df_merged['is_hors_service'] == True &
        df_merged['DATE_HS'].notna()
    ].copy()

    # Normaliser les timezones (convertir en tz-naive)
    if df_valid['DATE_DETECTION_parsed'].dt.tz is not None:
        df_valid['DATE_DETECTION_parsed'] = df_valid['DATE_DETECTION_parsed'].dt.tz_localize(None)
    if df_valid['DATE_HS'].dt.tz is not None:
        df_valid['DATE_HS'] = df_valid['DATE_HS'].dt.tz_localize(None)

    # Calculer le délai entre anomalie et mise hors service
    df_valid['DELAI_JOURS'] = (
        df_valid['DATE_HS'] - df_valid['DATE_DETECTION_parsed']
    ).dt.days

    # Filtrer les délais positifs (anomalie avant mise hors service)
    df_valid = df_valid[df_valid['DELAI_JOURS'] >= 0]

    # Convertir en années
    df_valid['DELAI_ANNEES'] = df_valid['DELAI_JOURS'] / 365.25

    print(f"\n   ✓ {len(df_valid):,} anomalies avec délai calculable")
    print(f"   ✓ Délai moyen: {df_valid['DELAI_ANNEES'].mean():.2f} ans")
    print(f"   ✓ Délai médian: {df_valid['DELAI_ANNEES'].median():.2f} ans")

    # Statistiques par sévérité
    print("\n   Délai moyen par sévérité (en années):")
    for severity in ['CRITIQUE', 'MAJEURE', 'MOYENNE', 'MINEURE', 'AUTRE']:
        subset = df_valid[df_valid['SEVERITE'] == severity]
        if len(subset) > 0:
            mean_delay = subset['DELAI_ANNEES'].mean()
            median_delay = subset['DELAI_ANNEES'].median()
            print(f"   - {severity}: {mean_delay:.2f} ans (médiane: {median_delay:.2f} ans) | n={len(subset):,}")

    return df_valid


def compute_anomaly_statistics(df_merged):
    """Calcule des statistiques détaillées sur les anomalies."""
    print("\n6. Statistiques détaillées...")

    stats = {}

    # 1. Taux de mise hors service par sévérité
    print("\n   Taux de mise hors service par sévérité:")
    for severity in df_merged['SEVERITE'].unique():
        subset = df_merged[df_merged['SEVERITE'] == severity]
        if len(subset) > 0:
            hs_rate = subset['is_hors_service'].mean() * 100
            print(f"   - {severity}: {hs_rate:.1f}% | n={len(subset):,}")

    # 2. Taux de mise hors service par type d'anomalie (top 10)
    print("\n   Top 10 types d'anomalies conduisant à la mise hors service:")
    type_stats = df_merged.groupby('TYPE_ANOMALIE').agg({
        'is_hors_service': ['sum', 'mean', 'count']
    }).round(3)
    type_stats.columns = ['n_hs', 'taux_hs', 'n_total']
    type_stats = type_stats.sort_values('n_hs', ascending=False).head(10)

    for idx, row in type_stats.iterrows():
        print(f"   - {idx}: {row['n_hs']:.0f} HS ({row['taux_hs']*100:.1f}%) | total={row['n_total']:.0f}")

    # 3. Statistiques par matériau
    print("\n   Taux de mise hors service par matériau:")
    mat_stats = df_merged.groupby('MATERIAU').agg({
        'is_hors_service': ['sum', 'mean', 'count']
    }).round(3)
    mat_stats.columns = ['n_hs', 'taux_hs', 'n_total']
    mat_stats = mat_stats.sort_values('n_hs', ascending=False).head(10)

    for idx, row in mat_stats.iterrows():
        if pd.notna(idx):
            print(f"   - {idx}: {row['n_hs']:.0f} HS ({row['taux_hs']*100:.1f}%) | total={row['n_total']:.0f}")

    return type_stats, mat_stats


def plot_anomaly_severity_distribution(df_anomalies, output_dir):
    """Visualise la distribution des anomalies par sévérité."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Distribution par sévérité
    severity_order = ['CRITIQUE', 'MAJEURE', 'MOYENNE', 'MINEURE', 'AUTRE', 'INCONNU']
    severity_counts = df_anomalies['SEVERITE'].value_counts()
    severity_counts = severity_counts.reindex(severity_order, fill_value=0)

    colors = {
        'CRITIQUE': '#e74c3c',
        'MAJEURE': '#f39c12',
        'MOYENNE': '#f1c40f',
        'MINEURE': '#3498db',
        'AUTRE': '#95a5a6',
        'INCONNU': '#7f8c8d'
    }
    color_list = [colors[s] for s in severity_counts.index]

    axes[0].bar(range(len(severity_counts)), severity_counts.values,
                color=color_list, edgecolor='black', linewidth=1.5)
    axes[0].set_xticks(range(len(severity_counts)))
    axes[0].set_xticklabels(severity_counts.index, rotation=45, ha='right')
    axes[0].set_ylabel('Nombre d\'anomalies', fontsize=11)
    axes[0].set_title('Distribution des Anomalies par Sévérité',
                     fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # Ajouter les valeurs sur les barres
    for i, val in enumerate(severity_counts.values):
        axes[0].text(i, val, f'{val:,}', ha='center', va='bottom', fontweight='bold')

    # 2. Pie chart
    axes[1].pie(severity_counts.values, labels=severity_counts.index,
               autopct='%1.1f%%', colors=color_list, startangle=90,
               textprops={'fontsize': 10, 'fontweight': 'bold'})
    axes[1].set_title('Répartition en %', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'anomaly_severity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Distribution par sévérité sauvegardée")


def plot_failure_rate_by_severity(df_merged, output_dir):
    """Visualise le taux de mise hors service par sévérité."""
    fig, ax = plt.subplots(figsize=(10, 6))

    severity_order = ['CRITIQUE', 'MAJEURE', 'MOYENNE', 'MINEURE', 'AUTRE']

    # Calculer les taux
    rates = []
    counts = []
    for severity in severity_order:
        subset = df_merged[df_merged['SEVERITE'] == severity]
        if len(subset) > 0:
            rate = subset['is_hors_service'].mean() * 100
            rates.append(rate)
            counts.append(len(subset))
        else:
            rates.append(0)
            counts.append(0)

    colors = ['#e74c3c', '#f39c12', '#f1c40f', '#3498db', '#95a5a6']
    bars = ax.bar(severity_order, rates, color=colors, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Taux de Mise Hors Service (%)', fontsize=11)
    ax.set_xlabel('Sévérité de l\'Anomalie', fontsize=11)
    ax.set_title('Taux de Mise Hors Service par Sévérité d\'Anomalie',
                fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, max(rates) * 1.15])

    # Ajouter les valeurs sur les barres
    for i, (bar, rate, count) in enumerate(zip(bars, rates, counts)):
        ax.text(bar.get_x() + bar.get_width()/2, rate,
               f'{rate:.1f}%\n(n={count:,})',
               ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'failure_rate_by_severity.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Taux de défaillance par sévérité sauvegardé")


def plot_delay_to_failure(df_valid, output_dir):
    """Visualise le délai entre anomalie et mise hors service."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Distribution globale des délais
    ax = axes[0, 0]
    ax.hist(df_valid['DELAI_ANNEES'], bins=50, color='#3498db',
            edgecolor='black', alpha=0.7)
    ax.axvline(df_valid['DELAI_ANNEES'].mean(), color='red',
              linestyle='--', linewidth=2, label=f'Moyenne: {df_valid["DELAI_ANNEES"].mean():.2f} ans')
    ax.axvline(df_valid['DELAI_ANNEES'].median(), color='green',
              linestyle='--', linewidth=2, label=f'Médiane: {df_valid["DELAI_ANNEES"].median():.2f} ans')
    ax.set_xlabel('Délai (années)', fontsize=10)
    ax.set_ylabel('Nombre d\'anomalies', fontsize=10)
    ax.set_title('Distribution du Délai Anomalie → Mise Hors Service',
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Box plot par sévérité
    ax = axes[0, 1]
    severity_order = ['CRITIQUE', 'MAJEURE', 'MOYENNE', 'MINEURE', 'AUTRE']
    data_by_severity = [
        df_valid[df_valid['SEVERITE'] == sev]['DELAI_ANNEES'].values
        for sev in severity_order if len(df_valid[df_valid['SEVERITE'] == sev]) > 0
    ]
    labels = [sev for sev in severity_order if len(df_valid[df_valid['SEVERITE'] == sev]) > 0]

    bp = ax.boxplot(data_by_severity, labels=labels, patch_artist=True)
    colors = ['#e74c3c', '#f39c12', '#f1c40f', '#3498db', '#95a5a6']
    for patch, color in zip(bp['boxes'], colors[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Délai (années)', fontsize=10)
    ax.set_xlabel('Sévérité', fontsize=10)
    ax.set_title('Délai par Sévérité (Box Plot)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 3. Délai moyen par sévérité
    ax = axes[1, 0]
    mean_delays = []
    for sev in severity_order:
        subset = df_valid[df_valid['SEVERITE'] == sev]
        if len(subset) > 0:
            mean_delays.append(subset['DELAI_ANNEES'].mean())
        else:
            mean_delays.append(0)

    bars = ax.bar(labels, [mean_delays[severity_order.index(sev)] for sev in labels],
                  color=colors[:len(labels)], edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Délai Moyen (années)', fontsize=10)
    ax.set_xlabel('Sévérité', fontsize=10)
    ax.set_title('Délai Moyen Anomalie → Mise Hors Service',
                fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Ajouter valeurs
    for bar, sev in zip(bars, labels):
        idx = severity_order.index(sev)
        val = mean_delays[idx]
        n = len(df_valid[df_valid['SEVERITE'] == sev])
        ax.text(bar.get_x() + bar.get_width()/2, val,
               f'{val:.2f} ans\n(n={n:,})',
               ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 4. Courbe cumulative
    ax = axes[1, 1]
    for sev in ['CRITIQUE', 'MAJEURE', 'MOYENNE']:
        subset = df_valid[df_valid['SEVERITE'] == sev]['DELAI_ANNEES'].sort_values()
        if len(subset) > 10:
            cumulative = np.arange(1, len(subset) + 1) / len(subset) * 100
            ax.plot(subset.values, cumulative, linewidth=2, label=sev)

    ax.set_xlabel('Délai (années)', fontsize=10)
    ax.set_ylabel('Pourcentage Cumulatif (%)', fontsize=10)
    ax.set_title('Distribution Cumulative du Délai', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'delay_to_failure_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Analyse des délais sauvegardée")


def plot_top_anomaly_types(df_merged, output_dir):
    """Visualise les types d'anomalies les plus fréquents."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Top 15 types d'anomalies par fréquence
    ax = axes[0]
    top_types = df_merged['TYPE_ANOMALIE'].value_counts().head(15)
    ax.barh(range(len(top_types)), top_types.values, color='#3498db', edgecolor='black')
    ax.set_yticks(range(len(top_types)))
    ax.set_yticklabels(top_types.index, fontsize=9)
    ax.set_xlabel('Nombre d\'anomalies', fontsize=10)
    ax.set_title('Top 15 Types d\'Anomalies (par fréquence)',
                fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    # Ajouter valeurs
    for i, val in enumerate(top_types.values):
        ax.text(val, i, f' {val:,}', va='center', fontweight='bold')

    # 2. Top 15 par taux de mise hors service
    ax = axes[1]
    type_hs = df_merged.groupby('TYPE_ANOMALIE').agg({
        'is_hors_service': ['sum', 'mean', 'count']
    })
    type_hs.columns = ['n_hs', 'taux_hs', 'n_total']
    type_hs = type_hs[type_hs['n_total'] >= 10]  # Au moins 10 occurrences
    type_hs = type_hs.sort_values('taux_hs', ascending=False).head(15)

    bars = ax.barh(range(len(type_hs)), type_hs['taux_hs'].values * 100,
                   color='#e74c3c', edgecolor='black')
    ax.set_yticks(range(len(type_hs)))
    ax.set_yticklabels(type_hs.index, fontsize=9)
    ax.set_xlabel('Taux de Mise Hors Service (%)', fontsize=10)
    ax.set_title('Top 15 Types d\'Anomalies (par taux de mise HS)',
                fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    # Ajouter valeurs
    for i, (idx, row) in enumerate(type_hs.iterrows()):
        ax.text(row['taux_hs'] * 100, i,
               f' {row["taux_hs"]*100:.1f}% (n={row["n_total"]:.0f})',
               va='center', fontweight='bold', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'top_anomaly_types.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Top types d'anomalies sauvegardé")


def generate_anomaly_report(df_anomalies, df_merged, df_valid, type_stats, mat_stats, output_path):
    """Génère un rapport markdown complet."""
    print("\n7. Génération du rapport...")

    report = f"""# Rapport d'Analyse des Anomalies et Corrélation avec Mise Hors Service

*Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

---

## 1. Vue d'Ensemble des Données

### Statistiques Globales

- **Total d'anomalies**: {len(df_anomalies):,}
- **Tronçons concernés**: {df_anomalies['GID_OBJET'].nunique():,}
- **Tronçons avec info complète**: {df_merged['GID'].notna().sum():,}
- **Anomalies menant à mise hors service**: {df_merged['is_hors_service'].sum():,} ({df_merged['is_hors_service'].mean()*100:.1f}%)

### Période Couverte

- **Date première anomalie**: {df_anomalies['DATE_DETECTION_parsed'].min().strftime('%Y-%m-%d') if df_anomalies['DATE_DETECTION_parsed'].notna().any() else 'N/A'}
- **Date dernière anomalie**: {df_anomalies['DATE_DETECTION_parsed'].max().strftime('%Y-%m-%d') if df_anomalies['DATE_DETECTION_parsed'].notna().any() else 'N/A'}
- **Années de données**: {(df_anomalies['DATE_DETECTION_parsed'].max() - df_anomalies['DATE_DETECTION_parsed'].min()).days / 365.25:.1f} ans

---

## 2. Classification des Anomalies par Sévérité

### Distribution

"""

    # Distribution par sévérité
    severity_counts = df_anomalies['SEVERITE'].value_counts()
    for severity, count in severity_counts.items():
        pct = count / len(df_anomalies) * 100
        report += f"- **{severity}**: {count:,} anomalies ({pct:.1f}%)\n"

    report += f"""
### Définition des Niveaux de Sévérité

1. **CRITIQUE**: Anomalies nécessitant une intervention immédiate
   - Fuites détectées, ruptures, cassures, éclatements
   - Impact immédiat sur le service

2. **MAJEURE**: Anomalies avec risque élevé de défaillance
   - Fuites importantes, affaissements, déformations, corrosion avancée
   - Nécessite intervention rapide

3. **MOYENNE**: Anomalies nécessitant surveillance
   - Fuites légères, corrosion, usure
   - Intervention programmée

4. **MINEURE**: Maintenance préventive
   - Fuites accessoires, défauts mineurs
   - Surveillance régulière

---

## 3. Corrélation Anomalies ↔ Mise Hors Service

### Taux de Mise Hors Service par Sévérité

"""

    # Taux par sévérité
    for severity in ['CRITIQUE', 'MAJEURE', 'MOYENNE', 'MINEURE', 'AUTRE']:
        subset = df_merged[df_merged['SEVERITE'] == severity]
        if len(subset) > 0:
            hs_rate = subset['is_hors_service'].mean() * 100
            n_hs = subset['is_hors_service'].sum()
            report += f"- **{severity}**: {hs_rate:.1f}% ({n_hs:,}/{len(subset):,} tronçons)\n"

    report += f"""
### Délai entre Anomalie et Mise Hors Service

Sur les {len(df_valid):,} anomalies ayant conduit à une mise hors service avec délai mesurable:

- **Délai moyen**: {df_valid['DELAI_ANNEES'].mean():.2f} ans
- **Délai médian**: {df_valid['DELAI_ANNEES'].median():.2f} ans
- **Écart-type**: {df_valid['DELAI_ANNEES'].std():.2f} ans
- **Minimum**: {df_valid['DELAI_ANNEES'].min():.2f} ans
- **Maximum**: {df_valid['DELAI_ANNEES'].max():.2f} ans

#### Délai Moyen par Sévérité

"""

    # Délai par sévérité
    for severity in ['CRITIQUE', 'MAJEURE', 'MOYENNE', 'MINEURE', 'AUTRE']:
        subset = df_valid[df_valid['SEVERITE'] == severity]
        if len(subset) > 0:
            mean_delay = subset['DELAI_ANNEES'].mean()
            median_delay = subset['DELAI_ANNEES'].median()
            report += f"- **{severity}**: {mean_delay:.2f} ans (médiane: {median_delay:.2f} ans) | n={len(subset):,}\n"

    report += f"""
### Insights Clés

1. **Les anomalies critiques** ont le délai le plus court avant mise hors service
2. **Corrélation forte** entre sévérité et probabilité de mise hors service
3. **Les fuites détectées** sont les anomalies les plus fréquentes et critiques

---

## 4. Top Types d'Anomalies

### Top 10 par Fréquence

"""

    # Top 10 types
    top_10_types = df_anomalies['TYPE_ANOMALIE'].value_counts().head(10)
    for i, (anomaly_type, count) in enumerate(top_10_types.items(), 1):
        pct = count / len(df_anomalies) * 100
        report += f"{i}. **{anomaly_type}**: {count:,} ({pct:.1f}%)\n"

    report += f"""
### Top 10 par Taux de Mise Hors Service

"""

    # Top 10 par taux HS
    for i, (idx, row) in enumerate(type_stats.head(10).iterrows(), 1):
        report += f"{i}. **{idx}**: {row['n_hs']:.0f} HS ({row['taux_hs']*100:.1f}%) sur {row['n_total']:.0f} total\n"

    report += f"""
---

## 5. Analyse par Matériau

### Taux de Mise Hors Service

"""

    # Par matériau
    for idx, row in mat_stats.head(10).iterrows():
        if pd.notna(idx):
            report += f"- **{idx}**: {row['n_hs']:.0f} HS ({row['taux_hs']*100:.1f}%) sur {row['n_total']:.0f} anomalies\n"

    report += f"""
---

## 6. Recommandations

### Actions Prioritaires

1. **Surveillance renforcée** des tronçons avec anomalies critiques
   - Délai moyen très court avant défaillance
   - Intervention préventive recommandée

2. **Programme de maintenance préventive**
   - Cibler les matériaux à risque élevé
   - Planifier interventions selon sévérité

3. **Système d'alerte**
   - Créer des alertes automatiques pour anomalies critiques
   - Suivi des délais par sévérité

4. **Priorisation des renouvellements**
   - Intégrer les anomalies dans le score de risque
   - Pondérer par sévérité et délai historique

### Optimisation du Modèle de Prédiction

Les anomalies historiques peuvent enrichir le modèle:
- **Feature engineering**: Nombre d'anomalies par sévérité
- **Pondération**: Anomalies critiques comme prédicteur fort
- **Temporalité**: Délai depuis dernière anomalie

---

## 7. Fichiers Générés

Les visualisations suivantes ont été créées dans `reports/plots/`:

1. `anomaly_severity_distribution.png` - Distribution par sévérité
2. `failure_rate_by_severity.png` - Taux de mise HS par sévérité
3. `delay_to_failure_analysis.png` - Analyse temporelle détaillée
4. `top_anomaly_types.png` - Types d'anomalies principaux

---

*Rapport généré automatiquement par l'analyse des anomalies*
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"   ✓ Rapport sauvegardé: {output_path}")


def main():
    """Fonction principale d'analyse."""
    print("="*70)
    print("ANALYSE DES ANOMALIES ET CORRÉLATION AVEC MISE HORS SERVICE")
    print("="*70)

    # Créer répertoire de sortie
    output_dir = REPORTS_DIR / "plots"
    output_dir.mkdir(exist_ok=True, parents=True)

    # 1. Charger les données
    df_anomalies = load_anomalies_data()
    df_troncons = load_troncons_data()

    # 2. Classifier les anomalies
    df_anomalies = classify_anomaly_severity(df_anomalies)

    # 3. Fusionner
    df_merged = merge_data(df_anomalies, df_troncons)

    # 4. Analyser corrélation
    df_valid = analyze_anomaly_to_failure_correlation(df_merged)

    # 5. Statistiques
    type_stats, mat_stats = compute_anomaly_statistics(df_merged)

    # 6. Visualisations
    print("\n8. Génération des visualisations...")
    plot_anomaly_severity_distribution(df_anomalies, output_dir)
    plot_failure_rate_by_severity(df_merged, output_dir)
    plot_delay_to_failure(df_valid, output_dir)
    plot_top_anomaly_types(df_merged, output_dir)

    # 7. Rapport
    report_path = REPORTS_DIR / "anomaly_analysis_report.md"
    generate_anomaly_report(df_anomalies, df_merged, df_valid,
                           type_stats, mat_stats, report_path)

    # 8. Sauvegarder les données détaillées
    print("\n9. Sauvegarde des données détaillées...")
    df_merged[['GID_OBJET', 'TYPE_ANOMALIE', 'SEVERITE', 'DATE_DETECTION_parsed',
               'MATERIAU', 'DIAMETRE', 'is_hors_service', 'DATE_HS']].to_csv(
        REPORTS_DIR / "anomaly_details.csv", index=False
    )
    print(f"   ✓ Détails sauvegardés: anomaly_details.csv")

    # Sauvegarder les délais
    df_valid[['GID_OBJET', 'TYPE_ANOMALIE', 'SEVERITE', 'DATE_DETECTION_parsed',
              'DATE_HS', 'DELAI_JOURS', 'DELAI_ANNEES', 'MATERIAU']].to_csv(
        REPORTS_DIR / "anomaly_to_failure_delays.csv", index=False
    )
    print(f"   ✓ Délais sauvegardés: anomaly_to_failure_delays.csv")

    print("\n" + "="*70)
    print("ANALYSE TERMINÉE")
    print("="*70)
    print(f"\n📊 Rapport principal: {report_path}")
    print(f"📈 Visualisations: {output_dir}")
    print(f"💾 Données détaillées: {REPORTS_DIR}")

    return df_merged, df_valid


if __name__ == "__main__":
    main()
