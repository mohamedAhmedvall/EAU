"""
Step 1: Analyse descriptive de survie (Kaplan-Meier)
- Courbes de survie globales et par groupes
- Statistiques de durée de vie par groupe
- Identification des patterns et biais
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from config import (
    ARTIFACTS_DIR, REPORTS_DIR,
    COL_PIPE_ID, COL_PIPE_ID_ANOMALIE, COL_MATERIAU, COL_DIAMETRE, COL_LONGUEUR
)


def load_data():
    """Charge les données préparées."""
    df_trafic = pd.read_pickle(ARTIFACTS_DIR / "df_trafic_parsed.pkl")
    df_anomalies = pd.read_pickle(ARTIFACTS_DIR / "df_anomalies_parsed.pkl")
    return df_trafic, df_anomalies


def prepare_survival_data(df_trafic: pd.DataFrame, max_obs_date: pd.Timestamp) -> pd.DataFrame:
    """
    Prépare les données pour l'analyse de survie.
    Calcule la durée de vie observée et l'indicateur d'événement.
    """
    df = df_trafic.copy()

    # Exclure les tronçons avec dates incohérentes (DDP > DHS)
    valid_dates = df['DDP_parsed'].notna()
    has_dhs = df['DHS_parsed'].notna()
    coherent = ~has_dhs | (df['DDP_parsed'] <= df['DHS_parsed'])

    df = df[valid_dates & coherent].copy()

    # Calcul de la durée de vie observée (en années)
    # Pour les tronçons avec DHS <= max_obs_date : durée = DHS - DDP
    # Pour les autres (censurés) : durée = max_obs_date - DDP

    df['event'] = 0  # 0 = censuré, 1 = événement observé
    df['duration_days'] = np.nan

    # Tronçons avec DHS observé
    mask_event = has_dhs & (df['DHS_parsed'] <= max_obs_date)
    df.loc[mask_event, 'event'] = 1
    df.loc[mask_event, 'duration_days'] = (df.loc[mask_event, 'DHS_parsed'] - df.loc[mask_event, 'DDP_parsed']).dt.days

    # Tronçons censurés (pas de DHS ou DHS > max_obs_date)
    mask_censored = ~mask_event
    df.loc[mask_censored, 'duration_days'] = (max_obs_date - df.loc[mask_censored, 'DDP_parsed']).dt.days

    # Conversion en années
    df['duration_years'] = df['duration_days'] / 365.25

    # Exclure les durées négatives ou nulles
    df = df[df['duration_days'] > 0].copy()

    # Ajout de variables de groupement
    # Décennie d'installation
    df['decade_install'] = (df['DDP_parsed'].dt.year // 10) * 10
    df['decade_install_str'] = df['decade_install'].astype(str) + 's'

    # Bins de diamètre
    df['diametre_bin'] = pd.cut(
        df[COL_DIAMETRE],
        bins=[0, 80, 100, 150, 200, 300, 1000, 10000],
        labels=['≤80', '80-100', '100-150', '150-200', '200-300', '300-1000', '>1000']
    )

    # Bins de longueur
    df['longueur_bin'] = pd.cut(
        df[COL_LONGUEUR],
        bins=[0, 20, 50, 100, 200, 500, 10000000],
        labels=['≤20m', '20-50m', '50-100m', '100-200m', '200-500m', '>500m']
    )

    return df


def add_anomaly_count(df_survival: pd.DataFrame, df_anomalies: pd.DataFrame,
                      max_obs_date: pd.Timestamp) -> pd.DataFrame:
    """
    Ajoute le nombre d'anomalies pré-observation pour chaque tronçon.
    """
    # Filtrer les anomalies avant max_obs_date
    ano = df_anomalies[df_anomalies['DATE_DETECTION'] <= max_obs_date].copy()

    # Compter les anomalies par tronçon
    ano_count = ano.groupby(COL_PIPE_ID_ANOMALIE).size().reset_index(name='n_anomalies')

    # Joindre
    df = df_survival.merge(
        ano_count,
        left_on=COL_PIPE_ID,
        right_on=COL_PIPE_ID_ANOMALIE,
        how='left'
    )
    df['n_anomalies'] = df['n_anomalies'].fillna(0).astype(int)

    # Bins d'anomalies
    df['anomaly_group'] = pd.cut(
        df['n_anomalies'],
        bins=[-1, 0, 1, 3, 1000],
        labels=['0', '1', '2-3', '4+']
    )

    return df


def fit_and_plot_km_global(df: pd.DataFrame, output_dir: Path):
    """
    Courbe Kaplan-Meier globale.
    """
    kmf = KaplanMeierFitter()
    kmf.fit(df['duration_years'], event_observed=df['event'], label='Global')

    fig, ax = plt.subplots(figsize=(10, 6))
    kmf.plot_survival_function(ax=ax)
    ax.set_xlabel('Années depuis installation')
    ax.set_ylabel('Probabilité de survie')
    ax.set_title('Courbe de Survie Kaplan-Meier - Tous tronçons')
    ax.set_xlim(0, 120)
    ax.grid(True, alpha=0.3)

    # Ajouter des stats
    median_survival = kmf.median_survival_time_
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=median_survival, color='gray', linestyle='--', alpha=0.5)
    ax.text(median_survival + 2, 0.52, f'Médiane: {median_survival:.1f} ans', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'km_global.png', dpi=150)
    plt.close()

    return kmf


def fit_and_plot_km_by_group(df: pd.DataFrame, group_col: str, title: str,
                             output_dir: Path, max_groups: int = 10):
    """
    Courbes Kaplan-Meier stratifiées par groupe.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    groups = df[group_col].dropna().unique()
    # Trier les groupes si possible
    try:
        groups = sorted(groups)
    except:
        pass

    # Limiter le nombre de groupes pour la lisibilité
    if len(groups) > max_groups:
        # Prendre les groupes les plus fréquents
        top_groups = df[group_col].value_counts().head(max_groups).index.tolist()
        groups = [g for g in groups if g in top_groups]

    stats = []
    for group in groups:
        mask = df[group_col] == group
        if mask.sum() < 50:  # Minimum 50 observations
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(df.loc[mask, 'duration_years'],
                event_observed=df.loc[mask, 'event'],
                label=str(group))
        kmf.plot_survival_function(ax=ax)

        # Statistiques
        median = kmf.median_survival_time_
        n_total = mask.sum()
        n_events = df.loc[mask, 'event'].sum()
        stats.append({
            'group': group,
            'n_total': n_total,
            'n_events': n_events,
            'event_rate': 100 * n_events / n_total,
            'median_survival': median,
            'q25_survival': kmf.percentile(0.75),  # 75% survie = 25% défailli
            'q50_survival': median,
            'q75_survival': kmf.percentile(0.25)   # 25% survie = 75% défailli
        })

    ax.set_xlabel('Années depuis installation')
    ax.set_ylabel('Probabilité de survie')
    ax.set_title(title)
    ax.set_xlim(0, 120)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left')

    plt.tight_layout()
    filename = f'km_by_{group_col.lower().replace(" ", "_")}.png'
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()

    return pd.DataFrame(stats)


def compute_survival_statistics(df: pd.DataFrame) -> dict:
    """
    Calcule des statistiques de survie globales et par groupe.
    """
    stats = {}

    # Global
    kmf = KaplanMeierFitter()
    kmf.fit(df['duration_years'], event_observed=df['event'])
    stats['global'] = {
        'n_total': len(df),
        'n_events': df['event'].sum(),
        'event_rate': 100 * df['event'].sum() / len(df),
        'median_survival': kmf.median_survival_time_,
        'mean_duration_observed': df['duration_years'].mean()
    }

    return stats


def generate_survival_report(df: pd.DataFrame, stats_global: dict,
                             stats_by_mat: pd.DataFrame,
                             stats_by_decade: pd.DataFrame,
                             stats_by_anomaly: pd.DataFrame,
                             stats_by_diametre: pd.DataFrame) -> str:
    """
    Génère le rapport d'analyse de survie en markdown.
    """
    report = []
    report.append("# Analyse Descriptive de Survie (Kaplan-Meier)\n")
    report.append(f"*Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # Section 1: Résumé global
    report.append("\n## 1. Statistiques Globales\n")
    g = stats_global['global']
    report.append(f"- **Nombre total de tronçons analysés**: {g['n_total']:,}")
    report.append(f"- **Nombre d'événements (DHS observé)**: {g['n_events']:,}")
    report.append(f"- **Taux d'événement global**: {g['event_rate']:.1f}%")
    report.append(f"- **Durée de vie médiane (Kaplan-Meier)**: {g['median_survival']:.1f} ans")
    report.append(f"- **Durée d'observation moyenne**: {g['mean_duration_observed']:.1f} ans")

    report.append("\n### Interprétation")
    report.append("La durée de vie médiane représente l'âge auquel 50% des tronçons sont défaillants/abandonnés.")
    report.append("Un taux d'événement de {:.1f}% signifie que la majorité des tronçons sont encore en service (censurés à droite).".format(100 - g['event_rate']))

    # Section 2: Par matériau
    report.append("\n## 2. Survie par Matériau\n")
    report.append("![Courbes KM par matériau](km_by_mat.png)\n")
    report.append("| Matériau | N total | N événements | Taux événement | Durée médiane (ans) |")
    report.append("|----------|---------|--------------|----------------|---------------------|")
    for _, row in stats_by_mat.sort_values('median_survival').iterrows():
        median_str = f"{row['median_survival']:.1f}" if pd.notna(row['median_survival']) else ">obs"
        report.append(f"| {row['group']} | {row['n_total']:,} | {row['n_events']:,} | {row['event_rate']:.1f}% | {median_str} |")

    report.append("\n### Observations par matériau")
    # Identifier les matériaux à risque
    if len(stats_by_mat) > 0:
        mat_sorted = stats_by_mat.dropna(subset=['median_survival']).sort_values('median_survival')
        if len(mat_sorted) > 0:
            worst_mat = mat_sorted.iloc[0]
            best_mat = mat_sorted.iloc[-1]
            report.append(f"- **Matériau le plus fragile**: {worst_mat['group']} (médiane: {worst_mat['median_survival']:.1f} ans)")
            report.append(f"- **Matériau le plus durable**: {best_mat['group']} (médiane: {best_mat['median_survival']:.1f} ans)")

    # Section 3: Par décennie d'installation
    report.append("\n## 3. Survie par Décennie d'Installation\n")
    report.append("![Courbes KM par décennie](km_by_decade_install_str.png)\n")
    report.append("| Décennie | N total | N événements | Taux événement | Durée médiane (ans) |")
    report.append("|----------|---------|--------------|----------------|---------------------|")
    for _, row in stats_by_decade.sort_values('group').iterrows():
        median_str = f"{row['median_survival']:.1f}" if pd.notna(row['median_survival']) else ">obs"
        report.append(f"| {row['group']} | {row['n_total']:,} | {row['n_events']:,} | {row['event_rate']:.1f}% | {median_str} |")

    report.append("\n### Observations temporelles")
    report.append("- Les tronçons anciens ont un biais de survivant: ceux qui sont encore en service ont \"survécu\" plus longtemps.")
    report.append("- Les tronçons récents sont censurés (pas assez de recul pour observer leur défaillance).")

    # Section 4: Par historique d'anomalies
    report.append("\n## 4. Survie par Historique d'Anomalies\n")
    report.append("![Courbes KM par groupe d'anomalies](km_by_anomaly_group.png)\n")
    report.append("| Groupe anomalies | N total | N événements | Taux événement | Durée médiane (ans) |")
    report.append("|------------------|---------|--------------|----------------|---------------------|")
    for _, row in stats_by_anomaly.iterrows():
        median_str = f"{row['median_survival']:.1f}" if pd.notna(row['median_survival']) else ">obs"
        report.append(f"| {row['group']} | {row['n_total']:,} | {row['n_events']:,} | {row['event_rate']:.1f}% | {median_str} |")

    report.append("\n### Observations sur les anomalies")
    report.append("- Les tronçons avec plus d'anomalies historiques ont généralement une durée de vie plus courte.")
    report.append("- Cela confirme l'utilité de l'historique des fuites comme prédicteur de risque.")

    # Section 5: Par diamètre
    report.append("\n## 5. Survie par Diamètre\n")
    report.append("![Courbes KM par diamètre](km_by_diametre_bin.png)\n")
    report.append("| Diamètre | N total | N événements | Taux événement | Durée médiane (ans) |")
    report.append("|----------|---------|--------------|----------------|---------------------|")
    for _, row in stats_by_diametre.iterrows():
        median_str = f"{row['median_survival']:.1f}" if pd.notna(row['median_survival']) else ">obs"
        report.append(f"| {row['group']} | {row['n_total']:,} | {row['n_events']:,} | {row['event_rate']:.1f}% | {median_str} |")

    # Section 6: Considérations méthodologiques
    report.append("\n## 6. Considérations Méthodologiques\n")
    report.append("### Biais potentiels identifiés")
    report.append("1. **Biais de survivant**: Les tronçons très anciens encore en service sont par définition des \"survivants\"")
    report.append("2. **Censure à droite importante**: {:.1f}% des tronçons n'ont pas encore d'événement observé".format(100 - g['event_rate']))
    report.append("3. **Troncature à gauche**: Les tronçons posés avant le début de l'observation des anomalies peuvent avoir un historique incomplet")
    report.append("")
    report.append("### Implications pour la modélisation")
    report.append("- Utiliser une approche freeze+horizon pour éviter la fuite de données temporelles")
    report.append("- Les features basées sur l'historique des anomalies doivent être calculées strictement avant FREEZE_DATE")
    report.append("- Les durées de vie médianes par matériau seront utilisées comme features de référence")

    return "\n".join(report)


def main():
    """
    Exécution de l'analyse de survie descriptive.
    """
    print("=" * 60)
    print("STEP 1: ANALYSE DESCRIPTIVE DE SURVIE")
    print("=" * 60)

    # Chargement des données
    print("\nChargement des données...")
    df_trafic, df_anomalies = load_data()

    # Date max d'observation
    import json
    with open(ARTIFACTS_DIR / "metadata.json", 'r') as f:
        metadata = json.load(f)
    max_obs_date = pd.Timestamp(metadata['max_observation_date'])
    print(f"Date max d'observation: {max_obs_date}")

    # Préparation des données de survie
    print("\nPréparation des données de survie...")
    df_surv = prepare_survival_data(df_trafic, max_obs_date)
    print(f"  -> {len(df_surv)} tronçons analysables")
    print(f"  -> {df_surv['event'].sum()} événements observés ({100*df_surv['event'].mean():.1f}%)")

    # Ajout du comptage d'anomalies
    print("\nAjout de l'historique d'anomalies...")
    df_surv = add_anomaly_count(df_surv, df_anomalies, max_obs_date)

    # Création du dossier de sortie pour les plots
    plots_dir = REPORTS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Courbe KM globale
    print("\nCourbe Kaplan-Meier globale...")
    kmf_global = fit_and_plot_km_global(df_surv, plots_dir)

    # Courbes par groupe
    print("Courbes par matériau...")
    stats_mat = fit_and_plot_km_by_group(df_surv, COL_MATERIAU, 'Survie par Matériau', plots_dir)

    print("Courbes par décennie d'installation...")
    stats_decade = fit_and_plot_km_by_group(df_surv, 'decade_install_str', 'Survie par Décennie d\'Installation', plots_dir)

    print("Courbes par groupe d'anomalies...")
    stats_anomaly = fit_and_plot_km_by_group(df_surv, 'anomaly_group', 'Survie par Historique d\'Anomalies', plots_dir)

    print("Courbes par diamètre...")
    stats_diametre = fit_and_plot_km_by_group(df_surv, 'diametre_bin', 'Survie par Diamètre', plots_dir)

    print("Courbes par longueur...")
    stats_longueur = fit_and_plot_km_by_group(df_surv, 'longueur_bin', 'Survie par Longueur', plots_dir)

    # Statistiques globales
    print("\nCalcul des statistiques...")
    stats_global = compute_survival_statistics(df_surv)

    # Génération du rapport
    print("\nGénération du rapport...")
    report = generate_survival_report(
        df_surv, stats_global,
        stats_mat, stats_decade, stats_anomaly, stats_diametre
    )

    report_path = REPORTS_DIR / "survival_eda.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Rapport sauvegardé: {report_path}")

    # Sauvegarde des statistiques de survie par matériau (pour le feature engineering)
    survival_stats = {
        'by_material': stats_mat.to_dict('records'),
        'by_decade': stats_decade.to_dict('records'),
        'by_anomaly_group': stats_anomaly.to_dict('records'),
        'global': stats_global
    }
    import json
    with open(ARTIFACTS_DIR / "survival_stats.json", 'w') as f:
        json.dump(survival_stats, f, indent=2, default=str)

    # Sauvegarde du DataFrame préparé
    df_surv.to_pickle(ARTIFACTS_DIR / "df_survival_prepared.pkl")

    print("\n" + "=" * 60)
    print("ANALYSE DE SURVIE TERMINÉE")
    print("=" * 60)

    return df_surv, stats_global


if __name__ == "__main__":
    main()
