"""
Step 0: Audit des données et alignement
- Chargement des fichiers
- Parsing robuste des dates
- Analyse de la qualité des données
- Génération du rapport d'audit
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_DIR, REPORTS_DIR, ARTIFACTS_DIR,
    TRAFIC_FILE, ANOMALIES_FILE,
    COL_PIPE_ID, COL_PIPE_ID_ANOMALIE, COL_DATE_INSTALL, COL_DATE_HS,
    COL_MATERIAU, COL_DIAMETRE, COL_LONGUEUR, COL_STATUT,
    COL_DATE_ANOMALIE, COL_TYPE_ANOMALIE
)


def load_and_parse_trafic(filepath: Path) -> pd.DataFrame:
    """
    Charge et parse la table patrimoine (v1_trafic_prepared.csv).
    """
    print(f"Chargement de {filepath.name}...")
    df = pd.read_csv(filepath)

    # Parsing des dates
    df['DDP_parsed'] = pd.to_datetime(df[COL_DATE_INSTALL], errors='coerce', utc=True)
    df['DDP_parsed'] = df['DDP_parsed'].dt.tz_localize(None)  # Enlever timezone

    df['DHS_parsed'] = pd.to_datetime(df[COL_DATE_HS], errors='coerce', utc=True)
    df['DHS_parsed'] = df['DHS_parsed'].dt.tz_localize(None)

    print(f"  -> {len(df)} lignes chargées")
    return df


def load_and_parse_anomalies(filepath: Path) -> pd.DataFrame:
    """
    Charge et parse la table des anomalies (historiqueanomalie.csv).
    """
    print(f"Chargement de {filepath.name}...")
    df = pd.read_csv(filepath)

    # Parsing des dates
    df['DATE_DETECTION'] = pd.to_datetime(df[COL_DATE_ANOMALIE], errors='coerce', utc=True)
    df['DATE_DETECTION'] = df['DATE_DETECTION'].dt.tz_localize(None)

    print(f"  -> {len(df)} lignes chargées")
    return df


def audit_trafic(df: pd.DataFrame) -> dict:
    """
    Audit complet de la table patrimoine.
    """
    audit = {}

    # Statistiques générales
    audit['n_rows'] = len(df)
    audit['n_cols'] = len(df.columns)
    audit['columns'] = list(df.columns)

    # Doublons sur GID
    audit['n_duplicates_gid'] = df[COL_PIPE_ID].duplicated().sum()
    audit['n_unique_gid'] = df[COL_PIPE_ID].nunique()

    # Valeurs manquantes
    missing = df.isnull().sum()
    audit['missing'] = missing[missing > 0].to_dict()

    # Analyse des dates
    audit['dates'] = {}

    # DDP (Date de pose)
    ddp_valid = df['DDP_parsed'].notna()
    audit['dates']['DDP_valid'] = ddp_valid.sum()
    audit['dates']['DDP_missing'] = (~ddp_valid).sum()
    if ddp_valid.any():
        audit['dates']['DDP_min'] = df.loc[ddp_valid, 'DDP_parsed'].min()
        audit['dates']['DDP_max'] = df.loc[ddp_valid, 'DDP_parsed'].max()

    # DHS (Date hors service)
    dhs_valid = df['DHS_parsed'].notna()
    audit['dates']['DHS_valid'] = dhs_valid.sum()
    audit['dates']['DHS_missing'] = (~dhs_valid).sum()
    if dhs_valid.any():
        audit['dates']['DHS_min'] = df.loc[dhs_valid, 'DHS_parsed'].min()
        audit['dates']['DHS_max'] = df.loc[dhs_valid, 'DHS_parsed'].max()

    # Incohérences dates: DDP > DHS
    both_valid = ddp_valid & dhs_valid
    if both_valid.any():
        incoherent = df.loc[both_valid, 'DDP_parsed'] > df.loc[both_valid, 'DHS_parsed']
        audit['dates']['n_ddp_after_dhs'] = incoherent.sum()
    else:
        audit['dates']['n_ddp_after_dhs'] = 0

    # Dates futures (> aujourd'hui)
    today = pd.Timestamp.now().normalize()
    audit['dates']['today'] = today

    if ddp_valid.any():
        future_ddp = df.loc[ddp_valid, 'DDP_parsed'] > today
        audit['dates']['n_future_ddp'] = future_ddp.sum()

    if dhs_valid.any():
        future_dhs = df.loc[dhs_valid, 'DHS_parsed'] > today
        audit['dates']['n_future_dhs'] = future_dhs.sum()

    # Analyse des matériaux
    audit['materiau'] = df[COL_MATERIAU].value_counts().to_dict()
    audit['n_materiau_missing'] = df[COL_MATERIAU].isna().sum()

    # Analyse des statuts
    audit['statut'] = df[COL_STATUT].value_counts().to_dict()

    # Statistiques numériques
    audit['diametre'] = {
        'min': df[COL_DIAMETRE].min(),
        'max': df[COL_DIAMETRE].max(),
        'mean': df[COL_DIAMETRE].mean(),
        'median': df[COL_DIAMETRE].median(),
        'missing': df[COL_DIAMETRE].isna().sum()
    }

    audit['longueur'] = {
        'min': df[COL_LONGUEUR].min(),
        'max': df[COL_LONGUEUR].max(),
        'mean': df[COL_LONGUEUR].mean(),
        'sum_km': df[COL_LONGUEUR].sum() / 1000,
        'missing': df[COL_LONGUEUR].isna().sum()
    }

    return audit


def audit_anomalies(df: pd.DataFrame) -> dict:
    """
    Audit complet de la table des anomalies.
    """
    audit = {}

    # Statistiques générales
    audit['n_rows'] = len(df)
    audit['n_cols'] = len(df.columns)
    audit['columns'] = list(df.columns)

    # Analyse des identifiants
    audit['n_unique_gid'] = df[COL_PIPE_ID_ANOMALIE].nunique()
    audit['n_duplicates'] = df.duplicated().sum()

    # Analyse des dates
    date_valid = df['DATE_DETECTION'].notna()
    audit['dates'] = {
        'valid': date_valid.sum(),
        'missing': (~date_valid).sum()
    }

    if date_valid.any():
        audit['dates']['min'] = df.loc[date_valid, 'DATE_DETECTION'].min()
        audit['dates']['max'] = df.loc[date_valid, 'DATE_DETECTION'].max()

        # Dates futures
        today = pd.Timestamp.now().normalize()
        future = df.loc[date_valid, 'DATE_DETECTION'] > today
        audit['dates']['n_future'] = future.sum()

    # Types d'anomalies
    audit['types'] = df[COL_TYPE_ANOMALIE].value_counts().to_dict()

    # Distribution par année
    df_temp = df.copy()
    df_temp['annee'] = df_temp['DATE_DETECTION'].dt.year
    audit['par_annee'] = df_temp['annee'].value_counts().sort_index().to_dict()

    # Anomalies par tronçon
    anomalies_par_troncon = df.groupby(COL_PIPE_ID_ANOMALIE).size()
    audit['anomalies_par_troncon'] = {
        'min': anomalies_par_troncon.min(),
        'max': anomalies_par_troncon.max(),
        'mean': anomalies_par_troncon.mean(),
        'median': anomalies_par_troncon.median(),
        'q75': anomalies_par_troncon.quantile(0.75),
        'q90': anomalies_par_troncon.quantile(0.90),
        'q99': anomalies_par_troncon.quantile(0.99)
    }

    return audit


def check_join_keys(df_trafic: pd.DataFrame, df_anomalies: pd.DataFrame) -> dict:
    """
    Vérifie l'alignement des clés de jointure entre les tables.
    """
    gid_trafic = set(df_trafic[COL_PIPE_ID].unique())
    gid_anomalies = set(df_anomalies[COL_PIPE_ID_ANOMALIE].unique())

    join_check = {
        'n_gid_trafic': len(gid_trafic),
        'n_gid_anomalies': len(gid_anomalies),
        'n_common': len(gid_trafic & gid_anomalies),
        'n_only_trafic': len(gid_trafic - gid_anomalies),
        'n_only_anomalies': len(gid_anomalies - gid_trafic),
        'pct_trafic_with_anomalies': 100 * len(gid_trafic & gid_anomalies) / len(gid_trafic) if len(gid_trafic) > 0 else 0,
        'pct_anomalies_matched': 100 * len(gid_trafic & gid_anomalies) / len(gid_anomalies) if len(gid_anomalies) > 0 else 0
    }

    # Echantillon des GID dans anomalies mais pas dans trafic
    orphan_anomalies = gid_anomalies - gid_trafic
    if orphan_anomalies:
        join_check['sample_orphan_anomalies'] = list(orphan_anomalies)[:10]

    return join_check


def compute_max_observation_date(df_trafic: pd.DataFrame, df_anomalies: pd.DataFrame) -> pd.Timestamp:
    """
    Calcule la date maximale d'observation dans les données (pour définir FREEZE_DATE).
    Exclut les dates futures implausibles.
    """
    today = pd.Timestamp.now().normalize()

    # Date max dans DHS (abandons observés)
    dhs_valid = df_trafic['DHS_parsed'].notna() & (df_trafic['DHS_parsed'] <= today)
    max_dhs = df_trafic.loc[dhs_valid, 'DHS_parsed'].max() if dhs_valid.any() else pd.NaT

    # Date max dans anomalies
    date_valid = df_anomalies['DATE_DETECTION'].notna() & (df_anomalies['DATE_DETECTION'] <= today)
    max_anomalie = df_anomalies.loc[date_valid, 'DATE_DETECTION'].max() if date_valid.any() else pd.NaT

    # Prendre le max des deux
    dates = [d for d in [max_dhs, max_anomalie] if pd.notna(d)]
    max_obs = max(dates) if dates else today

    print(f"\nDates maximales d'observation:")
    print(f"  - Max DHS: {max_dhs}")
    print(f"  - Max anomalie: {max_anomalie}")
    print(f"  - Max observation retenue: {max_obs}")

    return max_obs


def generate_audit_report(audit_trafic: dict, audit_anomalies: dict,
                          join_check: dict, max_obs_date: pd.Timestamp) -> str:
    """
    Génère le rapport d'audit en markdown.
    """
    report = []
    report.append("# Rapport d'Audit des Données\n")
    report.append(f"*Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # Section 1: Résumé
    report.append("\n## 1. Résumé Exécutif\n")
    report.append(f"- **Table Patrimoine (v1_trafic_prepared.csv)**: {audit_trafic['n_rows']:,} tronçons")
    report.append(f"- **Table Anomalies (historiqueanomalie.csv)**: {audit_anomalies['n_rows']:,} anomalies")
    report.append(f"- **Tronçons avec au moins une anomalie**: {join_check['n_common']:,} ({join_check['pct_trafic_with_anomalies']:.1f}%)")
    report.append(f"- **Date maximale d'observation**: {max_obs_date.strftime('%Y-%m-%d')}")

    # Taux d'événement (DHS renseigné = événement de défaillance)
    taux_defaillance = 100 * audit_trafic['dates']['DHS_valid'] / audit_trafic['n_rows']
    report.append(f"- **Taux de tronçons avec DHS (défaillants/abandonnés)**: {audit_trafic['dates']['DHS_valid']:,} ({taux_defaillance:.1f}%)")

    # Section 2: Table Patrimoine
    report.append("\n## 2. Table Patrimoine (v1_trafic_prepared.csv)\n")
    report.append(f"### 2.1 Structure")
    report.append(f"- Nombre de lignes: {audit_trafic['n_rows']:,}")
    report.append(f"- Nombre de colonnes: {audit_trafic['n_cols']}")
    report.append(f"- Colonnes: `{', '.join(audit_trafic['columns'])}`")
    report.append(f"- Identifiants uniques (GID): {audit_trafic['n_unique_gid']:,}")
    report.append(f"- Doublons sur GID: {audit_trafic['n_duplicates_gid']}")

    report.append(f"\n### 2.2 Dates")
    report.append(f"**Date de Pose (DDP):**")
    report.append(f"- Valides: {audit_trafic['dates']['DDP_valid']:,}")
    report.append(f"- Manquantes: {audit_trafic['dates']['DDP_missing']}")
    if 'DDP_min' in audit_trafic['dates']:
        report.append(f"- Plage: {audit_trafic['dates']['DDP_min'].strftime('%Y-%m-%d')} → {audit_trafic['dates']['DDP_max'].strftime('%Y-%m-%d')}")
    if 'n_future_ddp' in audit_trafic['dates']:
        report.append(f"- Dates futures (implausibles): {audit_trafic['dates']['n_future_ddp']}")

    report.append(f"\n**Date Hors Service (DHS):**")
    report.append(f"- Valides: {audit_trafic['dates']['DHS_valid']:,}")
    report.append(f"- Manquantes (= en service): {audit_trafic['dates']['DHS_missing']:,}")
    if 'DHS_min' in audit_trafic['dates']:
        report.append(f"- Plage: {audit_trafic['dates']['DHS_min'].strftime('%Y-%m-%d')} → {audit_trafic['dates']['DHS_max'].strftime('%Y-%m-%d')}")
    if 'n_future_dhs' in audit_trafic['dates']:
        report.append(f"- Dates futures (implausibles): {audit_trafic['dates']['n_future_dhs']}")

    report.append(f"\n**Incohérences:**")
    report.append(f"- DDP après DHS: {audit_trafic['dates']['n_ddp_after_dhs']}")

    report.append(f"\n### 2.3 Matériaux")
    report.append("| Matériau | Nombre | % |")
    report.append("|----------|--------|---|")
    total = audit_trafic['n_rows']
    for mat, count in sorted(audit_trafic['materiau'].items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        report.append(f"| {mat} | {count:,} | {pct:.1f}% |")

    report.append(f"\n### 2.4 Statuts")
    report.append("| Statut | Nombre | % |")
    report.append("|--------|--------|---|")
    for statut, count in sorted(audit_trafic['statut'].items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        report.append(f"| {statut} | {count:,} | {pct:.1f}% |")

    report.append(f"\n### 2.5 Caractéristiques Physiques")
    report.append(f"**Diamètre (mm):**")
    report.append(f"- Min: {audit_trafic['diametre']['min']}")
    report.append(f"- Max: {audit_trafic['diametre']['max']}")
    report.append(f"- Moyenne: {audit_trafic['diametre']['mean']:.1f}")
    report.append(f"- Médiane: {audit_trafic['diametre']['median']}")

    report.append(f"\n**Longueur (m):**")
    report.append(f"- Min: {audit_trafic['longueur']['min']:.1f}")
    report.append(f"- Max: {audit_trafic['longueur']['max']:.1f}")
    report.append(f"- Moyenne: {audit_trafic['longueur']['mean']:.1f}")
    report.append(f"- Total réseau: {audit_trafic['longueur']['sum_km']:.1f} km")

    # Section 3: Table Anomalies
    report.append("\n## 3. Table Anomalies (historiqueanomalie.csv)\n")
    report.append(f"### 3.1 Structure")
    report.append(f"- Nombre de lignes (anomalies): {audit_anomalies['n_rows']:,}")
    report.append(f"- Tronçons uniques concernés: {audit_anomalies['n_unique_gid']:,}")
    report.append(f"- Lignes dupliquées exactes: {audit_anomalies['n_duplicates']}")

    report.append(f"\n### 3.2 Dates de détection")
    report.append(f"- Valides: {audit_anomalies['dates']['valid']:,}")
    report.append(f"- Manquantes: {audit_anomalies['dates']['missing']}")
    if 'min' in audit_anomalies['dates']:
        report.append(f"- Plage: {audit_anomalies['dates']['min'].strftime('%Y-%m-%d')} → {audit_anomalies['dates']['max'].strftime('%Y-%m-%d')}")
    if 'n_future' in audit_anomalies['dates']:
        report.append(f"- Dates futures (implausibles): {audit_anomalies['dates']['n_future']}")

    report.append(f"\n### 3.3 Types d'anomalies")
    report.append("| Type | Nombre | % |")
    report.append("|------|--------|---|")
    total_ano = audit_anomalies['n_rows']
    for typ, count in sorted(audit_anomalies['types'].items(), key=lambda x: -x[1]):
        pct = 100 * count / total_ano
        report.append(f"| {typ} | {count:,} | {pct:.1f}% |")

    report.append(f"\n### 3.4 Distribution des anomalies par tronçon")
    stats = audit_anomalies['anomalies_par_troncon']
    report.append(f"- Min: {stats['min']}")
    report.append(f"- Max: {stats['max']}")
    report.append(f"- Moyenne: {stats['mean']:.2f}")
    report.append(f"- Médiane: {stats['median']:.0f}")
    report.append(f"- 75ème percentile: {stats['q75']:.0f}")
    report.append(f"- 90ème percentile: {stats['q90']:.0f}")
    report.append(f"- 99ème percentile: {stats['q99']:.0f}")

    # Section 4: Alignement des clés
    report.append("\n## 4. Alignement des Tables\n")
    report.append(f"- GID dans patrimoine: {join_check['n_gid_trafic']:,}")
    report.append(f"- GID dans anomalies: {join_check['n_gid_anomalies']:,}")
    report.append(f"- GID communs: {join_check['n_common']:,}")
    report.append(f"- GID uniquement dans patrimoine: {join_check['n_only_trafic']:,}")
    report.append(f"- GID uniquement dans anomalies (orphelins): {join_check['n_only_anomalies']:,}")
    report.append(f"- % de tronçons avec anomalies: {join_check['pct_trafic_with_anomalies']:.1f}%")
    report.append(f"- % d'anomalies matchées: {join_check['pct_anomalies_matched']:.1f}%")

    if 'sample_orphan_anomalies' in join_check:
        report.append(f"\n**Échantillon de GID orphelins (anomalies sans tronçon):**")
        report.append(f"`{join_check['sample_orphan_anomalies']}`")

    # Section 5: Décisions de nettoyage
    report.append("\n## 5. Décisions de Nettoyage\n")
    report.append("1. **Dates futures**: Seront exclues pour le calcul de FREEZE_DATE")
    report.append("2. **DDP > DHS**: Ces tronçons seront signalés et potentiellement exclus")
    report.append("3. **Anomalies orphelines**: Seront ignorées (pas de jointure possible)")
    report.append("4. **DHS manquant**: Interprété comme tronçon toujours en service (censuré)")

    # Section 6: Recommandations
    report.append("\n## 6. Définition des Horizons\n")
    report.append(f"**Date maximale d'observation retenue**: {max_obs_date.strftime('%Y-%m-%d')}")
    report.append("\n**FREEZE_DATE par horizon:**")
    for h in [1, 3, 5]:
        freeze = max_obs_date - pd.DateOffset(years=h)
        report.append(f"- Horizon {h} an(s): FREEZE_DATE = {freeze.strftime('%Y-%m-%d')}")

    return "\n".join(report)


def main():
    """
    Exécution de l'audit complet des données.
    """
    print("=" * 60)
    print("STEP 0: AUDIT DES DONNÉES")
    print("=" * 60)

    # Chargement des données
    df_trafic = load_and_parse_trafic(TRAFIC_FILE)
    df_anomalies = load_and_parse_anomalies(ANOMALIES_FILE)

    # Audit de chaque table
    print("\nAudit de la table patrimoine...")
    audit_t = audit_trafic(df_trafic)

    print("Audit de la table anomalies...")
    audit_a = audit_anomalies(df_anomalies)

    # Vérification de l'alignement
    print("Vérification de l'alignement des clés...")
    join_check = check_join_keys(df_trafic, df_anomalies)

    # Date maximale d'observation
    max_obs = compute_max_observation_date(df_trafic, df_anomalies)

    # Génération du rapport
    print("\nGénération du rapport d'audit...")
    report = generate_audit_report(audit_t, audit_a, join_check, max_obs)

    # Sauvegarde
    report_path = REPORTS_DIR / "data_audit.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nRapport sauvegardé: {report_path}")

    # Sauvegarde des DataFrames nettoyés pour les étapes suivantes
    df_trafic.to_pickle(ARTIFACTS_DIR / "df_trafic_parsed.pkl")
    df_anomalies.to_pickle(ARTIFACTS_DIR / "df_anomalies_parsed.pkl")
    print(f"DataFrames sauvegardés dans {ARTIFACTS_DIR}")

    # Sauvegarde de la date max d'observation
    import json
    metadata = {
        'max_observation_date': max_obs.strftime('%Y-%m-%d'),
        'audit_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(ARTIFACTS_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("AUDIT TERMINÉ")
    print("=" * 60)

    return df_trafic, df_anomalies, max_obs


if __name__ == "__main__":
    main()
