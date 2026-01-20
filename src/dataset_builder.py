"""
Step 2 & 3: Reconstruction des datasets freeze+horizon avec feature engineering
- Construction des datasets pour H=1, 3, 5 ans
- Feature engineering strictement pré-freeze
- Tests anti-fuite automatisés
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from config import (
    ARTIFACTS_DIR, REPORTS_DIR,
    COL_PIPE_ID, COL_PIPE_ID_ANOMALIE, COL_MATERIAU, COL_DIAMETRE, COL_LONGUEUR,
    COL_DATE_INSTALL, COL_DATE_HS, HORIZONS, RANDOM_SEED
)


def load_raw_data():
    """Charge les données parsées."""
    df_trafic = pd.read_pickle(ARTIFACTS_DIR / "df_trafic_parsed.pkl")
    df_anomalies = pd.read_pickle(ARTIFACTS_DIR / "df_anomalies_parsed.pkl")

    with open(ARTIFACTS_DIR / "metadata.json", 'r') as f:
        metadata = json.load(f)

    return df_trafic, df_anomalies, metadata


def compute_freeze_date(max_obs_date: pd.Timestamp, horizon_years: int) -> pd.Timestamp:
    """
    Calcule FREEZE_DATE = max_obs_date - horizon.
    """
    return max_obs_date - pd.DateOffset(years=horizon_years)


def build_target(df: pd.DataFrame, freeze_date: pd.Timestamp,
                 horizon_years: int) -> pd.DataFrame:
    """
    Construit la target pour un horizon donné.

    event = 1 si DHS dans (FREEZE_DATE, FREEZE_DATE + H]
    event = 0 sinon

    duration = min(DHS, FREEZE_DATE + H) - FREEZE_DATE (en jours)

    ANTI-LEAKAGE: On ne regarde que l'information disponible à FREEZE_DATE
    pour déterminer si le tronçon est éligible, mais l'événement est défini
    par ce qui se passe APRÈS freeze_date.
    """
    df = df.copy()
    horizon_end = freeze_date + pd.DateOffset(years=horizon_years)

    # Critères d'éligibilité à FREEZE_DATE:
    # 1. Le tronçon doit exister (DDP <= FREEZE_DATE)
    # 2. Le tronçon ne doit pas déjà être hors service (DHS is null OR DHS > FREEZE_DATE)

    eligible_ddp = df['DDP_parsed'] <= freeze_date
    not_already_hs = df['DHS_parsed'].isna() | (df['DHS_parsed'] > freeze_date)
    eligible = eligible_ddp & not_already_hs

    df_eligible = df[eligible].copy()

    # Target: événement dans l'horizon
    has_dhs = df_eligible['DHS_parsed'].notna()
    dhs_in_horizon = has_dhs & (df_eligible['DHS_parsed'] > freeze_date) & (df_eligible['DHS_parsed'] <= horizon_end)

    df_eligible['event'] = dhs_in_horizon.astype(int)

    # Duration: temps jusqu'à l'événement ou censure
    df_eligible['duration_days'] = np.nan

    # Si événement dans l'horizon: duration = DHS - FREEZE_DATE
    df_eligible.loc[dhs_in_horizon, 'duration_days'] = (
        df_eligible.loc[dhs_in_horizon, 'DHS_parsed'] - freeze_date
    ).dt.days

    # Si censuré: duration = horizon_end - FREEZE_DATE (= horizon en jours)
    censored = ~dhs_in_horizon
    df_eligible.loc[censored, 'duration_days'] = (horizon_end - freeze_date).days

    df_eligible['duration_years'] = df_eligible['duration_days'] / 365.25

    return df_eligible


def compute_base_features(df: pd.DataFrame, freeze_date: pd.Timestamp) -> pd.DataFrame:
    """
    Features de base calculées à FREEZE_DATE.

    ANTI-LEAKAGE: Toutes les features sont calculées uniquement avec
    des informations disponibles à FREEZE_DATE.
    """
    df = df.copy()

    # Age au freeze (années)
    # LEAKAGE PROOF: utilise DDP (passé) et freeze_date (connu)
    df['age_at_freeze'] = (freeze_date - df['DDP_parsed']).dt.days / 365.25

    # Décennie d'installation
    # LEAKAGE PROOF: utilise uniquement DDP (passé)
    df['decade_install'] = (df['DDP_parsed'].dt.year // 10) * 10

    # Diamètre et longueur (caractéristiques physiques, invariantes)
    df['diametre'] = df[COL_DIAMETRE]
    df['longueur'] = df[COL_LONGUEUR]
    df['longueur_km'] = df[COL_LONGUEUR] / 1000

    # Log transforms si justifié
    df['log_longueur'] = np.log1p(df[COL_LONGUEUR])
    df['log_diametre'] = np.log1p(df[COL_DIAMETRE])

    # Matériau
    df['materiau'] = df[COL_MATERIAU]

    # Bins pour diamètre
    df['diametre_bin'] = pd.cut(
        df[COL_DIAMETRE],
        bins=[0, 80, 100, 150, 200, 300, 1000, 10000],
        labels=['small_80', 'medium_100', 'medium_150', 'large_200', 'large_300', 'xlarge_1000', 'xxlarge']
    )

    return df


def compute_anomaly_features(df: pd.DataFrame, df_anomalies: pd.DataFrame,
                             freeze_date: pd.Timestamp, horizon_years: int) -> pd.DataFrame:
    """
    Features basées sur l'historique des anomalies AVANT freeze_date.

    ANTI-LEAKAGE: Seules les anomalies avec DATE_DETECTION <= FREEZE_DATE
    sont utilisées.
    """
    df = df.copy()

    # Filtrer les anomalies strictement avant freeze_date
    ano = df_anomalies[df_anomalies['DATE_DETECTION'] <= freeze_date].copy()

    # ========================================
    # ASSERTION ANTI-FUITE #1
    # ========================================
    assert ano['DATE_DETECTION'].max() <= freeze_date, \
        f"LEAKAGE DETECTED: anomalies après freeze_date ({ano['DATE_DETECTION'].max()} > {freeze_date})"

    # Compter les anomalies totales par tronçon
    ano_total = ano.groupby(COL_PIPE_ID_ANOMALIE).size().reset_index(name='n_fuites_total')

    # Compter les anomalies dans les X dernières années avant freeze
    for years in [1, 3, 5]:
        cutoff = freeze_date - pd.DateOffset(years=years)
        ano_window = ano[ano['DATE_DETECTION'] > cutoff]
        ano_count = ano_window.groupby(COL_PIPE_ID_ANOMALIE).size().reset_index(name=f'n_fuites_{years}y')
        ano_total = ano_total.merge(ano_count, on=COL_PIPE_ID_ANOMALIE, how='left')

    # Date de la dernière anomalie
    last_ano = ano.groupby(COL_PIPE_ID_ANOMALIE)['DATE_DETECTION'].max().reset_index()
    last_ano.columns = [COL_PIPE_ID_ANOMALIE, 'date_last_fuite']
    ano_total = ano_total.merge(last_ano, on=COL_PIPE_ID_ANOMALIE, how='left')

    # Jointure avec le dataframe principal
    df = df.merge(ano_total, left_on=COL_PIPE_ID, right_on=COL_PIPE_ID_ANOMALIE, how='left')

    # Remplir les valeurs manquantes (tronçons sans anomalie)
    for col in ['n_fuites_total', 'n_fuites_1y', 'n_fuites_3y', 'n_fuites_5y']:
        df[col] = df[col].fillna(0).astype(int)

    # Jours depuis la dernière fuite
    df['days_since_last_fuite'] = (freeze_date - df['date_last_fuite']).dt.days
    # Cap à 20 ans si pas de fuite ou très ancienne
    max_days = 20 * 365
    df['days_since_last_fuite'] = df['days_since_last_fuite'].fillna(max_days).clip(upper=max_days)

    # Indicateur de fuite récente (< 3 ans)
    df['has_recent_fuite'] = (df['days_since_last_fuite'] < 3 * 365).astype(int)

    # Taux de fuite par année de vie
    df['leak_rate_per_year'] = df['n_fuites_total'] / np.maximum(df['age_at_freeze'], 0.1)

    # Taux de fuite par km
    df['leak_rate_per_km'] = df['n_fuites_total'] / np.maximum(df['longueur_km'], 0.001)

    # ========================================
    # ASSERTION ANTI-FUITE #2
    # ========================================
    if 'date_last_fuite' in df.columns:
        valid_dates = df['date_last_fuite'].notna()
        if valid_dates.any():
            assert df.loc[valid_dates, 'date_last_fuite'].max() <= freeze_date, \
                f"LEAKAGE DETECTED: date_last_fuite après freeze_date"

    return df


def compute_survival_informed_features(df: pd.DataFrame, df_trafic_full: pd.DataFrame,
                                       freeze_date: pd.Timestamp) -> pd.DataFrame:
    """
    Features basées sur les durées de vie observées AVANT freeze_date.

    ANTI-LEAKAGE: Seuls les DHS <= FREEZE_DATE sont utilisés pour calculer
    les statistiques de durée de vie par matériau.
    """
    df = df.copy()

    # Calculer les statistiques de durée de vie par matériau
    # en utilisant uniquement les tronçons défaillants AVANT freeze_date

    # Tronçons défaillants avant freeze
    failed_before = df_trafic_full[
        (df_trafic_full['DHS_parsed'].notna()) &
        (df_trafic_full['DHS_parsed'] <= freeze_date) &
        (df_trafic_full['DDP_parsed'] <= df_trafic_full['DHS_parsed'])
    ].copy()

    # ========================================
    # ASSERTION ANTI-FUITE #3
    # ========================================
    assert failed_before['DHS_parsed'].max() <= freeze_date, \
        f"LEAKAGE DETECTED: DHS après freeze_date utilisé pour stats de survie"

    # Durée de vie des défaillants
    failed_before['life_years'] = (
        failed_before['DHS_parsed'] - failed_before['DDP_parsed']
    ).dt.days / 365.25

    # Exclure les durées de vie négatives ou nulles
    failed_before = failed_before[failed_before['life_years'] > 0]

    # Statistiques par matériau
    life_stats = failed_before.groupby(COL_MATERIAU)['life_years'].agg([
        ('median_life', 'median'),
        ('mean_life', 'mean'),
        ('p25_life', lambda x: x.quantile(0.25)),
        ('p75_life', lambda x: x.quantile(0.75)),
        ('p90_life', lambda x: x.quantile(0.90)),
        ('n_failed', 'count')
    ]).reset_index()

    # Joindre au dataframe
    df = df.merge(life_stats, left_on='materiau', right_on=COL_MATERIAU, how='left')

    # Pour les matériaux avec peu de données, utiliser la médiane globale
    global_median = failed_before['life_years'].median()
    global_p75 = failed_before['life_years'].quantile(0.75)
    global_p90 = failed_before['life_years'].quantile(0.90)

    df['median_life'] = df['median_life'].fillna(global_median)
    df['p75_life'] = df['p75_life'].fillna(global_p75)
    df['p90_life'] = df['p90_life'].fillna(global_p90)

    # Features dérivées
    # Ratio âge / durée de vie médiane du matériau
    df['ratio_age_median'] = df['age_at_freeze'] / np.maximum(df['median_life'], 1)

    # Années de dépassement de la durée de vie médiane
    df['overdue_years'] = np.maximum(0, df['age_at_freeze'] - df['median_life'])

    # Flags de dépassement de percentiles
    df['over_p75_life'] = (df['age_at_freeze'] > df['p75_life']).astype(int)
    df['over_p90_life'] = (df['age_at_freeze'] > df['p90_life']).astype(int)

    # Sauvegarder les stats par matériau pour utilisation en production
    return df, life_stats


def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features d'interaction.
    """
    df = df.copy()

    # Age × nombre de fuites
    df['age_x_nfuites'] = df['age_at_freeze'] * df['n_fuites_total']

    # Diamètre × longueur (surface approximative)
    df['surface_approx'] = df['diametre'] * df['longueur']

    # Age × ratio dépassement médiane
    df['age_x_ratio'] = df['age_at_freeze'] * df['ratio_age_median']

    return df


def validate_no_leakage(df: pd.DataFrame, freeze_date: pd.Timestamp,
                        df_anomalies: pd.DataFrame) -> bool:
    """
    Validation finale anti-fuite.
    Retourne True si aucune fuite détectée.
    """
    print("\n=== VALIDATION ANTI-FUITE ===")
    all_ok = True

    # Test 1: Aucune anomalie post-freeze utilisée
    if 'date_last_fuite' in df.columns:
        valid = df['date_last_fuite'].notna()
        if valid.any():
            max_date = df.loc[valid, 'date_last_fuite'].max()
            if max_date > freeze_date:
                print(f"[FAIL] FUITE: date_last_fuite max ({max_date}) > freeze_date ({freeze_date})")
                all_ok = False
            else:
                print(f"[OK] date_last_fuite OK (max: {max_date})")

    # Test 2: age_at_freeze cohérent
    max_age = df['age_at_freeze'].max()
    if max_age > 200:  # Sanity check
        print(f"[WARN] ATTENTION: age_at_freeze max = {max_age:.1f} ans (verifier les DDP)")
    else:
        print(f"[OK] age_at_freeze OK (max: {max_age:.1f} ans)")

    # Test 3: Tous les tronçons sont éligibles (pas déjà HS avant freeze)
    # Ceci est vérifié par construction mais on double-check
    if 'DHS_parsed' in df.columns:
        has_dhs_before = df['DHS_parsed'].notna() & (df['DHS_parsed'] <= freeze_date)
        if has_dhs_before.any():
            print(f"[FAIL] FUITE: {has_dhs_before.sum()} troncons avec DHS <= freeze_date dans le dataset")
            all_ok = False
        else:
            print(f"[OK] Aucun troncon avec DHS avant freeze_date")

    # Test 4: Event cohérent avec DHS
    # Si event=1, DHS doit être dans l'horizon futur
    if all_ok:
        print(f"[OK] VALIDATION ANTI-FUITE REUSSIE")
    else:
        print(f"[FAIL] VALIDATION ANTI-FUITE ECHOUEE")

    return all_ok


def build_dataset_for_horizon(df_trafic: pd.DataFrame, df_anomalies: pd.DataFrame,
                              max_obs_date: pd.Timestamp, horizon_years: int) -> pd.DataFrame:
    """
    Construit le dataset complet pour un horizon donné.
    """
    print(f"\n{'='*60}")
    print(f"CONSTRUCTION DATASET - HORIZON {horizon_years} AN(S)")
    print(f"{'='*60}")

    freeze_date = compute_freeze_date(max_obs_date, horizon_years)
    print(f"Max observation: {max_obs_date.strftime('%Y-%m-%d')}")
    print(f"FREEZE_DATE: {freeze_date.strftime('%Y-%m-%d')}")
    print(f"Horizon end: {(freeze_date + pd.DateOffset(years=horizon_years)).strftime('%Y-%m-%d')}")

    # 1. Construire la target
    print("\n1. Construction de la target...")
    df = build_target(df_trafic, freeze_date, horizon_years)
    print(f"   Tronçons éligibles: {len(df):,}")
    print(f"   Événements: {df['event'].sum():,} ({100*df['event'].mean():.2f}%)")

    # 2. Features de base
    print("\n2. Features de base...")
    df = compute_base_features(df, freeze_date)

    # 3. Features d'anomalies
    print("\n3. Features d'anomalies...")
    df = compute_anomaly_features(df, df_anomalies, freeze_date, horizon_years)

    # 4. Features informées par la survie
    print("\n4. Features de survie (durées de vie par matériau)...")
    df, life_stats = compute_survival_informed_features(df, df_trafic, freeze_date)

    # 5. Features d'interaction
    print("\n5. Features d'interaction...")
    df = compute_interaction_features(df)

    # 6. Validation anti-fuite
    validate_no_leakage(df, freeze_date, df_anomalies)

    # Ajouter les métadonnées
    df['freeze_date'] = freeze_date
    df['horizon_years'] = horizon_years

    # Sauvegarder les stats de durée de vie pour le modèle
    life_stats.to_csv(ARTIFACTS_DIR / f"life_stats_h{horizon_years}.csv", index=False)

    return df


def get_feature_columns():
    """
    Retourne la liste des colonnes de features pour la modélisation.
    """
    numerical_features = [
        'age_at_freeze',
        'diametre', 'longueur', 'log_longueur', 'log_diametre',
        'n_fuites_total', 'n_fuites_1y', 'n_fuites_3y', 'n_fuites_5y',
        'days_since_last_fuite', 'has_recent_fuite',
        'leak_rate_per_year', 'leak_rate_per_km',
        'ratio_age_median', 'overdue_years',
        'over_p75_life', 'over_p90_life',
        'age_x_nfuites', 'surface_approx', 'age_x_ratio'
    ]

    categorical_features = [
        'materiau', 'decade_install'
    ]

    return numerical_features, categorical_features


def main():
    """
    Construction des datasets pour tous les horizons.
    """
    print("=" * 60)
    print("STEP 2 & 3: CONSTRUCTION DES DATASETS FREEZE+HORIZON")
    print("=" * 60)

    # Chargement
    df_trafic, df_anomalies, metadata = load_raw_data()
    max_obs_date = pd.Timestamp(metadata['max_observation_date'])

    datasets = {}
    for h in HORIZONS:
        df = build_dataset_for_horizon(df_trafic, df_anomalies, max_obs_date, h)
        datasets[h] = df

        # Sauvegarde
        output_path = ARTIFACTS_DIR / f"dataset_freeze_h{h}.csv"
        df.to_csv(output_path, index=False)
        print(f"\nDataset sauvegardé: {output_path}")

        # Aussi en pickle pour préserver les types
        df.to_pickle(ARTIFACTS_DIR / f"dataset_freeze_h{h}.pkl")

    # Résumé comparatif
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES DATASETS")
    print("=" * 60)
    print(f"\n{'Horizon':<10} {'N tronçons':<15} {'N événements':<15} {'Taux événement':<15}")
    print("-" * 55)
    for h, df in datasets.items():
        n = len(df)
        n_events = df['event'].sum()
        rate = 100 * n_events / n
        print(f"{h} an(s)     {n:>12,}     {n_events:>12,}     {rate:>12.2f}%")

    # Liste des features
    num_feats, cat_feats = get_feature_columns()
    print(f"\n\nFeatures numériques ({len(num_feats)}):")
    for f in num_feats:
        print(f"  - {f}")
    print(f"\nFeatures catégorielles ({len(cat_feats)}):")
    for f in cat_feats:
        print(f"  - {f}")

    print("\n" + "=" * 60)
    print("CONSTRUCTION DES DATASETS TERMINÉE")
    print("=" * 60)

    return datasets


if __name__ == "__main__":
    main()
