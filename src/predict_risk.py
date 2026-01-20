"""
Module de production: predict_risk()
Fonction principale pour scorer de nouveaux tronçons.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
from typing import Union
import warnings
warnings.filterwarnings('ignore')

# Configuration par défaut
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "artifacts" / "best_model.joblib"


def load_model(model_path: Union[str, Path] = None):
    """
    Charge le modèle depuis le fichier.
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    return joblib.load(model_path)


def compute_features_for_prediction(
    df_assets: pd.DataFrame,
    df_anomalies: pd.DataFrame,
    freeze_date: pd.Timestamp,
    life_stats: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Calcule les features pour la prédiction.

    IMPORTANT: Cette fonction reproduit exactement le feature engineering
    de l'entraînement, en utilisant uniquement des données <= freeze_date.

    Parameters
    ----------
    df_assets : pd.DataFrame
        Table patrimoine avec colonnes: GID, DDP, DHS, MAT, DIAMETRE, LNG
    df_anomalies : pd.DataFrame
        Table anomalies avec colonnes: GID_OBJET, DATE_DETECTION_parsed
    freeze_date : pd.Timestamp
        Date de gel (toutes les features sont calculées avant cette date)
    life_stats : pd.DataFrame, optional
        Statistiques de durée de vie par matériau (si None, calculées depuis df_assets)

    Returns
    -------
    pd.DataFrame
        DataFrame avec toutes les features nécessaires à la prédiction
    """
    df = df_assets.copy()

    # Parser les dates si nécessaire
    if 'DDP_parsed' not in df.columns:
        df['DDP_parsed'] = pd.to_datetime(df['DDP'], errors='coerce', utc=True)
        df['DDP_parsed'] = df['DDP_parsed'].dt.tz_localize(None)

    if 'DHS_parsed' not in df.columns:
        if 'DHS' in df.columns:
            df['DHS_parsed'] = pd.to_datetime(df['DHS'], errors='coerce', utc=True)
            df['DHS_parsed'] = df['DHS_parsed'].dt.tz_localize(None)
        else:
            df['DHS_parsed'] = pd.NaT

    # Filtrer: garder uniquement les tronçons actifs à freeze_date
    # (DDP <= freeze_date ET (DHS is null OR DHS > freeze_date))
    mask_active = (
        (df['DDP_parsed'] <= freeze_date) &
        (df['DHS_parsed'].isna() | (df['DHS_parsed'] > freeze_date))
    )
    df = df[mask_active].copy()

    # =====================
    # FEATURES DE BASE
    # =====================

    # Age au freeze
    df['age_at_freeze'] = (freeze_date - df['DDP_parsed']).dt.days / 365.25

    # Décennie d'installation
    df['decade_install'] = (df['DDP_parsed'].dt.year // 10) * 10

    # Caractéristiques physiques
    df['diametre'] = df['DIAMETRE']
    df['longueur'] = df['LNG']
    df['longueur_km'] = df['LNG'] / 1000
    df['log_longueur'] = np.log1p(df['LNG'])
    df['log_diametre'] = np.log1p(df['DIAMETRE'])
    df['materiau'] = df['MAT']

    # =====================
    # FEATURES D'ANOMALIES
    # =====================

    # Filtrer anomalies avant freeze_date
    ano = df_anomalies.copy()
    if 'DATE_DETECTION' not in ano.columns:
        ano['DATE_DETECTION'] = pd.to_datetime(ano['DATE_DETECTION_parsed'], errors='coerce', utc=True)
        ano['DATE_DETECTION'] = ano['DATE_DETECTION'].dt.tz_localize(None)

    ano = ano[ano['DATE_DETECTION'] <= freeze_date]

    # Compter les anomalies
    ano_total = ano.groupby('GID_OBJET').size().reset_index(name='n_fuites_total')

    for years in [1, 3, 5]:
        cutoff = freeze_date - pd.DateOffset(years=years)
        ano_window = ano[ano['DATE_DETECTION'] > cutoff]
        ano_count = ano_window.groupby('GID_OBJET').size().reset_index(name=f'n_fuites_{years}y')
        ano_total = ano_total.merge(ano_count, on='GID_OBJET', how='left')

    # Date dernière anomalie
    last_ano = ano.groupby('GID_OBJET')['DATE_DETECTION'].max().reset_index()
    last_ano.columns = ['GID_OBJET', 'date_last_fuite']
    ano_total = ano_total.merge(last_ano, on='GID_OBJET', how='left')

    # Joindre au dataframe
    df = df.merge(ano_total, left_on='GID', right_on='GID_OBJET', how='left')

    # Remplir valeurs manquantes
    for col in ['n_fuites_total', 'n_fuites_1y', 'n_fuites_3y', 'n_fuites_5y']:
        df[col] = df[col].fillna(0).astype(int)

    # Jours depuis dernière fuite
    df['days_since_last_fuite'] = (freeze_date - df['date_last_fuite']).dt.days
    max_days = 20 * 365
    df['days_since_last_fuite'] = df['days_since_last_fuite'].fillna(max_days).clip(upper=max_days)

    df['has_recent_fuite'] = (df['days_since_last_fuite'] < 3 * 365).astype(int)

    # Taux de fuite
    df['leak_rate_per_year'] = df['n_fuites_total'] / np.maximum(df['age_at_freeze'], 0.1)
    df['leak_rate_per_km'] = df['n_fuites_total'] / np.maximum(df['longueur_km'], 0.001)

    # =====================
    # FEATURES DE SURVIE
    # =====================

    if life_stats is None:
        # Calculer depuis les défaillances avant freeze_date
        failed = df_assets[
            (df_assets['DHS_parsed'].notna()) &
            (df_assets['DHS_parsed'] <= freeze_date) &
            (df_assets['DDP_parsed'] <= df_assets['DHS_parsed'])
        ].copy()
        failed['life_years'] = (failed['DHS_parsed'] - failed['DDP_parsed']).dt.days / 365.25
        failed = failed[failed['life_years'] > 0]

        life_stats = failed.groupby('MAT')['life_years'].agg([
            ('median_life', 'median'),
            ('p75_life', lambda x: x.quantile(0.75)),
            ('p90_life', lambda x: x.quantile(0.90))
        ]).reset_index()
        life_stats.columns = ['MAT', 'median_life', 'p75_life', 'p90_life']

    df = df.merge(life_stats, left_on='materiau', right_on='MAT', how='left')

    # Valeurs par défaut globales
    global_median = 50.0  # Valeur par défaut raisonnable
    df['median_life'] = df['median_life'].fillna(global_median)
    df['p75_life'] = df['p75_life'].fillna(global_median * 1.2)
    df['p90_life'] = df['p90_life'].fillna(global_median * 1.5)

    # Features dérivées
    df['ratio_age_median'] = df['age_at_freeze'] / np.maximum(df['median_life'], 1)
    df['overdue_years'] = np.maximum(0, df['age_at_freeze'] - df['median_life'])
    df['over_p75_life'] = (df['age_at_freeze'] > df['p75_life']).astype(int)
    df['over_p90_life'] = (df['age_at_freeze'] > df['p90_life']).astype(int)

    # =====================
    # FEATURES D'INTERACTION
    # =====================

    df['age_x_nfuites'] = df['age_at_freeze'] * df['n_fuites_total']
    df['surface_approx'] = df['diametre'] * df['longueur']
    df['age_x_ratio'] = df['age_at_freeze'] * df['ratio_age_median']

    return df


def predict_risk(
    df_assets: pd.DataFrame,
    df_anomalies: pd.DataFrame,
    freeze_date: Union[str, pd.Timestamp],
    horizon_years: int = 1,
    model_path: Union[str, Path] = None
) -> pd.Series:
    """
    Prédit le score de risque pour chaque tronçon.

    Le score retourné est un nombre entre 0 et 1, où un score plus élevé
    indique un risque plus important de défaillance dans l'horizon donné.
    Les tronçons peuvent être ordonnés par score décroissant pour obtenir
    la liste de priorité de renouvellement.

    Parameters
    ----------
    df_assets : pd.DataFrame
        Table patrimoine avec au minimum les colonnes:
        - GID: identifiant unique du tronçon
        - DDP: date de pose (ISO format ou datetime)
        - DHS: date hors service (null si en service)
        - MAT: matériau
        - DIAMETRE: diamètre en mm
        - LNG: longueur en mètres

    df_anomalies : pd.DataFrame
        Table des anomalies avec au minimum les colonnes:
        - GID_OBJET: identifiant du tronçon concerné
        - DATE_DETECTION_parsed: date de détection (ISO format ou datetime)

    freeze_date : str or pd.Timestamp
        Date de gel. Toutes les features sont calculées avec des données
        antérieures ou égales à cette date. Format: 'YYYY-MM-DD'

    horizon_years : int, default=1
        Horizon de prédiction en années (1, 3, ou 5)

    model_path : str or Path, optional
        Chemin vers le modèle. Si None, utilise le best_model.joblib

    Returns
    -------
    pd.Series
        Série indexée par GID contenant les scores de risque (0-1).
        Un score plus élevé = risque plus important.

    Example
    -------
    >>> # Charger les données
    >>> df_assets = pd.read_csv('patrimoine.csv')
    >>> df_anomalies = pd.read_csv('anomalies.csv')
    >>>
    >>> # Prédire les risques
    >>> scores = predict_risk(
    ...     df_assets,
    ...     df_anomalies,
    ...     freeze_date='2024-01-01',
    ...     horizon_years=1
    ... )
    >>>
    >>> # Top 10 tronçons les plus risqués
    >>> top_10 = scores.nlargest(10)
    >>> print(top_10)
    """
    # Convertir freeze_date si nécessaire
    if isinstance(freeze_date, str):
        freeze_date = pd.Timestamp(freeze_date)

    # Charger le modèle
    artifact = load_model(model_path)
    model = artifact['model']
    encoders = artifact['encoders']
    feature_cols = artifact['metadata']['feature_cols']

    # Charger les stats de durée de vie
    artifacts_dir = Path(model_path).parent if model_path else DEFAULT_MODEL_PATH.parent
    life_stats_path = artifacts_dir / f"life_stats_h{horizon_years}.csv"

    if life_stats_path.exists():
        life_stats = pd.read_csv(life_stats_path)
    else:
        life_stats = None

    # Calculer les features
    df_features = compute_features_for_prediction(
        df_assets, df_anomalies, freeze_date, life_stats
    )

    if len(df_features) == 0:
        return pd.Series(dtype=float)

    # Préparer X avec les mêmes colonnes que l'entraînement
    # Encoder les catégorielles
    for col, le in encoders.items():
        if col in df_features.columns:
            # Gérer les catégories non vues
            df_features[col + '_encoded'] = df_features[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )
        else:
            df_features[col + '_encoded'] = 0

    # Sélectionner les colonnes de features
    X = df_features[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Prédire
    if hasattr(model, 'predict_proba'):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = model.predict(X)

    # Retourner comme Series indexée par GID
    return pd.Series(scores, index=df_features['GID'].values, name='risk_score')


def get_prioritized_list(
    df_assets: pd.DataFrame,
    df_anomalies: pd.DataFrame,
    freeze_date: Union[str, pd.Timestamp],
    horizon_years: int = 1,
    top_k_pct: float = 0.10,
    model_path: Union[str, Path] = None
) -> pd.DataFrame:
    """
    Retourne la liste priorisée des tronçons à renouveler.

    Parameters
    ----------
    df_assets, df_anomalies, freeze_date, horizon_years, model_path :
        Voir predict_risk()
    top_k_pct : float, default=0.10
        Pourcentage des tronçons à retourner (ex: 0.10 = top 10%)

    Returns
    -------
    pd.DataFrame
        DataFrame avec les tronçons prioritaires, incluant:
        - GID
        - risk_score
        - rank
        - Caractéristiques principales du tronçon
    """
    # Calculer les scores
    scores = predict_risk(df_assets, df_anomalies, freeze_date, horizon_years, model_path)

    # Trier par score décroissant
    scores_sorted = scores.sort_values(ascending=False)

    # Prendre le top-k
    n_top = int(np.ceil(len(scores_sorted) * top_k_pct))
    top_gids = scores_sorted.head(n_top)

    # Construire le DataFrame de sortie
    result = pd.DataFrame({
        'GID': top_gids.index,
        'risk_score': top_gids.values,
        'rank': range(1, n_top + 1)
    })

    # Ajouter les infos du tronçon
    df_assets_info = df_assets.set_index('GID')[['MAT', 'DIAMETRE', 'LNG', 'DDP']]
    result = result.merge(df_assets_info, left_on='GID', right_index=True, how='left')

    return result


# Test rapide si exécuté directement
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from config import ARTIFACTS_DIR, DATA_DIR

    print("Test de predict_risk()")
    print("=" * 60)

    # Charger les données
    df_assets = pd.read_csv(DATA_DIR / "v1_trafic_prepared.csv")
    df_anomalies = pd.read_csv(DATA_DIR / "historiqueanomalie.csv")

    # Parser les dates
    df_assets['DDP_parsed'] = pd.to_datetime(df_assets['DDP'], errors='coerce', utc=True)
    df_assets['DDP_parsed'] = df_assets['DDP_parsed'].dt.tz_localize(None)
    df_assets['DHS_parsed'] = pd.to_datetime(df_assets['DHS'], errors='coerce', utc=True)
    df_assets['DHS_parsed'] = df_assets['DHS_parsed'].dt.tz_localize(None)

    df_anomalies['DATE_DETECTION'] = pd.to_datetime(df_anomalies['DATE_DETECTION_parsed'], errors='coerce', utc=True)
    df_anomalies['DATE_DETECTION'] = df_anomalies['DATE_DETECTION'].dt.tz_localize(None)

    # Tester
    freeze_date = '2024-01-01'
    print(f"Freeze date: {freeze_date}")

    scores = predict_risk(df_assets, df_anomalies, freeze_date, horizon_years=1)
    print(f"\nNombre de troncons scores: {len(scores)}")
    print(f"Score min: {scores.min():.4f}")
    print(f"Score max: {scores.max():.4f}")
    print(f"Score median: {scores.median():.4f}")

    print("\nTop 10 troncons les plus risques:")
    top_10 = get_prioritized_list(df_assets, df_anomalies, freeze_date, top_k_pct=0.001)
    print(top_10.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("Test termine avec succes!")
