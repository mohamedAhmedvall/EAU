"""
Module de prédiction pour l'API.
Charge le modèle et effectue les prédictions.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PipeRiskPredictor:
    """
    Classe pour prédire le risque de défaillance des canalisations.
    """

    def __init__(self, model_path: Path = None):
        """
        Initialise le prédicteur.

        Parameters
        ----------
        model_path : Path, optional
            Chemin vers le modèle. Si None, cherche dans ../artifacts/
        """
        self.model = None
        self.encoders = None
        self.metadata = None
        self.life_stats = None
        self.is_loaded = False

        if model_path is None:
            model_path = Path(__file__).parent.parent / "artifacts" / "best_model.joblib"

        self.model_path = model_path
        self.artifacts_dir = model_path.parent

    def load(self) -> bool:
        """
        Charge le modèle et ses dépendances.

        Returns
        -------
        bool
            True si le chargement a réussi
        """
        try:
            # Charger l'artifact principal
            artifact = joblib.load(self.model_path)
            self.model = artifact['model']
            self.encoders = artifact['encoders']
            self.metadata = artifact['metadata']

            # Charger les stats de durée de vie
            horizon = self.metadata.get('horizon_years', 1)
            life_stats_path = self.artifacts_dir / f"life_stats_h{horizon}.csv"
            if life_stats_path.exists():
                self.life_stats = pd.read_csv(life_stats_path)
            else:
                self.life_stats = None

            self.is_loaded = True
            return True

        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            self.is_loaded = False
            return False

    def _prepare_features(
        self,
        df_pipes: pd.DataFrame,
        df_anomalies: pd.DataFrame,
        freeze_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Prépare les features pour la prédiction.
        """
        df = df_pipes.copy()

        # Parser les dates
        df['DDP_parsed'] = pd.to_datetime(df['DDP'], errors='coerce')
        if 'DHS' in df.columns:
            df['DHS_parsed'] = pd.to_datetime(df['DHS'], errors='coerce')
        else:
            df['DHS_parsed'] = pd.NaT

        # Filtrer: garder uniquement les tronçons actifs à freeze_date
        mask_active = (
            (df['DDP_parsed'] <= freeze_date) &
            (df['DHS_parsed'].isna() | (df['DHS_parsed'] > freeze_date))
        )
        df = df[mask_active].copy()

        if len(df) == 0:
            return df

        # =====================
        # FEATURES DE BASE
        # =====================
        df['age_at_freeze'] = (freeze_date - df['DDP_parsed']).dt.days / 365.25
        df['decade_install'] = (df['DDP_parsed'].dt.year // 10) * 10
        df['diametre'] = df['DIAMETRE']
        df['longueur'] = df['LNG']
        df['longueur_km'] = df['LNG'] / 1000
        df['log_longueur'] = np.log1p(df['LNG'])
        df['log_diametre'] = np.log1p(df['DIAMETRE'])
        df['materiau'] = df['MAT']

        # =====================
        # FEATURES D'ANOMALIES
        # =====================
        if len(df_anomalies) > 0:
            ano = df_anomalies.copy()
            ano['DATE_DETECTION'] = pd.to_datetime(ano['DATE_DETECTION'], errors='coerce')
            ano = ano[ano['DATE_DETECTION'] <= freeze_date]

            if len(ano) > 0:
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

                # Joindre
                df = df.merge(ano_total, left_on='GID', right_on='GID_OBJET', how='left')

        # Remplir valeurs manquantes
        for col in ['n_fuites_total', 'n_fuites_1y', 'n_fuites_3y', 'n_fuites_5y']:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].fillna(0).astype(int)

        # Jours depuis dernière fuite
        if 'date_last_fuite' not in df.columns:
            df['date_last_fuite'] = pd.NaT

        df['days_since_last_fuite'] = (freeze_date - df['date_last_fuite']).dt.days
        max_days = 20 * 365
        df['days_since_last_fuite'] = df['days_since_last_fuite'].fillna(max_days).clip(upper=max_days)

        df['has_recent_fuite'] = (df['days_since_last_fuite'] < 3 * 365).astype(int)
        df['leak_rate_per_year'] = df['n_fuites_total'] / np.maximum(df['age_at_freeze'], 0.1)
        df['leak_rate_per_km'] = df['n_fuites_total'] / np.maximum(df['longueur_km'], 0.001)

        # =====================
        # FEATURES DE SURVIE
        # =====================
        if self.life_stats is not None:
            df = df.merge(self.life_stats, left_on='materiau', right_on='MAT', how='left')

        # Valeurs par défaut
        global_median = 50.0
        for col in ['median_life', 'p75_life', 'p90_life']:
            if col not in df.columns:
                df[col] = global_median
            df[col] = df[col].fillna(global_median)

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

    def predict(
        self,
        pipes: List[Dict],
        anomalies: List[Dict],
        freeze_date: str,
        horizon_years: int = 1
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Prédit les scores de risque pour une liste de tronçons.

        Parameters
        ----------
        pipes : List[Dict]
            Liste des tronçons (format dictionnaire)
        anomalies : List[Dict]
            Liste des anomalies
        freeze_date : str
            Date de gel (format YYYY-MM-DD)
        horizon_years : int
            Horizon de prédiction

        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            DataFrame avec les prédictions et résumé statistique
        """
        if not self.is_loaded:
            raise RuntimeError("Le modèle n'est pas chargé. Appelez load() d'abord.")

        # Convertir en DataFrames
        df_pipes = pd.DataFrame(pipes)
        df_anomalies = pd.DataFrame(anomalies) if anomalies else pd.DataFrame()
        freeze_dt = pd.Timestamp(freeze_date)

        # Préparer les features
        df_features = self._prepare_features(df_pipes, df_anomalies, freeze_dt)

        if len(df_features) == 0:
            return pd.DataFrame(), {"error": "Aucun tronçon éligible"}

        # Encoder les catégorielles
        feature_cols = self.metadata['feature_cols']

        for col, le in self.encoders.items():
            if col in df_features.columns:
                df_features[col + '_encoded'] = df_features[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
            else:
                df_features[col + '_encoded'] = 0

        # Sélectionner les colonnes de features
        X = df_features[feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Prédire
        if hasattr(self.model, 'predict_proba'):
            scores = self.model.predict_proba(X)[:, 1]
        else:
            scores = self.model.predict(X)

        # Construire le résultat
        result = pd.DataFrame({
            'GID': df_features['GID'].values,
            'risk_score': scores
        })

        # Trier et ajouter le rang
        result = result.sort_values('risk_score', ascending=False).reset_index(drop=True)
        result['rank'] = range(1, len(result) + 1)

        # Catégoriser le risque
        def categorize_risk(score):
            if score >= 0.7:
                return "CRITIQUE"
            elif score >= 0.4:
                return "ELEVE"
            elif score >= 0.2:
                return "MOYEN"
            else:
                return "FAIBLE"

        result['risk_category'] = result['risk_score'].apply(categorize_risk)

        # Résumé (convertir en types Python natifs pour JSON)
        summary = {
            'n_critique': int((result['risk_category'] == 'CRITIQUE').sum()),
            'n_eleve': int((result['risk_category'] == 'ELEVE').sum()),
            'n_moyen': int((result['risk_category'] == 'MOYEN').sum()),
            'n_faible': int((result['risk_category'] == 'FAIBLE').sum()),
            'score_mean': float(result['risk_score'].mean()),
            'score_median': float(result['risk_score'].median()),
            'score_max': float(result['risk_score'].max()),
            'score_min': float(result['risk_score'].min())
        }

        return result, summary

    def get_top_k(
        self,
        pipes: List[Dict],
        anomalies: List[Dict],
        freeze_date: str,
        horizon_years: int = 1,
        top_k_percent: float = 0.10
    ) -> pd.DataFrame:
        """
        Retourne le top-k% des tronçons les plus risqués.
        """
        result, _ = self.predict(pipes, anomalies, freeze_date, horizon_years)

        if len(result) == 0:
            return result

        n_top = int(np.ceil(len(result) * top_k_percent))
        return result.head(n_top)

    def get_model_info(self) -> Dict:
        """
        Retourne les informations sur le modèle.
        """
        if not self.is_loaded:
            return {"error": "Modèle non chargé"}

        return {
            'model_type': self.model.name if hasattr(self.model, 'name') else type(self.model).__name__,
            'horizon_years': self.metadata.get('horizon_years', 1),
            'n_features': len(self.metadata.get('feature_cols', [])),
            'feature_names': self.metadata.get('feature_cols', []),
            'training_date': self.metadata.get('created_at', 'N/A'),
            'performance': self.metadata.get('test_metrics', {})
        }


# Singleton pour réutilisation
_predictor_instance: Optional[PipeRiskPredictor] = None


def get_predictor() -> PipeRiskPredictor:
    """
    Retourne l'instance singleton du prédicteur.
    """
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = PipeRiskPredictor()
        _predictor_instance.load()
    return _predictor_instance
