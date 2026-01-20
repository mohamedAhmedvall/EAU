"""
Modèles Pydantic pour l'API REST.
Définition des schémas de requête et réponse.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date


class PipeSegment(BaseModel):
    """Schéma d'un tronçon de canalisation."""
    GID: int = Field(..., description="Identifiant unique du tronçon")
    DDP: str = Field(..., description="Date de pose (format ISO: YYYY-MM-DD)")
    DHS: Optional[str] = Field(None, description="Date hors service (null si en service)")
    MAT: str = Field(..., description="Matériau (FT, FTG, PEHD, PVC, etc.)")
    DIAMETRE: float = Field(..., description="Diamètre en mm")
    LNG: float = Field(..., description="Longueur en mètres")

    class Config:
        json_schema_extra = {
            "example": {
                "GID": 410966779,
                "DDP": "2009-01-01",
                "DHS": None,
                "MAT": "FT",
                "DIAMETRE": 100.0,
                "LNG": 82.726
            }
        }


class Anomaly(BaseModel):
    """Schéma d'une anomalie/fuite."""
    GID_OBJET: int = Field(..., description="Identifiant du tronçon concerné")
    DATE_DETECTION: str = Field(..., description="Date de détection (format ISO)")
    TYPE_ANOMALIE: Optional[str] = Field("FUITE", description="Type d'anomalie")

    class Config:
        json_schema_extra = {
            "example": {
                "GID_OBJET": 410030412,
                "DATE_DETECTION": "2017-08-02",
                "TYPE_ANOMALIE": "FUITE_DETECT_TR"
            }
        }


class PredictionRequest(BaseModel):
    """Requête de prédiction."""
    pipes: List[PipeSegment] = Field(..., description="Liste des tronçons à scorer")
    anomalies: List[Anomaly] = Field(default=[], description="Historique des anomalies")
    freeze_date: str = Field(..., description="Date de gel (format: YYYY-MM-DD)")
    horizon_years: int = Field(default=1, description="Horizon de prédiction (1, 3 ou 5 ans)")

    class Config:
        json_schema_extra = {
            "example": {
                "pipes": [
                    {"GID": 410966779, "DDP": "2009-01-01", "DHS": None, "MAT": "FT", "DIAMETRE": 100, "LNG": 82.7},
                    {"GID": 410000670, "DDP": "1963-01-01", "DHS": None, "MAT": "FT", "DIAMETRE": 300, "LNG": 71.4}
                ],
                "anomalies": [
                    {"GID_OBJET": 410000670, "DATE_DETECTION": "2015-03-15", "TYPE_ANOMALIE": "FUITE_SIGNAL_TR"},
                    {"GID_OBJET": 410000670, "DATE_DETECTION": "2018-07-22", "TYPE_ANOMALIE": "FUITE_DETECT_TR"}
                ],
                "freeze_date": "2024-01-01",
                "horizon_years": 1
            }
        }


class PipeRiskScore(BaseModel):
    """Score de risque pour un tronçon."""
    GID: int = Field(..., description="Identifiant du tronçon")
    risk_score: float = Field(..., description="Score de risque (0-1, plus élevé = plus risqué)")
    rank: int = Field(..., description="Rang de priorité (1 = plus risqué)")
    risk_category: str = Field(..., description="Catégorie de risque (CRITIQUE, ELEVE, MOYEN, FAIBLE)")


class PredictionResponse(BaseModel):
    """Réponse de prédiction."""
    success: bool
    freeze_date: str
    horizon_years: int
    total_pipes: int
    predictions: List[PipeRiskScore]
    summary: dict

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "freeze_date": "2024-01-01",
                "horizon_years": 1,
                "total_pipes": 2,
                "predictions": [
                    {"GID": 410000670, "risk_score": 0.85, "rank": 1, "risk_category": "CRITIQUE"},
                    {"GID": 410966779, "risk_score": 0.23, "rank": 2, "risk_category": "FAIBLE"}
                ],
                "summary": {
                    "n_critique": 1,
                    "n_eleve": 0,
                    "n_moyen": 0,
                    "n_faible": 1
                }
            }
        }


class TopKRequest(BaseModel):
    """Requête pour obtenir le top-k des tronçons prioritaires."""
    pipes: List[PipeSegment]
    anomalies: List[Anomaly] = []
    freeze_date: str
    horizon_years: int = 1
    top_k_percent: float = Field(default=0.10, description="Pourcentage du top (0.05, 0.10, 0.20)")


class HealthResponse(BaseModel):
    """Réponse de health check."""
    status: str
    model_loaded: bool
    model_type: str
    horizon: int
    version: str


class ModelInfoResponse(BaseModel):
    """Informations sur le modèle."""
    model_type: str
    horizon_years: int
    n_features: int
    feature_names: List[str]
    training_date: str
    performance: dict
