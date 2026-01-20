"""
API REST pour le modèle de priorisation des renouvellements de canalisations.
FastAPI application principale.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import io
from datetime import datetime

from models import (
    PipeSegment, Anomaly, PredictionRequest, PredictionResponse,
    PipeRiskScore, TopKRequest, HealthResponse, ModelInfoResponse
)
from predictor import get_predictor, PipeRiskPredictor

# Créer l'application FastAPI
app = FastAPI(
    title="API Priorisation Renouvellement Canalisations",
    description="""
## API de prédiction du risque de défaillance des canalisations d'eau potable

Cette API permet de :
- **Scorer** des tronçons de canalisation selon leur risque de défaillance
- **Prioriser** les renouvellements en identifiant les tronçons les plus à risque
- **Obtenir le top-k%** des tronçons prioritaires

### Modèle utilisé
- **Type** : LightGBM (Gradient Boosting)
- **Horizon** : 1 an
- **Performance** : Capture@10% = 60%, Lift = 6x

### Métriques clés
- En ciblant les 10% de tronçons les plus risqués, on capture 60% des défaillances réelles
- C'est 6 fois mieux qu'une sélection aléatoire
    """,
    version="1.0.0",
    contact={
        "name": "Data Science Team",
        "email": "datascience@example.com"
    }
)

# CORS middleware pour permettre les appels cross-origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le prédicteur au démarrage
predictor: PipeRiskPredictor = None


@app.on_event("startup")
async def startup_event():
    """Charge le modèle au démarrage de l'API."""
    global predictor
    predictor = get_predictor()
    if predictor.is_loaded:
        print("Modele charge avec succes")
    else:
        print("ATTENTION: Echec du chargement du modele")


# ===========================================
# ENDPOINTS
# ===========================================

@app.get("/", tags=["Info"])
async def root():
    """Page d'accueil de l'API."""
    return {
        "message": "API Priorisation Renouvellement Canalisations",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """
    Vérifie l'état de santé de l'API et du modèle.
    """
    model_info = predictor.get_model_info() if predictor and predictor.is_loaded else {}

    return HealthResponse(
        status="healthy" if predictor and predictor.is_loaded else "unhealthy",
        model_loaded=predictor.is_loaded if predictor else False,
        model_type=model_info.get('model_type', 'N/A'),
        horizon=model_info.get('horizon_years', 0),
        version="1.0.0"
    )


@app.get("/model/info", tags=["Info"])
async def get_model_info():
    """
    Retourne les informations détaillées sur le modèle chargé.
    """
    if not predictor or not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Modele non charge")

    info = predictor.get_model_info()
    # Convertir les numpy types en types Python natifs
    if 'performance' in info:
        perf = {}
        for k, v in info['performance'].items():
            if hasattr(v, 'item'):
                perf[k] = v.item()
            else:
                perf[k] = v
        info['performance'] = perf
    return info


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_risk(request: PredictionRequest):
    """
    Prédit le score de risque pour une liste de tronçons.

    **Paramètres:**
    - `pipes`: Liste des tronçons à scorer
    - `anomalies`: Historique des anomalies (optionnel)
    - `freeze_date`: Date de gel (format YYYY-MM-DD)
    - `horizon_years`: Horizon de prédiction (1, 3 ou 5 ans)

    **Retourne:**
    - Liste des tronçons avec leur score de risque et catégorie
    - Résumé statistique
    """
    if not predictor or not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Modele non charge")

    try:
        # Convertir les modèles Pydantic en dictionnaires
        pipes_data = [p.model_dump() for p in request.pipes]
        anomalies_data = [a.model_dump() for a in request.anomalies]

        # Prédire
        result_df, summary = predictor.predict(
            pipes=pipes_data,
            anomalies=anomalies_data,
            freeze_date=request.freeze_date,
            horizon_years=request.horizon_years
        )

        if len(result_df) == 0:
            raise HTTPException(
                status_code=400,
                detail="Aucun troncon eligible pour la prediction"
            )

        # Convertir en réponse
        predictions = [
            PipeRiskScore(
                GID=int(row['GID']),
                risk_score=float(row['risk_score']),
                rank=int(row['rank']),
                risk_category=row['risk_category']
            )
            for _, row in result_df.iterrows()
        ]

        return PredictionResponse(
            success=True,
            freeze_date=request.freeze_date,
            horizon_years=request.horizon_years,
            total_pipes=len(predictions),
            predictions=predictions,
            summary=summary
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/top-k", tags=["Prediction"])
async def predict_top_k(request: TopKRequest):
    """
    Retourne uniquement le top-k% des tronçons les plus risqués.

    **Paramètres:**
    - `top_k_percent`: Pourcentage du top (ex: 0.10 pour top 10%)

    Utile pour obtenir directement la liste de priorisation.
    """
    if not predictor or not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Modele non charge")

    try:
        pipes_data = [p.model_dump() for p in request.pipes]
        anomalies_data = [a.model_dump() for a in request.anomalies]

        result_df = predictor.get_top_k(
            pipes=pipes_data,
            anomalies=anomalies_data,
            freeze_date=request.freeze_date,
            horizon_years=request.horizon_years,
            top_k_percent=request.top_k_percent
        )

        if len(result_df) == 0:
            return {"success": False, "message": "Aucun troncon eligible"}

        return {
            "success": True,
            "freeze_date": request.freeze_date,
            "top_k_percent": request.top_k_percent,
            "n_pipes": len(result_df),
            "pipes": result_df.to_dict('records')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/single", tags=["Prediction"])
async def predict_single_pipe(
    pipe: PipeSegment,
    freeze_date: str = Query(..., description="Date de gel (YYYY-MM-DD)"),
    anomalies: List[Anomaly] = []
):
    """
    Prédit le risque pour un seul tronçon.

    Endpoint simplifié pour scorer un tronçon individuel.
    """
    if not predictor or not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Modele non charge")

    try:
        pipes_data = [pipe.model_dump()]
        anomalies_data = [a.model_dump() for a in anomalies]

        result_df, _ = predictor.predict(
            pipes=pipes_data,
            anomalies=anomalies_data,
            freeze_date=freeze_date,
            horizon_years=1
        )

        if len(result_df) == 0:
            raise HTTPException(
                status_code=400,
                detail="Troncon non eligible (verifier les dates)"
            )

        row = result_df.iloc[0]
        return {
            "GID": int(row['GID']),
            "risk_score": float(row['risk_score']),
            "risk_category": row['risk_category'],
            "interpretation": get_risk_interpretation(row['risk_score'])
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch/csv", tags=["Prediction Batch"])
async def predict_from_csv(
    pipes_file: UploadFile = File(..., description="Fichier CSV des troncons"),
    anomalies_file: UploadFile = File(None, description="Fichier CSV des anomalies (optionnel)"),
    freeze_date: str = Query(..., description="Date de gel (YYYY-MM-DD)"),
    horizon_years: int = Query(1, description="Horizon de prediction"),
    top_k_percent: Optional[float] = Query(None, description="Si specifie, retourne uniquement le top-k%")
):
    """
    Prédiction en batch depuis des fichiers CSV.

    **Format CSV tronçons attendu:**
    ```
    GID,DDP,DHS,MAT,DIAMETRE,LNG
    410966779,2009-01-01,,FT,100,82.726
    ```

    **Format CSV anomalies attendu:**
    ```
    GID_OBJET,DATE_DETECTION,TYPE_ANOMALIE
    410030412,2017-08-02,FUITE_DETECT_TR
    ```
    """
    if not predictor or not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Modele non charge")

    try:
        # Lire le CSV des tronçons
        pipes_content = await pipes_file.read()
        df_pipes = pd.read_csv(io.BytesIO(pipes_content))

        # Vérifier les colonnes requises
        required_cols = ['GID', 'DDP', 'MAT', 'DIAMETRE', 'LNG']
        missing_cols = [c for c in required_cols if c not in df_pipes.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Colonnes manquantes dans le fichier troncons: {missing_cols}"
            )

        # Lire le CSV des anomalies si fourni
        if anomalies_file:
            anomalies_content = await anomalies_file.read()
            df_anomalies = pd.read_csv(io.BytesIO(anomalies_content))
        else:
            df_anomalies = pd.DataFrame()

        # Convertir en liste de dictionnaires
        pipes_data = df_pipes.to_dict('records')
        anomalies_data = df_anomalies.to_dict('records') if len(df_anomalies) > 0 else []

        # Prédire
        if top_k_percent:
            result_df = predictor.get_top_k(
                pipes=pipes_data,
                anomalies=anomalies_data,
                freeze_date=freeze_date,
                horizon_years=horizon_years,
                top_k_percent=top_k_percent
            )
            return {
                "success": True,
                "freeze_date": freeze_date,
                "top_k_percent": top_k_percent,
                "n_input_pipes": len(df_pipes),
                "n_output_pipes": len(result_df),
                "pipes": result_df.to_dict('records')
            }
        else:
            result_df, summary = predictor.predict(
                pipes=pipes_data,
                anomalies=anomalies_data,
                freeze_date=freeze_date,
                horizon_years=horizon_years
            )
            return {
                "success": True,
                "freeze_date": freeze_date,
                "n_input_pipes": len(df_pipes),
                "n_scored_pipes": len(result_df),
                "summary": summary,
                "pipes": result_df.to_dict('records')
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================
# FONCTIONS UTILITAIRES
# ===========================================

def get_risk_interpretation(score: float) -> str:
    """Retourne une interprétation textuelle du score de risque."""
    if score >= 0.7:
        return "Risque CRITIQUE - Renouvellement prioritaire recommande"
    elif score >= 0.4:
        return "Risque ELEVE - A surveiller de pres, planifier le renouvellement"
    elif score >= 0.2:
        return "Risque MOYEN - Surveillance standard"
    else:
        return "Risque FAIBLE - Pas d'action immediate necessaire"


# ===========================================
# POINT D'ENTREE
# ===========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
