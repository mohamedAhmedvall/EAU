# API REST - Priorisation des Renouvellements de Canalisations

## Description

API REST permettant de scorer les tronçons de canalisation d'eau potable selon leur risque de défaillance, afin de prioriser les renouvellements.

## Installation

```bash
cd api
pip install -r requirements.txt
```

## Démarrage

```bash
cd api
python run_api.py
```

L'API sera accessible sur `http://localhost:8000`

## Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints

### 1. Health Check
```
GET /health
```
Vérifie l'état de l'API et du modèle.

### 2. Informations du modèle
```
GET /model/info
```
Retourne les détails du modèle (type, features, performance).

### 3. Prédiction batch
```
POST /predict
```
Prédit le risque pour une liste de tronçons.

**Corps de la requête:**
```json
{
  "pipes": [
    {
      "GID": 410966779,
      "DDP": "2009-01-01",
      "DHS": null,
      "MAT": "FT",
      "DIAMETRE": 100,
      "LNG": 82.7
    }
  ],
  "anomalies": [
    {
      "GID_OBJET": 410966779,
      "DATE_DETECTION": "2020-05-15",
      "TYPE_ANOMALIE": "FUITE_SIGNAL_TR"
    }
  ],
  "freeze_date": "2024-01-01",
  "horizon_years": 1
}
```

**Réponse:**
```json
{
  "success": true,
  "freeze_date": "2024-01-01",
  "horizon_years": 1,
  "total_pipes": 1,
  "predictions": [
    {
      "GID": 410966779,
      "risk_score": 0.45,
      "rank": 1,
      "risk_category": "ELEVE"
    }
  ],
  "summary": {
    "n_critique": 0,
    "n_eleve": 1,
    "n_moyen": 0,
    "n_faible": 0
  }
}
```

### 4. Top-K prioritaire
```
POST /predict/top-k
```
Retourne uniquement le top-k% des tronçons les plus risqués.

### 5. Prédiction unitaire
```
POST /predict/single?freeze_date=2024-01-01
```
Score un seul tronçon.

### 6. Prédiction depuis CSV
```
POST /predict/batch/csv
```
Upload de fichiers CSV pour prédiction en masse.

## Catégories de risque

| Score | Catégorie | Recommandation |
|-------|-----------|----------------|
| ≥ 0.7 | CRITIQUE | Renouvellement prioritaire |
| 0.4 - 0.7 | ELEVE | Planifier le renouvellement |
| 0.2 - 0.4 | MOYEN | Surveillance standard |
| < 0.2 | FAIBLE | Pas d'action immédiate |

## Exemple avec curl

```bash
# Health check
curl http://localhost:8000/health

# Prédiction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pipes": [{"GID": 123, "DDP": "1980-01-01", "DHS": null, "MAT": "FT", "DIAMETRE": 100, "LNG": 50}],
    "anomalies": [],
    "freeze_date": "2024-01-01",
    "horizon_years": 1
  }'
```

## Exemple Python

```python
import requests

# Prédiction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "pipes": [
            {"GID": 123, "DDP": "1980-01-01", "DHS": None, "MAT": "FT", "DIAMETRE": 100, "LNG": 50}
        ],
        "anomalies": [],
        "freeze_date": "2024-01-01",
        "horizon_years": 1
    }
)

result = response.json()
for pred in result["predictions"]:
    print(f"GID {pred['GID']}: {pred['risk_category']} (score: {pred['risk_score']:.2f})")
```

## Performance du modèle

- **Type**: LightGBM
- **Capture@10%**: 60% (en ciblant 10% du réseau, on capture 60% des défaillances)
- **Lift**: 6x (6 fois mieux qu'une sélection aléatoire)

## Test

```bash
# Démarrer l'API dans un terminal
python run_api.py

# Dans un autre terminal, lancer les tests
python test_api.py
```
