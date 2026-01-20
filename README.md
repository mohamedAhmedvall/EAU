# Priorisation des Renouvellements de Canalisations d'Eau Potable

Modèle de Machine Learning pour prédire le risque de défaillance des canalisations et prioriser les renouvellements.

## Performance du Modèle

| Métrique | Valeur |
|----------|--------|
| **Modèle** | LightGBM |
| **Horizon** | 1 an |
| **Capture@10%** | 60.1% |
| **Lift@10%** | 6.01x |
| **ROC-AUC** | 0.866 |

> En ciblant les 10% de tronçons les plus risqués, on capture 60% des défaillances futures.

## Structure du Projet

```
EAU/
├── data/                          # Données sources
│   ├── v1_trafic_prepared.csv     # Table patrimoine (218k tronçons)
│   └── historiqueanomalie.csv     # Table anomalies (31k anomalies)
├── src/                           # Code source
│   ├── config.py                  # Configuration
│   ├── data_audit.py              # Audit des données
│   ├── survival_eda.py            # Analyse Kaplan-Meier
│   ├── dataset_builder.py         # Feature engineering
│   ├── modeling.py                # Entraînement des modèles
│   ├── explainability.py          # SHAP et importances
│   ├── predict_risk.py            # Fonction de prédiction
│   └── generate_final_report.py   # Génération du rapport
├── api/                           # API REST (FastAPI)
│   ├── main.py                    # Application FastAPI
│   ├── predictor.py               # Module de prédiction
│   ├── models.py                  # Schémas Pydantic
│   └── README.md                  # Documentation API
├── artifacts/                     # Modèles et métadonnées
│   ├── best_model.joblib          # Modèle sérialisé
│   └── life_stats_h1.csv          # Stats survie par matériau
└── reports/                       # Rapports et graphiques
    ├── final_report.md
    ├── data_audit.md
    ├── survival_eda.md
    └── plots/                     # Courbes KM, SHAP, etc.
```

## Installation

```bash
# Cloner le repo
git clone https://github.com/VOTRE_USERNAME/priorisation-canalisations.git
cd priorisation-canalisations

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### 1. Prédiction en Python

```python
from src.predict_risk import predict_risk, get_prioritized_list
import pandas as pd

# Charger les données
df_assets = pd.read_csv('data/v1_trafic_prepared.csv')
df_anomalies = pd.read_csv('data/historiqueanomalie.csv')

# Prédire les risques
scores = predict_risk(
    df_assets,
    df_anomalies,
    freeze_date='2024-01-01',
    horizon_years=1
)

# Top 10% prioritaire
priorites = get_prioritized_list(
    df_assets, df_anomalies,
    freeze_date='2024-01-01',
    top_k_pct=0.10
)
print(priorites.head(20))
```

### 2. API REST

```bash
# Démarrer l'API
cd api
python run_api.py

# L'API est accessible sur http://localhost:8000
# Documentation Swagger : http://localhost:8000/docs
```

Exemple d'appel :
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pipes": [{"GID": 123, "DDP": "1980-01-01", "MAT": "FT", "DIAMETRE": 100, "LNG": 50}],
    "anomalies": [],
    "freeze_date": "2024-01-01",
    "horizon_years": 1
  }'
```

## Features Utilisées

Le modèle utilise 22 features :

| Catégorie | Features |
|-----------|----------|
| **Base** | age_at_freeze, diametre, longueur, materiau, decade_install |
| **Anomalies** | n_fuites_total, n_fuites_1y/3y/5y, days_since_last_fuite, leak_rate_per_year |
| **Survie** | ratio_age_median, overdue_years, over_p75_life, over_p90_life |
| **Interactions** | age_x_nfuites, surface_approx, age_x_ratio |

## Méthodologie

1. **Formulation temporelle** : Approche freeze + horizon pour éviter toute fuite de données
2. **Analyse de survie** : Kaplan-Meier par matériau, décennie, historique de fuites
3. **Feature engineering** : 22 features strictement calculées avant FREEZE_DATE
4. **Modélisation** : Benchmark de 5 modèles (Baseline, Logistic, Cox, HistGB, LightGBM)
5. **Évaluation** : Métriques business (Capture@k, Lift@k) prioritaires

## Résultats

### Benchmark des Modèles (Horizon 1 an)

| Modèle | ROC-AUC | Capture@5% | Capture@10% | Capture@20% |
|--------|---------|------------|-------------|-------------|
| Baseline | 0.654 | 2.0% | 3.8% | 27.0% |
| Logistic | 0.702 | 9.1% | 24.7% | 46.7% |
| Cox PH | 0.684 | 4.0% | 13.9% | 35.9% |
| HistGB | 0.861 | 43.9% | 59.3% | 73.7% |
| **LightGBM** | **0.866** | **44.4%** | **60.1%** | **76.0%** |

### Top 5 Features (par importance)

1. Longueur du tronçon
2. Âge au freeze
3. Surface approximative (diamètre × longueur)
4. Diamètre
5. Ratio âge / durée de vie médiane du matériau

## Licence

MIT License

## Auteur

Projet réalisé dans le cadre d'une alternance Data Science.
