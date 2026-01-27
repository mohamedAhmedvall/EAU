# Plan de Renouvellement AEP - Application Streamlit

Application de planification du renouvellement du reseau d'eau potable basee sur le scoring ML.

## Pre-requis

- Python 3.9+
- Modele entraine dans `artifacts/best_model.joblib`
- Donnees dans `data/v1_trafic_prepared.csv` et `data/historiqueanomalie.csv`

## Installation

```bash
cd /home/user/EAU
pip install -r streamlit_app/requirements.txt
```

Si OR-Tools n'est pas disponible, le solveur greedy sera utilise automatiquement.

## Lancement

```bash
cd /home/user/EAU
streamlit run streamlit_app/app.py
```

L'application sera accessible sur `http://localhost:8501`.

## Configuration

### Chemins des donnees

Les chemins par defaut sont :
- Patrimoine : `data/v1_trafic_prepared.csv`
- Anomalies : `data/historiqueanomalie.csv`
- Modele : `artifacts/best_model.joblib`
- Life stats : `artifacts/life_stats_h{horizon}.csv`

Ces chemins sont relatifs au repertoire racine du projet (`/home/user/EAU`).

### Mode developpement (sample)

Pour accelerer le chargement en dev, modifier `data_loader.py` et passer `sample_n=50000`
dans `load_patrimoine()`.

## Pages de l'application

### 1. Scenario
- Definir freeze_date, horizon, budget, contraintes
- Choisir entre baseline ML (top-K) et plan optimise (solveur)
- Generer le scenario

### 2. Carte & Troncons
- Carte des troncons colores par risque (si geometrie disponible)
- Tableau filtrable avec tous les troncons
- Detail d'un troncon (features principales)

### 3. Plan optimise
- Tableau des chantiers (troncons selectionnes)
- Verification des contraintes
- Comparaison baseline vs optimise

### 4. Bilan Risque
- KPI cards : risque total / traite / non traite
- Courbe de capture @K
- Analyse par decile
- Distribution par materiau et age

### 5. Export
- CSV du plan (troncons selectionnes)
- CSV complet (tous les scores)
- GeoJSON du plan
- Rapport scenario (Markdown)

## Exemples de scenarios

### Scenario 1 : Baseline annuel
- freeze_date : derniere date observee
- horizon : 1 an
- mode : Baseline ML
- top-K : 10%
- budget : 10 M EUR

### Scenario 2 : Plan optimise sous contraintes
- freeze_date : derniere date observee
- horizon : 1 an
- mode : Plan optimise (solveur)
- budget : 5 M EUR
- lineaire max : 30 km
- cout/km : 200 000 EUR
- exclusion : materiaux PEHD (trop recent)

## Tests

```bash
cd /home/user/EAU
pytest streamlit_app/tests/ -v
```

### Tests disponibles

- `test_no_leakage.py` : verification anti-fuite de donnees temporelles
- `test_scenario_kpis.py` : coherence des KPIs (risk_total = treated + untreated, etc.)

## Architecture

```
streamlit_app/
├── app.py                          # Application principale
├── src/
│   ├── data_loader.py              # Chargement + cache
│   ├── scoring_service.py          # Wrapper predict_risk
│   ├── scenario.py                 # ScenarioParams + ScenarioResult
│   ├── optimization/
│   │   ├── ortools_solver.py       # Solveur MILP (OR-Tools)
│   │   └── greedy_solver.py        # Solveur greedy (fallback)
│   ├── viz/
│   │   ├── map_view.py             # Carte Folium / tableau
│   │   └── charts.py               # Graphiques Plotly
│   └── exports.py                  # CSV / GeoJSON / rapport
├── tests/
│   ├── test_no_leakage.py
│   └── test_scenario_kpis.py
├── requirements.txt
├── README_RUN.md
└── SCHEMA.md
```

## Performance

- Le chargement des donnees est cache (st.cache_data)
- Le modele est cache (st.cache_resource)
- Le scoring de 200k lignes prend ~10-20s
- Le solveur OR-Tools a un timeout de 60s par defaut
- Le solveur greedy est quasi-instantane
