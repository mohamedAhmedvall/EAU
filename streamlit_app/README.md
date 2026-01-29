# Optiplan - Application de planification des renouvellements

Application Streamlit pour l'optimisation des plans de renouvellement de canalisations d'eau potable.

## Installation

```bash
cd streamlit_app
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

L'application sera accessible sur http://localhost:8501

## Fonctionnalités

### 1. Scénarios
- Créer, dupliquer, supprimer des scénarios
- Filtrer par statut (Brouillon / Terminé)

### 2. Simulation
- Configurer les paramètres :
  - Horizon temporel (1-30 ans)
  - Budget annuel
  - Contraintes de linéaire (min/max %)
  - Coefficient α (risque ML vs opportunités)
- Visualiser les résultats :
  - KPIs (budget, linéaire, risque)
  - Détail par année
  - Liste des tronçons sélectionnés
- Exporter en CSV

### 3. Paramètres
- Coûts de renouvellement par matériau/diamètre
- Seuils de classement (5 classes de risque)

## Moteur d'optimisation

Le moteur utilise la programmation linéaire en nombres entiers (MILP) avec PuLP :

```
Maximiser : Σ (score_i × longueur_i × x_i)

Contraintes :
  - Budget max : Σ (coût_i × x_i) ≤ budget
  - Linéaire min : Σ (longueur_i × x_i) ≥ min_km
  - Linéaire max : Σ (longueur_i × x_i) ≤ max_km

où score_i = α × risque_ML + (1-α) × opportunité
```

## Structure

```
streamlit_app/
├── app.py                 # Point d'entrée
├── requirements.txt       # Dépendances
├── src/
│   ├── models.py          # Dataclasses
│   ├── data_loader.py     # Chargement données
│   └── optimizer.py       # Moteur MILP
├── pages/
│   ├── 1_Scenarios.py     # Liste des scénarios
│   ├── 2_Simulation.py    # Config et résultats
│   └── 3_Parametres.py    # Coûts et seuils
└── data/
    └── scenarios/         # JSON sauvegardés
```
