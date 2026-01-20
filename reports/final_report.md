# Rapport Final - Modele de Priorisation des Renouvellements de Canalisations

*Genere le: 2026-01-19 19:49:09*


## 1. Resume Executif

### Modele Retenu
- **Horizon de prediction**: 1 an(s)
- **Type de modele**: LightGBM
- **Performance (Lift@10%)**: 6.01x
- **Capture@10%**: 60.1%
- **Capture@5%**: 44.4%
- **ROC-AUC**: 0.866

### Interpretation Business
En selectionnant les **10% de troncons les plus risques** selon le modele, on capture **60.1%** des defaillances reelles.
C'est un lift de **6.0x** par rapport a une selection aleatoire.

## 2. Audit des Donnees

Voir le rapport detaille: [data_audit.md](data_audit.md)

### Points Cles
- **Date maximale d'observation**: 2024-11-07
- **Table Patrimoine**: ~218,000 troncons
- **Table Anomalies**: ~31,000 anomalies
- **Taux de troncons avec anomalies**: 5.9%
- **Taux de troncons defaillants (DHS renseigne)**: 17.7%

### Decisions de Nettoyage
1. Exclusion des dates futures (implausibles)
2. Exclusion des troncons avec DDP > DHS (incoherent)
3. Anomalies orphelines ignorees (1,513 GID sans correspondance)

## 3. Analyse Descriptive de Survie

Voir le rapport detaille: [survival_eda.md](survival_eda.md)

### Observations Principales
- La duree de vie mediane varie significativement selon le materiau
- Les troncons avec historique d'anomalies ont une survie plus courte
- Biais de survivant identifie sur les troncons tres anciens

## 4. Feature Engineering

### Features Utilisees (22 au total)

**Features de base:**
- `age_at_freeze`: Age du troncon a la date de gel (annees)
- `diametre`, `longueur`: Caracteristiques physiques
- `materiau`: Type de materiau (FT, FTG, PEHD, etc.)
- `decade_install`: Decennie d'installation

**Features d'historique d'anomalies:**
- `n_fuites_total`: Nombre total de fuites avant freeze
- `n_fuites_1y`, `n_fuites_3y`, `n_fuites_5y`: Fuites sur fenetres glissantes
- `days_since_last_fuite`: Jours depuis la derniere fuite
- `has_recent_fuite`: Indicateur de fuite recente (<3 ans)
- `leak_rate_per_year`, `leak_rate_per_km`: Taux de fuite

**Features informees par la survie:**
- `ratio_age_median`: Age / duree de vie mediane du materiau
- `overdue_years`: Annees de depassement de la duree mediane
- `over_p75_life`, `over_p90_life`: Indicateurs de depassement de percentiles

**Features d'interaction:**
- `age_x_nfuites`: Age * nombre de fuites
- `surface_approx`: Diametre * longueur

### Protocole Anti-Fuite
**CRITIQUE**: Toutes les features sont calculees strictement avec des donnees <= FREEZE_DATE
- Assertions automatisees dans le code pour chaque groupe de features
- Validation finale avant chaque entrainement

## 5. Benchmark des Modeles

### Tableau Comparatif Complet
| Horizon | Modele | ROC-AUC | Capture@5% | Lift@5% | Capture@10% | Lift@10% | Capture@20% | Lift@20% |
|---------|--------|---------|------------|---------|-------------|----------|-------------|----------|
| 1 an(s) | Baseline (Rule-based) | 0.654 | 2.0% | 0.40 | 3.8% | 0.38 | 27.0% | 1.35 |
| 1 an(s) | Logistic Regression | 0.702 | 9.1% | 1.82 | 24.7% | 2.47 | 46.7% | 2.34 |
| 1 an(s) | Cox Proportional Hazards | 0.684 | 4.0% | 0.81 | 13.9% | 1.39 | 35.9% | 1.79 |
| 1 an(s) | HistGradientBoosting | 0.861 | 43.9% | 8.79 | 59.3% | 5.93 | 73.7% | 3.69 |
| 1 an(s) | LightGBM | 0.866 | 44.4% | 8.89 | 60.1% | 6.01 | 76.0% | 3.80 |
| 3 an(s) | Baseline (Rule-based) | 0.629 | 1.4% | 0.28 | 2.9% | 0.29 | 26.6% | 1.33 |
| 3 an(s) | Logistic Regression | 0.701 | 14.5% | 2.91 | 22.5% | 2.25 | 39.7% | 1.98 |
| 3 an(s) | Cox Proportional Hazards | 0.694 | 11.4% | 2.28 | 24.3% | 2.43 | 38.0% | 1.90 |
| 3 an(s) | HistGradientBoosting | 0.859 | 37.4% | 7.49 | 53.4% | 5.34 | 73.3% | 3.67 |
| 3 an(s) | LightGBM | 0.864 | 38.5% | 7.71 | 55.4% | 5.54 | 74.4% | 3.72 |
| 5 an(s) | Baseline (Rule-based) | 0.628 | 0.9% | 0.18 | 2.4% | 0.24 | 25.5% | 1.27 |
| 5 an(s) | Logistic Regression | 0.701 | 12.9% | 2.58 | 25.8% | 2.58 | 42.7% | 2.14 |
| 5 an(s) | Cox Proportional Hazards | 0.692 | 11.4% | 2.28 | 22.1% | 2.21 | 37.1% | 1.85 |
| 5 an(s) | HistGradientBoosting | 0.840 | 31.9% | 6.38 | 46.7% | 4.67 | 66.4% | 3.32 |
| 5 an(s) | LightGBM | 0.842 | 32.1% | 6.43 | 47.7% | 4.77 | 66.3% | 3.31 |

### Analyse par Horizon

**Horizon 1 an(s):**
- Meilleur modele: LightGBM
- Lift@10%: 6.01

**Horizon 3 an(s):**
- Meilleur modele: LightGBM
- Lift@10%: 5.54

**Horizon 5 an(s):**
- Meilleur modele: LightGBM
- Lift@10%: 4.77

### Choix Final
Le modele **LightGBM** avec un horizon de **1 an(s)** est retenu car:
1. Meilleur Lift@10% parmi toutes les configurations
2. Horizon court (1 an) adapte a une priorisation annuelle
3. Bon equilibre entre precision et interpretabilite

## 6. Explicabilite

Voir le rapport detaille: [explainability.md](explainability.md)

### Top 5 Features (par importance)
| Rang | Feature | Importance Relative |
|------|---------|---------------------|
| 3 | longueur | 100.0% |
| 1 | age_at_freeze | 70.3% |
| 19 | surface_approx | 65.9% |
| 2 | diametre | 65.3% |
| 14 | ratio_age_median | 55.4% |
| 20 | age_x_ratio | 35.4% |
| 15 | overdue_years | 20.9% |
| 13 | leak_rate_per_km | 14.6% |
| 21 | materiau_encoded | 14.3% |
| 10 | days_since_last_fuite | 14.2% |

### Interpretation Metier
- **Longueur**: Les troncons plus longs sont plus exposes aux defaillances
- **Age**: L'age est un facteur majeur de risque
- **Historique de fuites**: Fort signal predictif (les fuites appellent les fuites)
- **Ratio age/mediane**: Capture bien le vieillissement relatif au materiau

## 7. Validation Anti-Fuite

### Tests Automatises
1. **Anomalies**: Verification que DATE_DETECTION <= FREEZE_DATE
2. **DHS**: Aucun DHS futur utilise pour les stats de survie
3. **Coherence**: Tous les troncons dans le dataset sont actifs a FREEZE_DATE

### Resultats
[OK] Tous les tests anti-fuite passent pour les 3 horizons

## 8. Utilisation en Production

### Fichiers Fournis
```
artifacts/
  best_model.joblib       # Modele serialise
  model_h1_metadata.json  # Metadonnees
  life_stats_h1.csv       # Stats de survie par materiau
src/
  predict_risk.py         # Module de prediction
```

### Exemple d'Utilisation
```python
from predict_risk import predict_risk, get_prioritized_list

# Charger vos donnees
df_assets = pd.read_csv('patrimoine.csv')
df_anomalies = pd.read_csv('anomalies.csv')

# Predire les risques
scores = predict_risk(
    df_assets,
    df_anomalies,
    freeze_date='2024-01-01',
    horizon_years=1
)

# Obtenir le top 10% prioritaire
priorites = get_prioritized_list(
    df_assets, df_anomalies,
    freeze_date='2024-01-01',
    top_k_pct=0.10
)
```

## 9. Limitations et Plan de Monitoring

### Limitations Connues
1. **Biais de survivant**: Les troncons tres anciens sont des survivants; prudence sur les predictions
2. **Qualite des donnees**: Dependance a la completude de l'historique d'anomalies
3. **Drift potentiel**: Les caracteristiques des nouveaux troncons peuvent evoluer
4. **Causalite vs correlation**: Le modele identifie des correlations, pas des causes

### Plan de Monitoring
1. **Recalibration annuelle**: Re-entrainer le modele avec les nouvelles defaillances
2. **Suivi des performances**: Comparer les predictions vs defaillances reelles
3. **Detection de drift**: Surveiller la distribution des scores et features
4. **Feedback terrain**: Integrer les retours des equipes operationnelles

### Metriques de Suivi Recommandees
- Capture@10% sur les 12 mois suivant la prediction
- Taux de faux positifs dans le top 10%
- Evolution de la distribution des scores

## 10. Conclusion

Le modele developpe permet de **multiplier par 6.0** l'efficacite de selection des troncons a renouveler.
En ciblant les **10% les plus risques**, on capture **60%** des defaillances futures.

**Recommandations:**
1. Utiliser le modele pour la priorisation annuelle des renouvellements
2. Combiner le score de risque avec d'autres criteres (accessibilite, cout, criticite)
3. Re-entrainer annuellement avec les nouvelles donnees
4. Valider les predictions sur un echantillon avant deploiement complet

---
*Fin du rapport*