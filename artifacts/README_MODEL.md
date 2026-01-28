# Modele Final AEP - M4_competing_iso

*Version: v2 - Date: 2026-01-28*

---

## Resume

Ce modele predit le risque de defaillance des canalisations d'eau potable
sur un horizon de 1 an.

**Approche**: Competing Risks (LightGBM multiclass) + Calibration Isotonique

---

## Performance

| Metrique | Valeur |
|----------|--------|
| Lift@10% | 6.21 |
| Capture@10% | 62.1% |
| ROC-AUC | 0.849 |
| Brier Score | 0.010 |
| ECE | 0.005 |
| FN@10% | 150 |

**Amelioration vs baseline**: +5.6% Lift@10%, -91% Brier, -8% FN

---

## Fichiers

| Fichier | Description |
|---------|-------------|
| `best_model_h1_v2.joblib` | Modele complet (model + calibrator + encoders) |
| `metadata_h1_v2.json` | Metadata et metriques |

---

## Usage

```python
import joblib
import pandas as pd
import numpy as np

# Charger le modele
artifact = joblib.load("artifacts/best_model_h1_v2.joblib")
model = artifact["model"]
calibrator = artifact["calibrator"]
encoders = artifact["encoders"]

# Preparer les features (27 features requises)
# Voir metadata pour la liste complete

# Predire
proba_raw = model.predict_proba(X)[:, 1]  # P(defaillance reelle)
proba_calibree = calibrator.transform(proba_raw)

# Score pour optimisation
risque_attendu = proba_calibree * longueur_km
```

---

## Features (27)

### Base (22)
- age_at_freeze, diametre, longueur, log_longueur, log_diametre
- n_fuites_total, n_fuites_1y, n_fuites_3y, n_fuites_5y
- days_since_last_fuite, has_recent_fuite
- leak_rate_per_year, leak_rate_per_km
- ratio_age_median, overdue_years, over_p75_life, over_p90_life
- age_x_nfuites, surface_approx, age_x_ratio
- materiau_encoded, decade_install_encoded

### Etendues (5)
- age_cap, age_extreme_flag, age_winsor
- age_ratio_decade, age_residual_mat

---

## Limitations

1. **Segments faibles**: PEHD, PVC, FTVI ont 0-33% capture
2. **Ages extremes**: Tuyaux 1900-1920 moins bien couverts
3. **Abandons**: Heuristique approximative
4. **Facteurs externes**: Travaux tiers non captures

---

## Monitoring

- Recalibrer tous les 6 mois
- Surveiller ECE (alerte si > 0.05)
- Suivre capture par segment trimestriellement

---

## Anti-Leakage

Le modele a passe 4/4 tests anti-leakage:
- Aucune anomalie post-freeze dans les features
- Aucun DHS pre-freeze dans le dataset
- Coherence temporelle des features
- Target dans l'horizon futur uniquement
