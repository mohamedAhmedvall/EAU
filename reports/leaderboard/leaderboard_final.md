# LEADERBOARD FINAL - Modeles AEP Renouvellement

*Date: 2026-01-28*
*Horizon: 1 an*

---

## CLASSEMENT GLOBAL

| Rang | Modele | Lift@10% | Cap@10% | Brier | ECE | FN | Statut |
|------|--------|----------|---------|-------|-----|-----|--------|
| **1** | **M4_competing_iso** | **6.21** | **62.1%** | **0.010** | **0.005** | **150** | **RETENU** |
| 2 | M1_competing | 6.16 | 61.6% | 0.010 | 0.001 | 152 | Candidat |
| 3 | M0_baseline | 5.88 | 58.8% | 0.117 | 0.218 | 163 | Reference |
| 4 | M2_mono_clean | 4.92 | 49.2% | 0.134 | 0.228 | 201 | Rejete |
| 5 | M5_mono_iso | 4.87 | 48.7% | 0.011 | 0.002 | 203 | Rejete |
| 6 | M3_mono_reg | 4.72 | 47.2% | 0.150 | 0.258 | 209 | Rejete |

---

## DETAIL DES MODELES

### M4_competing_iso (RETENU)

**Description**: Multi-class competing risks + calibration isotonique post-hoc

**Configuration**:
- Objectif: multiclass (3 classes: survie, defaillance reelle, abandon preventif)
- Features: 27 (base + corrections biais)
- Calibration: Isotonique sur P(classe=1)

**Forces**:
- Meilleur Lift@10% (+5.6% vs baseline)
- Excellente calibration (Brier 0.010, ECE 0.005)
- Reduction FN (-8% vs baseline)
- Separation propre defaillances vs abandons

**Faiblesses**:
- Complexite accrue (modele multiclass + calibrateur)
- ROC-AUC legerement inferieur au baseline (0.849 vs 0.864)

---

### M1_competing

**Description**: Multi-class competing risks sans calibration

**Configuration**:
- Objectif: multiclass (3 classes)
- Features: 27 etendues
- Pas de post-processing

**Resultats**:
- Lift@10%: 6.16
- Tres bonne calibration native (Brier 0.010, ECE 0.001)

**Decision**: Proche de M4, mais M4 a meilleur Lift

---

### M0_baseline

**Description**: LightGBM binaire avec class_weight

**Configuration**:
- Objectif: binary
- Features: 22 base
- scale_pos_weight pour desequilibre

**Resultats**:
- Reference de comparaison
- Bonne performance ranking
- Calibration mediocre

**Decision**: Depassee par M4/M1

---

### M2_mono_clean et M3_mono_reg

**Description**: Labels nettoyes + contraintes monotones

**Configuration**:
- event_clean: abandons preventifs mis a 0
- Contraintes monotones sur age/anomalies
- Regularisation pour M3

**Resultats**:
- Degradation significative du Lift
- Pas d'amelioration calibration

**Decision**: REJETES - le nettoyage des labels degrade les performances
> Hypothese: les "abandons preventifs" contiennent du signal utile

---

### M5_mono_iso

**Description**: M2 + calibration isotonique

**Configuration**:
- Base M2 (cleaned labels + monotone)
- Calibration isotonique post-hoc

**Resultats**:
- Calibration amelioree vs M2
- Lift toujours degrade

**Decision**: REJETE - calibration ne compense pas la perte de ranking

---

## COMPARAISON PAR OBJECTIF

### Objectif 1: Ranking (Lift@10%)

| Modele | Lift@10% | Delta vs baseline |
|--------|----------|-------------------|
| M4_competing_iso | 6.21 | +5.6% |
| M1_competing | 6.16 | +4.7% |
| M0_baseline | 5.88 | reference |
| M2_mono_clean | 4.92 | -16.3% |

**Conclusion**: M4 gagne sur le ranking

### Objectif 2: Calibration (Brier + ECE)

| Modele | Brier | ECE | Pret optimisation? |
|--------|-------|-----|-------------------|
| M4_competing_iso | 0.010 | 0.005 | **OUI** |
| M1_competing | 0.010 | 0.001 | OUI |
| M5_mono_iso | 0.011 | 0.002 | OUI |
| M0_baseline | 0.117 | 0.218 | NON |

**Conclusion**: M4, M1, M5 tous bien calibres. M0 inutilisable pour optimisation.

### Objectif 3: Securite (FN)

| Modele | FN | Delta vs baseline |
|--------|-----|-------------------|
| M4_competing_iso | 150 | -8% |
| M1_competing | 152 | -7% |
| M0_baseline | 163 | reference |
| M3_mono_reg | 209 | +28% |

**Conclusion**: M4 le plus sur (moins de defaillances manquees)

---

## DECISION FINALE

### Modele retenu: M4_competing_iso

**Justification multi-criteres**:

1. **Primary (Lift@10%)**: +5.6% vs baseline - ACCEPTE
2. **Secondary (Calibration)**: Brier 0.010, ECE 0.005 - EXCELLENT
3. **Secondary (Stabilite)**: ROC-AUC 0.849 stable en CV
4. **Secondary (Securite)**: FN -8% - ACCEPTE
5. **Hard constraint (Leakage)**: 4/4 tests passes - OK

### Usage recommande

```python
# Scoring production
risque_attendu = proba_calibree * longueur_km

# Categorie de risque
if proba >= 0.5: categorie = "CRITIQUE"
elif proba >= 0.2: categorie = "ELEVE"
elif proba >= 0.1: categorie = "MOYEN"
else: categorie = "FAIBLE"
```

### Monitoring

- Recalibrer tous les 6 mois
- Surveiller capture sur PEHD/PVC/FTVI
- Alerter si ECE > 0.05

---

## ARTEFACTS

| Fichier | Description |
|---------|-------------|
| `artifacts/best_model_h1_v2.joblib` | Modele final + calibrateur |
| `artifacts/metadata_h1_v2.json` | Metadata et metriques |
| `reports/phase3_h1/model_comparison.csv` | Tableau comparatif |
| `reports/phase3_h1/plots/` | Graphiques |
