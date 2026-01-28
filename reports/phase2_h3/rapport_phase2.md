# Rapport PHASE 2 — Amelioration du modele de renouvellement

*Genere automatiquement (horizon: 3 an(s))*

---

## 1. Rappel des constats PHASE 1

| Constat | Impact |
|---------|--------|
| Survivorship bias ages extremes (1900-1920) | FP sur vieux tuyaux, FN sur jeunes |
| Calibration mediocre (ECE=0.45) | Scores non utilisables en optimisation |
| Segments PEHD/PVC sous-couverts | 0% capture sur materiaux recents |
| Abandons preventifs polluent le signal | Bruit dans le label y=1 |

## 2. Iteration 1 — Corrections biais (M1)

### A) Ages extremes

**Strategies testees:**
1. **Conservative (M1a)**: capping age a 110 ans + flag age extreme
2. **Structurelle (M1b)**: ratio age/decennie + residual age/materiau + winsorisation P99

### B) Abandons preventifs

**Definition operationnelle:**
- event=1 ET zero anomalie historique ET ratio_age_median < 0.6 ET age < 15 ans ET pas de fuite recente
- Detectes: 445 / 5435 evenements (8.2%)

**Approches testees:**
- A) Filtrage label (event -> 0 pour abandons)
- B) Competing risks (multi-class: 0=survie, 1=defaillance, 2=abandon)

### Resultats Iteration 1

| Modele | Lift@10% | Cap@10% | ROC-AUC | FN | Decision |
|--------|----------|---------|---------|----| ---------|
| M1a | 5.07 | 50.69% | 0.8117 | 536 | rejete |
| M1b | 4.98 | 49.77% | 0.8134 | 546 | rejete |
| M1c | 5.39 | 53.91% | 0.8836 | 501 | rejete |

## 3. Iteration 2 — Calibration & Stabilite (M2)

### Contraintes monotones

- age, ratio_age_median, overdue_years: monotone croissant (+1)
- days_since_last_fuite: monotone decroissant (-1)
- Regularisation: lambda=2.0, alpha=0.3, min_child=50

### Calibration

| Variante | Lift@10% | Brier | ECE |
|----------|----------|-------|-----|
| M2b_raw | 5.33 | 0.1583 | 0.2957 |
| M2b_iso | 5.32 | 0.0274 | 0.0009 |
| M2b_platt | 5.33 | 0.0275 | 0.0016 |

**Calibration retenue**: M2b_raw

### Definition risque_attendu

```
risque_attendu = proba_calibree × longueur_km
```
Ou proba_calibree est la sortie du modele apres calibration isotonique/Platt.

## 4. Iteration 3 — Comparaison finale

| Model | ROC_AUC | Lift_5 | Lift_10 | Lift_20 | Cap_5 | Cap_10 | Cap_20 | Brier | ECE | FN | FP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0_baseline | 0.8641 | 7.5805 | 5.4738 | 3.7397 | 0.3790 | 0.5474 | 0.7479 | 0.1467 | 0.2722 | 492 | 2842 |
| M0_baseline | 0.8641 | 7.5805 | 5.4738 | 3.7397 | 0.3790 | 0.5474 | 0.7479 | 0.1467 | 0.2722 | 492 | 2842 |
| M2b_raw | 0.8562 | 7.2677 | 5.3266 | 3.6477 | 0.3634 | 0.5327 | 0.7295 | 0.1583 | 0.2957 | 508 | 2858 |

## 5. Stabilite par segments

### Par materiau (M*)

| segment | n | events | capture | fn_rate | avg_score |
| --- | --- | --- | --- | --- | --- |
| A.C | 66 | 3 | 1.0000 | 0.0000 | 0.2240 |
| ACIE | 288 | 13 | 0.2308 | 0.7692 | 0.3328 |
| AUTRE | 1176 | 2 | 0.0000 | 1.0000 | 0.0209 |
| BTM | 435 | 10 | 0.6000 | 0.4000 | 0.2314 |
| FT | 18558 | 549 | 0.5556 | 0.4444 | 0.2887 |
| FTG | 8767 | 429 | 0.5921 | 0.4079 | 0.4490 |
| FTVI | 1633 | 14 | 0.2857 | 0.7143 | 0.1094 |
| PEHD | 1293 | 14 | 0.1429 | 0.8571 | 0.1019 |
| POLY | 1246 | 41 | 0.3902 | 0.6098 | 0.3680 |
| PVC | 734 | 10 | 0.2000 | 0.8000 | 0.1632 |

### Par decennie (M*)

| segment | n | events | capture | fn_rate | avg_score |
| --- | --- | --- | --- | --- | --- |
| 1900 | 3258 | 22 | 0.2727 | 0.7273 | 0.0950 |
| 1930 | 146 | 5 | 0.8000 | 0.2000 | 0.3518 |
| 1940 | 2017 | 179 | 0.6425 | 0.3575 | 0.6832 |
| 1950 | 1522 | 114 | 0.7632 | 0.2368 | 0.5438 |
| 1960 | 4584 | 275 | 0.5818 | 0.4182 | 0.5423 |
| 1970 | 5209 | 205 | 0.5610 | 0.4390 | 0.3859 |
| 1980 | 3940 | 66 | 0.4697 | 0.5303 | 0.1893 |
| 1990 | 2335 | 49 | 0.4694 | 0.5306 | 0.2198 |
| 2000 | 3784 | 75 | 0.3333 | 0.6667 | 0.2442 |
| 2010 | 5867 | 77 | 0.2468 | 0.7532 | 0.1721 |
| 2020 | 1696 | 19 | 0.4737 | 0.5263 | 0.1054 |

## 6. Recommandation finale

### Modele retenu: **M0_baseline**

- **Lift@10%**: 5.47
- **Capture@10%**: 54.74%
- **Brier Score**: 0.1467
- **ECE**: 0.2722
- **FN@10%**: 492
- **FP@10%**: 2842

### Hypotheses metier restantes

1. L'heuristique d'abandon preventif est une approximation — idealement confronter avec les motifs DHS du SIG.
2. Les segments PEHD/PVC/FTVI restent sous-represents — enrichir les donnees si possible.
3. Le modele ne capture pas les defaillances dues a des facteurs externes (travaux tiers, mouvements terrain).

### Pret pour moteur d'optimisation?

**OUI** sous reserve de:
- Utiliser `risque_attendu = proba_calibree * longueur_km` comme valeur
- Monitorer les segments faibles trimestriellement
- Recalibrer annuellement avec nouvelles donnees

## 7. Artefacts generes

| Fichier | Description |
|---------|-------------|
| `model_M0_h3.joblib` | Baseline LightGBM |
| `model_M1_best_h3.joblib` | Meilleur M1 (corrections biais) |
| `model_M2_best_h3.joblib` | M2 avec calibration |
| `model_M_star_h3.joblib` | Modele final retenu |
| `metadata_phase2_h3.json` | Metadata complete |
| `comparison_M0_M1_M2.csv` | Tableau comparatif |
| `plots/` | Graphiques AVANT/APRES |
