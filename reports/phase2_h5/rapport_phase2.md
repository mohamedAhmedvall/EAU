# Rapport PHASE 2 — Amelioration du modele de renouvellement

*Genere automatiquement (horizon: 5 an(s))*

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
- Detectes: 628 / 8481 evenements (7.4%)

**Approches testees:**
- A) Filtrage label (event -> 0 pour abandons)
- B) Competing risks (multi-class: 0=survie, 1=defaillance, 2=abandon)

### Resultats Iteration 1

| Modele | Lift@10% | Cap@10% | ROC-AUC | FN | Decision |
|--------|----------|---------|---------|----| ---------|
| M1a | 4.72 | 47.18% | 0.8053 | 899 | rejete |
| M1b | 4.74 | 47.41% | 0.8075 | 895 | rejete |
| M1c | 4.92 | 49.17% | 0.8747 | 862 | RETENU |

## 3. Iteration 2 — Calibration & Stabilite (M2)

### Contraintes monotones

- age, ratio_age_median, overdue_years: monotone croissant (+1)
- days_since_last_fuite: monotone decroissant (-1)
- Regularisation: lambda=2.0, alpha=0.3, min_child=50

### Calibration

| Variante | Lift@10% | Brier | ECE |
|----------|----------|-------|-----|
| M2b_raw | 4.57 | 0.1653 | 0.2655 |
| M2b_iso | 4.57 | 0.0431 | 0.0080 |
| M2b_platt | 4.57 | 0.0433 | 0.0105 |

**Calibration retenue**: M2b_raw

### Definition risque_attendu

```
risque_attendu = proba_calibree × longueur_km
```
Ou proba_calibree est la sortie du modele apres calibration isotonique/Platt.

## 4. Iteration 3 — Comparaison finale

| Model | ROC_AUC | Lift_5 | Lift_10 | Lift_20 | Cap_5 | Cap_10 | Cap_20 | Brier | ECE | FN | FP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0_baseline | 0.8417 | 6.3561 | 4.7052 | 3.3196 | 0.3178 | 0.4705 | 0.6639 | 0.1570 | 0.2737 | 898 | 2525 |
| M1c_competing | 0.8747 | 6.6274 | 4.9175 | 3.3962 | 0.3314 | 0.4917 | 0.6792 | 0.0418 | 0.0045 | 862 | 2489 |
| M2b_raw | 0.8007 | 6.0752 | 4.5711 | 3.1992 | 0.3038 | 0.4571 | 0.6398 | 0.1653 | 0.2655 | 924 | 2545 |

## 5. Stabilite par segments

### Par materiau (M*)

| segment | n | events | capture | fn_rate | avg_score |
| --- | --- | --- | --- | --- | --- |
| A.C | 79 | 5 | 1.0000 | 0.0000 | 0.0546 |
| ACIE | 286 | 14 | 0.4286 | 0.5714 | 0.0568 |
| BTM | 418 | 23 | 0.4783 | 0.5217 | 0.0470 |
| FT | 17633 | 758 | 0.3945 | 0.6055 | 0.0371 |
| FTBLU | 44 | 1 | 0.0000 | 1.0000 | 0.0000 |
| FTG | 9202 | 776 | 0.5979 | 0.4021 | 0.0813 |
| FTVI | 1128 | 10 | 0.1000 | 0.9000 | 0.0097 |
| PEHD | 1178 | 16 | 0.2500 | 0.7500 | 0.0112 |
| POLY | 1219 | 70 | 0.4429 | 0.5571 | 0.0599 |
| PVC | 781 | 19 | 0.6316 | 0.3684 | 0.0233 |

### Par decennie (M*)

| segment | n | events | capture | fn_rate | avg_score |
| --- | --- | --- | --- | --- | --- |
| 1900 | 3316 | 51 | 0.5882 | 0.4118 | 0.0139 |
| 1930 | 146 | 15 | 0.9333 | 0.0667 | 0.1254 |
| 1940 | 2130 | 299 | 0.6020 | 0.3980 | 0.1275 |
| 1950 | 1581 | 220 | 0.8364 | 0.1636 | 0.1327 |
| 1960 | 4791 | 418 | 0.4330 | 0.5670 | 0.0910 |
| 1970 | 5366 | 328 | 0.4634 | 0.5366 | 0.0611 |
| 1980 | 3956 | 81 | 0.4074 | 0.5926 | 0.0238 |
| 1990 | 2358 | 68 | 0.4265 | 0.5735 | 0.0283 |
| 2000 | 3887 | 95 | 0.2211 | 0.7789 | 0.0195 |
| 2010 | 5683 | 115 | 0.0348 | 0.9652 | 0.0025 |

## 6. Recommandation finale

### Modele retenu: **M1c_competing**

- **Lift@10%**: 4.92
- **Capture@10%**: 49.17%
- **Brier Score**: 0.0418
- **ECE**: 0.0045
- **FN@10%**: 862
- **FP@10%**: 2489

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
| `model_M0_h5.joblib` | Baseline LightGBM |
| `model_M1_best_h5.joblib` | Meilleur M1 (corrections biais) |
| `model_M2_best_h5.joblib` | M2 avec calibration |
| `model_M_star_h5.joblib` | Modele final retenu |
| `metadata_phase2_h5.json` | Metadata complete |
| `comparison_M0_M1_M2.csv` | Tableau comparatif |
| `plots/` | Graphiques AVANT/APRES |
