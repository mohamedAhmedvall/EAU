# Rapport PHASE 2 — Amelioration du modele de renouvellement

*Genere automatiquement (horizon: 1 an(s))*

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
- Detectes: 121 / 1982 evenements (6.1%)

**Approches testees:**
- A) Filtrage label (event -> 0 pour abandons)
- B) Competing risks (multi-class: 0=survie, 1=defaillance, 2=abandon)

### Resultats Iteration 1

| Modele | Lift@10% | Cap@10% | ROC-AUC | FN | Decision |
|--------|----------|---------|---------|----| ---------|
| M1a | 6.43 | 64.29% | 0.8671 | 140 | rejete |
| M1b | 6.40 | 64.03% | 0.8649 | 141 | rejete |
| M1c | 6.67 | 66.67% | 0.9152 | 132 | RETENU |

## 3. Iteration 2 — Calibration & Stabilite (M2)

### Contraintes monotones

- age, ratio_age_median, overdue_years: monotone croissant (+1)
- days_since_last_fuite: monotone decroissant (-1)
- Regularisation: lambda=2.0, alpha=0.3, min_child=50

### Calibration

| Variante | Lift@10% | Brier | ECE |
|----------|----------|-------|-----|
| M2b_raw | 6.07 | 0.1301 | 0.2335 |
| M2b_iso | 6.07 | 0.0104 | 0.0016 |
| M2b_platt | 6.07 | 0.0103 | 0.0015 |

**Calibration retenue**: M2b_raw

### Definition risque_attendu

```
risque_attendu = proba_calibree × longueur_km
```
Ou proba_calibree est la sortie du modele apres calibration isotonique/Platt.

## 4. Iteration 3 — Comparaison finale

| Model | ROC_AUC | Lift_5 | Lift_10 | Lift_20 | Cap_5 | Cap_10 | Cap_20 | Brier | ECE | FN | FP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0_baseline | 0.8640 | 8.7879 | 5.8838 | 3.7374 | 0.4394 | 0.5884 | 0.7475 | 0.1171 | 0.2176 | 163 | 3341 |
| M1c_competing | 0.9152 | 9.9495 | 6.6667 | 4.1035 | 0.4975 | 0.6667 | 0.8207 | 0.0102 | 0.0014 | 132 | 3310 |
| M2b_raw | 0.8587 | 8.9286 | 6.0714 | 3.8776 | 0.4464 | 0.6071 | 0.7755 | 0.1301 | 0.2335 | 154 | 3336 |

## 5. Stabilite par segments

### Par materiau (M*)

| segment | n | events | capture | fn_rate | avg_score |
| --- | --- | --- | --- | --- | --- |
| BTM | 432 | 3 | 1.0000 | 0.0000 | 0.0030 |
| FT | 19307 | 203 | 0.6995 | 0.3005 | 0.0093 |
| FTG | 8555 | 170 | 0.6529 | 0.3471 | 0.0187 |
| FTVI | 2261 | 5 | 0.0000 | 1.0000 | 0.0016 |
| PEHD | 1552 | 5 | 0.2000 | 0.8000 | 0.0015 |
| POLY | 1221 | 8 | 0.7500 | 0.2500 | 0.0107 |
| PVC | 718 | 2 | 0.5000 | 0.5000 | 0.0032 |

### Par decennie (M*)

| segment | n | events | capture | fn_rate | avg_score |
| --- | --- | --- | --- | --- | --- |
| 1900 | 3235 | 8 | 0.6250 | 0.3750 | 0.0026 |
| 1930 | 143 | 4 | 1.0000 | 0.0000 | 0.0134 |
| 1940 | 1924 | 72 | 0.6528 | 0.3472 | 0.0296 |
| 1950 | 1462 | 27 | 0.7037 | 0.2963 | 0.0188 |
| 1960 | 4398 | 106 | 0.7642 | 0.2358 | 0.0249 |
| 1970 | 4944 | 90 | 0.8222 | 0.1778 | 0.0168 |
| 1980 | 3948 | 21 | 0.6190 | 0.3810 | 0.0069 |
| 1990 | 2353 | 8 | 0.3750 | 0.6250 | 0.0076 |
| 2000 | 3868 | 27 | 0.5556 | 0.4444 | 0.0065 |
| 2010 | 5736 | 17 | 0.0588 | 0.9412 | 0.0003 |
| 2020 | 3721 | 14 | 0.0000 | 1.0000 | 0.0009 |

## 6. Recommandation finale

### Modele retenu: **M1c_competing**

- **Lift@10%**: 6.67
- **Capture@10%**: 66.67%
- **Brier Score**: 0.0102
- **ECE**: 0.0014
- **FN@10%**: 132
- **FP@10%**: 3310

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
| `model_M0.joblib` | Baseline LightGBM |
| `model_M1_best.joblib` | Meilleur M1 (corrections biais) |
| `model_M2_best.joblib` | M2 avec calibration |
| `model_M_star.joblib` | Modele final retenu |
| `metadata_phase2.json` | Metadata complete |
| `comparison_M0_M1_M2.csv` | Tableau comparatif |
| `plots/` | Graphiques AVANT/APRES |
