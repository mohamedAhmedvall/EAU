# Rapport PHASE 2 — Amelioration du modele de renouvellement AEP

---

## 1. Rappel des constats PHASE 1

| Probleme | Impact metier |
|----------|---------------|
| Survivorship bias ages extremes (1900-1920) | FP sur vieux tuyaux survivants, score biaise |
| Calibration mediocre (ECE~0.45) | Scores non utilisables en optimisation |
| Abandons preventifs dans le label | Bruit dans le signal de defaillance reelle |
| Segments PEHD/PVC/FTVI a 0% capture | Materiaux recents sous-representes |
| FN concentres sur tuyaux jeunes (42 vs 55 ans) | Defaillances precoces non detectees |

## 2. Strategies implementees

### A) Ages extremes

Deux strategies comparees:
1. **Conservative**: capping age a 110 ans, flag binaire age extreme, winsorisation P99
2. **Structurelle**: ratio age/mediane decennie, residu age/materiau — capture le vieillissement relatif plutot qu'absolu

### B) Abandons preventifs

**Definition operationnelle** (heuristique):
- DHS dans l'horizon (event=1)
- Zero anomalie historique (n_fuites_total=0)
- Jeune par rapport au materiau (ratio_age_median < 0.6)
- Age absolu < 15 ans
- Pas de fuite recente
- **Resultat**: 121 / 1982 evenements (6.1%) identifies

Trois approches testees:
- **A) Filtrage label**: relabel event→0 pour les abandons
- **B) Competing risks**: modele multi-classe (survie/defaillance/abandon)
- **C) Hybride**: label nettoye + features structurelles + calibration

### C) Calibration

- Isotonic Regression
- Platt Scaling (logistic)
- Comparaison ranking (Lift@K) avant/apres — regle: ne pas sacrifier le ranking

### D) Contraintes monotones

- **Selectif**: monotone seulement sur age et anomaly features
- longueur/diametre laisses libres (relation non-lineaire avec risque)
- days_since_last_fuite: monotone decroissant

## 3. Resultats comparatifs

### Tableau complet

| Modele             |   Lift_10 | Cap_10   |   Brier |    ECE |   FN |   FP | CV          |
|:-------------------|----------:|:---------|--------:|-------:|-----:|-----:|:------------|
| M0_baseline        |      5.88 | 58.8%    |  0.1171 | 0.2176 |  163 | 3341 | 6.06+/-0.14 |
| M1c_competing      |      6.16 | 61.6%    |  0.0101 | 0.0013 |  152 | 3330 | -           |
| M2c_mono_selective |      4.82 | 48.2%    |  0.1344 | 0.2289 |  205 | 3383 | 5.44+/-0.27 |
| M2d_mono_reg       |      4.72 | 47.2%    |  0.1497 | 0.2584 |  209 | 3387 | 5.30+/-0.35 |
| M2e_mono_iso       |      4.77 | 47.7%    |  0.0106 | 0.0018 |  207 | 3385 | -           |
| M3_clean_iso       |      6.09 | 60.9%    |  0.0103 | 0.0036 |  155 | 3333 | -           |
| M4_competing_iso   |      6.21 | 62.1%    |  0.0104 | 0.0045 |  150 | 3328 | -           |
| M5_ext_clean       |      5.98 | 59.8%    |  0.1078 | 0.1902 |  159 | 3337 | 6.66+/-0.23 |
| M5b_ext_clean_iso  |      5.96 | 59.6%    |  0.0102 | 0.004  |  160 | 3338 | -           |

### Decisions Go/No-Go

- **M1c_competing**: Lift@10% +0.28 | FN -11 → **GO**
- **M2c_mono_selective**: Lift@10% -1.06 | FN +42 → **NO-GO (degrade Lift@10%)**
- **M2d_mono_reg**: Lift@10% -1.16 | FN +46 → **NO-GO (degrade Lift@10%)**
- **M2e_mono_iso**: Lift@10% -1.11 | FN +44 → **NO-GO (degrade Lift@10%)**
- **M3_clean_iso**: Lift@10% +0.20 | FN -8 → **GO**
- **M4_competing_iso**: Lift@10% +0.33 | FN -13 → **GO**
- **M5_ext_clean**: Lift@10% +0.10 | FN -4 → **GO**
- **M5b_ext_clean_iso**: Lift@10% +0.08 | FN -3 → **GO**

## 4. Modele retenu: **M4_competing_iso**

### Metriques

| Metrique | Valeur |
|----------|--------|
| ROC-AUC | 0.8491 |
| Lift@5% | 9.70 |
| Lift@10% | 6.21 |
| Lift@20% | 3.83 |
| Capture@10% | 62.12% |
| Brier Score | 0.0104 |
| ECE | 0.0045 |
| FN@10% | 150 |
| FP@10% | 3328 |

### Gains vs M0 baseline

| Metrique | M0 | M* | Delta |
|----------|----|----|-------|
| Lift@10% | 5.88 | 6.21 | +0.33 |
| Capture@10% | 58.84% | 62.12% | +3.28% |
| FN@10% | 163 | 150 | -13 |
| Brier | 0.1171 | 0.0104 | -0.1067 |

## 5. Stabilite par segments

### Par materiau

| segment   |     n |   events |   captured |   capture |   fn_rate |   avg_score |
|:----------|------:|---------:|-----------:|----------:|----------:|------------:|
| ACIE      |   286 |        1 |          0 |  0        |  1        | 0.00040191  |
| BTM       |   432 |        3 |          1 |  0.333333 |  0.666667 | 0.00352969  |
| FT        | 19311 |      207 |        141 |  0.681159 |  0.318841 | 0.00993132  |
| FTG       |  8546 |      161 |         98 |  0.608696 |  0.391304 | 0.0172496   |
| FTTT      |    62 |        1 |          0 |  0        |  1        | 0           |
| FTVI      |  2259 |        3 |          1 |  0.333333 |  0.666667 | 0.00108161  |
| PEHD      |  1553 |        6 |          0 |  0        |  1        | 0.000564395 |
| POLY      |  1224 |       11 |          5 |  0.454545 |  0.545455 | 0.00922406  |
| PVC       |   719 |        3 |          0 |  0        |  1        | 0.00192501  |

### Par decennie

|   segment |    n |   events |   captured |   capture |   fn_rate |   avg_score |
|----------:|-----:|---------:|-----------:|----------:|----------:|------------:|
|      1900 | 3239 |       12 |          5 |  0.416667 |  0.583333 | 0.00213505  |
|      1940 | 1914 |       62 |         41 |  0.66129  |  0.33871  | 0.030064    |
|      1950 | 1462 |       27 |         19 |  0.703704 |  0.296296 | 0.0177178   |
|      1960 | 4395 |      103 |         75 |  0.728155 |  0.271845 | 0.0237903   |
|      1970 | 4946 |       92 |         66 |  0.717391 |  0.282609 | 0.0180427   |
|      1980 | 3948 |       21 |         14 |  0.666667 |  0.333333 | 0.0062812   |
|      1990 | 2363 |       18 |          8 |  0.444444 |  0.555556 | 0.00700349  |
|      2000 | 3866 |       25 |         14 |  0.56     |  0.44     | 0.00651216  |
|      2010 | 5744 |       25 |          0 |  0        |  1        | 0.000116204 |
|      2020 | 3715 |        8 |          1 |  0.125    |  0.875    | 0.000581775 |

## 6. Usage en optimisation

### Definition du risque attendu

```
risque_attendu_i = proba_calibree_i * longueur_km_i
```

Ou `proba_calibree_i` est la sortie du modele M* pour le troncon i.
Cette valeur peut etre utilisee comme "valeur" dans un solveur d'optimisation
sous contrainte budgetaire (knapsack) ou de chantiers groupes.

## 7. Recommandation finale

### Modele retenu: **M4_competing_iso**

**Justification:**
- Amelioration Lift@10% de 0.33 (+5.6%)
- Reduction de 13 faux negatifs (13 defaillances mieux detectees)
- Calibration nettement amelioree (Brier: 0.1171 → 0.0104)

### Hypotheses metier restantes

1. L'heuristique d'abandon preventif est une approximation — idealement valider avec les motifs DHS du SIG
2. Segments PEHD/PVC/FTVI sous-representes — enrichir si possible
3. Defaillances externes (travaux tiers, mouvement terrain) non modelisees
4. Decennies 2010-2020: faible capture car peu d'evenements historiques

### Pret pour moteur d'optimisation?

**OUI**, sous reserve de:
- Utiliser `risque_attendu = proba_calibree * longueur_km`
- Monitorer la performance par segment trimestriellement
- Recalibrer annuellement avec nouvelles donnees
- Ne pas utiliser les scores bruts comme probabilites absolues sans la couche de calibration

## 8. Artefacts

| Fichier | Description |
|---------|-------------|
| `model_M_star.joblib` | Modele final (M4_competing_iso) |
| `metadata_phase2.json` | Metadata (features, calibration, metrics) |
| `comparison_all_models.csv` | Tableau comparatif complet |
| `summary_phase2.json` | Resume JSON |
| `plots/` | Graphiques comparatifs |