# RAPPORT FINAL - Modele AEP v3

*Date: 2026-01-28*

---

## RESUME EXECUTIF

### Modele Retenu: HistGradientBoostingClassifier

| Metrique | Baseline Original | M4_competing_iso | **v3_final** | Amelioration |
|----------|-------------------|------------------|--------------|--------------|
| **Lift@10%** | 5.88 | 6.21 | **6.57** | **+11.7%** |
| **Capture@10%** | 58.8% | 62.1% | **65.7%** | +6.9 pp |
| **Brier** | 0.117 | 0.010 | **0.010** | -91% |
| **FN@10%** | 163 | 150 | **136** | **-16.6%** |
| **ROC-AUC** | 0.864 | 0.849 | **0.879** | +1.5% |

### Conclusion

Le modele v3_final represente une **amelioration majeure** par rapport a toutes les versions precedentes:
- **+11.7% de Lift@10%** par rapport au baseline original
- **16.6% de reduction des defaillances manquees** (FN)
- **Excellente calibration** (Brier = 0.010) pour l'optimisation
- **Plus simple** qu'un ensemble ou competing risks

---

## ARCHITECTURE TECHNIQUE

### Algorithme

```
HistGradientBoostingClassifier (sklearn)
├── max_iter: 400
├── learning_rate: 0.03
├── max_depth: None (unlimited)
├── min_samples_leaf: 50
└── random_state: 42

+ IsotonicRegression (calibration post-hoc)
```

### Features (27)

**Base (22)**:
- Physiques: age_at_freeze, diametre, longueur, log_*
- Anomalies: n_fuites_*, days_since_last_fuite, leak_rate_*
- Survie: ratio_age_median, overdue_years, over_p75/p90
- Interactions: age_x_nfuites, surface_approx, age_x_ratio
- Categoriques: materiau_enc, decade_install_enc

**Corrections biais (5)**:
- age_cap (clip 110 ans)
- age_extreme (flag >110 ans)
- age_winsor (P99)
- age_ratio_dec (ratio age/mediane decennie)
- age_resid_mat (residuel vs mediane materiau)

---

## COMPARAISON MODELES TESTES

### Phase 4 - Recherche exhaustive (60+ modeles)

| Rang | Modele | Lift@10% | Note |
|------|--------|----------|------|
| 1 | **HistGradientBoosting** | **6.57** | **RETENU** |
| 2 | HistGradientBoosting variants | 6.57 | Meme perf |
| 3 | XGBoost | 6.39 | -2.7% |
| 4 | LightGBM | 6.21 | -5.5% |
| 5 | Voting Ensemble | 6.44 | -2.0% |
| 6 | Stacking | 6.21 | -5.5% |
| 7 | Random Forest | 5.30 | -19.3% |

### Observations

1. **HistGradientBoosting** surpasse tous les autres algorithmes sur ce dataset
2. **Les ensembles n'ameliorent pas** le meilleur modele single
3. **La calibration isotonique** preserve le ranking et ameliore les probas
4. **Profondeur illimitee** essentielle (max_depth=None)
5. **min_samples_leaf=50** meilleur que 10 ou 20

---

## ANALYSE PAR SEGMENT

### Par Materiau

| Materiau | Events | Capture | Status |
|----------|--------|---------|--------|
| FT | 207 | 71.0% | OK |
| FTG | 161 | 63.4% | OK |
| POLY | 11 | 63.6% | OK |
| BTM | 3 | 66.7% | OK |
| FTVI | 3 | 33.3% | Attention |
| PEHD | 6 | 16.7% | **Probleme** |
| PVC | 3 | 0% | **Probleme** |
| ACIE | 1 | 0% | Rare |
| FTTT | 1 | 0% | Rare |

**Recommandation**: Surveiller PEHD/PVC - materiaux recents avec peu d'historique.

### Par Decennie

| Decennie | Events | Capture | Status |
|----------|--------|---------|--------|
| 1950 | 27 | 85.2% | Excellent |
| 1960 | 103 | 76.7% | Bon |
| 1970 | 92 | 73.9% | Bon |
| 1980 | 21 | 71.4% | Bon |
| 1940 | 62 | 59.7% | Moyen |
| 2000 | 25 | 60.0% | Moyen |
| 2020 | 8 | 50.0% | A surveiller |
| 1990 | 18 | 44.4% | A surveiller |
| 1900 | 12 | 41.7% | Ages extremes |
| **2010** | 25 | **12.0%** | **Probleme** |

**Recommandation**: Attention aux tuyaux installes 2010-2020 - modele moins performant.

---

## ANTI-LEAKAGE

### Tests automatiques

| Test | Resultat |
|------|----------|
| Anomalies post-freeze | PASS |
| DHS pre-freeze | PASS |
| Coherence temporelle | PASS |
| Target dans horizon | PASS |

**Zero leakage detecte.**

---

## USAGE PRODUCTION

### Training

```bash
python src/model_final.py --train
```

### Scoring

```bash
python src/model_final.py --score \
  --input data/new_pipes.csv \
  --output reports/risk_scores.csv
```

### API Python

```python
from model_final import FinalModel, prepare_features, get_X

# Charger le modele
model = FinalModel.load("artifacts/model_v3_final.joblib")

# Preparer les donnees
df = prepare_features(df_raw)
X = get_X(df)

# Predire
results = model.predict_risk(X)
# -> risk_score, risk_category, risk_rank
```

### Optimisation budget

```python
# Score pour solveur d'optimisation
risque_attendu = risk_score * longueur_km * cout_defaillance_par_km

# Contrainte: sum(selected) <= budget
```

---

## ARTEFACTS

| Fichier | Description |
|---------|-------------|
| `artifacts/model_v3_final.joblib` | Modele final (HGB + calibrateur) |
| `artifacts/metadata_v3_final.json` | Metadata et metriques |
| `src/model_final.py` | Script de production |
| `src/phase4_search.py` | Recherche exhaustive |
| `src/phase4_optimize.py` | Optimisation hyperparametres |
| `src/phase4_final.py` | Experiments finaux |

---

## MONITORING RECOMMANDE

1. **Metriques a suivre**:
   - Lift@10% (alerte si < 6.0)
   - ECE (alerte si > 0.05)
   - Capture par segment (alerte si segment < 30%)

2. **Frequence**:
   - Recalibrer tous les 6 mois
   - Re-entrainer annuellement

3. **Drift detection**:
   - Comparer distribution features train vs prod
   - Alerter si drift > 2 std

---

## LIMITATIONS CONNUES

1. **Segments faibles**: PEHD, PVC, decennie 2010
2. **Ages extremes**: >100 ans moins fiables
3. **Abandons preventifs**: Heuristique approximative
4. **Facteurs externes**: Travaux tiers non captures

---

## CONCLUSION

Le modele v3_final est **pret pour la production**:
- Performance superieure a toutes les versions precedentes
- Calibration excellente pour l'optimisation
- Simple a deployer (sklearn standard)
- Anti-leakage valide
