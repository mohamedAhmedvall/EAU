# Rapport Phase 3 - Modele Final AEP (H=1)

*Date: 2026-01-28 14:18*

---

## Resume Executif

**Modele retenu**: M4_competing_iso

- Lift@10%: 6.21
- Capture@10%: 62.12%
- Brier Score: 0.0104
- ECE: 0.0045
- FN@10%: 150

**Amelioration vs baseline**: +5.6% Lift@10%

## Comparaison des Modeles

| Model | ROC-AUC | Lift@10% | Cap@10% | Brier | ECE | FN |
|-------|---------|----------|---------|-------|-----|-----|
| M0_baseline | 0.8640 | 5.88 | 58.84% | 0.1171 | 0.2176 | 163 |
| M1_competing | 0.8484 | 6.16 | 61.62% | 0.0101 | 0.0013 | 152 |
| M2_mono_clean | 0.8151 | 4.92 | 49.24% | 0.1345 | 0.2283 | 201 |
| M3_mono_reg | 0.8128 | 4.72 | 47.22% | 0.1497 | 0.2584 | 209 |
| M4_competing_iso | 0.8491 | 6.21 | 62.12% | 0.0104 | 0.0045 | 150 |
| M5_mono_iso | 0.8109 | 4.87 | 48.74% | 0.0106 | 0.0018 | 203 |

## Stabilite par Segments

### Par Materiau

| segment   |     n |   events |   capture |   fn_rate |
|:----------|------:|---------:|----------:|----------:|
| ACIE      |   286 |        1 |  0        |  1        |
| BTM       |   432 |        3 |  0.333333 |  0.666667 |
| FT        | 19311 |      207 |  0.681159 |  0.318841 |
| FTG       |  8546 |      161 |  0.608696 |  0.391304 |
| FTTT      |    62 |        1 |  0        |  1        |
| FTVI      |  2259 |        3 |  0.333333 |  0.666667 |
| PEHD      |  1553 |        6 |  0        |  1        |
| POLY      |  1224 |       11 |  0.454545 |  0.545455 |
| PVC       |   719 |        3 |  0        |  1        |

### Par Decennie

|   segment |    n |   events |   capture |   fn_rate |
|----------:|-----:|---------:|----------:|----------:|
|      1900 | 3239 |       12 |  0.416667 |  0.583333 |
|      1940 | 1914 |       62 |  0.66129  |  0.33871  |
|      1950 | 1462 |       27 |  0.703704 |  0.296296 |
|      1960 | 4395 |      103 |  0.728155 |  0.271845 |
|      1970 | 4946 |       92 |  0.717391 |  0.282609 |
|      1980 | 3948 |       21 |  0.666667 |  0.333333 |
|      1990 | 2363 |       18 |  0.444444 |  0.555556 |
|      2000 | 3866 |       25 |  0.56     |  0.44     |
|      2010 | 5744 |       25 |  0        |  1        |
|      2020 | 3715 |        8 |  0.125    |  0.875    |

## Recommandations

1. **Utilisation du score**:
   - `risque_attendu = proba_calibree * longueur_km`
   - Utiliser pour le moteur d'optimisation sous contraintes budget

2. **Monitoring**:
   - Suivre les segments PEHD/PVC (sous-couverts)
   - Recalibrer trimestriellement

3. **Limitations**:
   - L'heuristique d'abandon preventif est approximative
   - Facteurs externes (travaux tiers) non captures

## Artefacts Generes

- `best_model_h1_v2.joblib`: Modele final
- `metadata_h1_v2.json`: Metadata
- `model_comparison.csv`: Tableau comparatif
- `plots/`: Graphiques d'analyse