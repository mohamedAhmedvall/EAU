# INVENTAIRE DES BRANCHES GIT - Projet AEP Renouvellement

*Date: 2026-01-28*

---

## RESUME EXECUTIF

| Branche | Statut | Lift@10% | Brier | Idee cle | Risque |
|---------|--------|----------|-------|----------|--------|
| main | Production | 6.01 | ~0.09 | Baseline LightGBM | Reference |
| claude/improve-water-renewal-model-LiFVo | **MEILLEUR** | ~6.35 | ~0.008 | M4 competing+iso | A valider |
| codex/ameliorer-le-modele | Archive | 6.01 | ~0.09 | Phase 2 init | Fusionne |
| claude/error-distribution-analysis | Archive | 6.01 | ~0.09 | Analyse erreurs | Fusionne |

---

## 1. BRANCHE: `main`

### Objectif
Branche de production stable avec le modele LightGBM baseline.

### Differences majeures
- Pipeline complet: data_audit -> survival_eda -> dataset_builder -> modeling
- 22 features: age, anomalies, survie, interactions
- Modele: LightGBM avec class_weight
- Validation: train/test 80/20 stratifie

### Resultats connus
- ROC-AUC: 0.866
- Lift@10%: 6.01
- Capture@10%: 60.1%
- Brier: ~0.09

### Risques
- Pas de correction pour ages extremes (1900-1920)
- Pas de filtrage des abandons preventifs
- Calibration mediocre (ECE eleve)
- Segments PEHD/PVC sous-couverts

---

## 2. BRANCHE: `claude/improve-water-renewal-model-LiFVo`

### Objectif
Amelioration iterative du modele avec corrections biais et calibration.

### Differences majeures
- **Nouveau fichier `phase2_full.py`** (1195 lignes): pipeline complet Phase 2
- **Nouveau fichier `phase2_iter2.py`** (805 lignes): iterations supplementaires
- **Scripts prod**: train.py, evaluate.py, score.py
- Corrections:
  - Ages extremes: capping 110 ans, winsorisation P99, features structurelles
  - Abandons preventifs: detection heuristique + relabeling
  - Competing risks: multiclass (survie/defaillance/abandon)
  - Calibration: isotonic + Platt
  - Contraintes monotones selectives

### Resultats connus (commit message)
- **M4_competing_iso retenu**
- Lift@10%: +5.6% vs baseline
- FN: -8% vs baseline
- Brier: -91% (0.09 -> ~0.008)

### Risques
- Non merge dans main
- A valider reproductibilite
- Tests anti-leakage a verifier

---

## 3. BRANCHE: `codex/ameliorer-le-modele-de-priorisation-des-renouvellements`

### Objectif
Premiere tentative d'amelioration du modele de priorisation.

### Differences majeures
- Ajout `phase2_experiments.py`
- Analyse anomalies: `anomaly_analysis.py`
- Calibration: `calibration_and_misclassified_analysis.py`
- Error analysis: `error_analysis.py`

### Resultats connus
- Phase 2 experiments initiaux
- Analyse des biais par segment
- Identification problemes calibration

### Risques
- Fusionne dans main (PR #6)
- Iterations incompletes

---

## 4. BRANCHE: `codex/ameliorer-le-modele-de-priorisation-des-renouvellements-3sjbkw`

### Objectif
Continuation de l'amelioration avec merge de l'analyse d'erreurs.

### Differences majeures
- Merge de `claude/error-distribution-analysis-aT2cn`
- Handling pickle aliasing
- Reliability bins implementation

### Resultats connus
- Fusionne dans main (PR #7)

### Risques
- Aucun (deja en production)

---

## 5. BRANCHE: `claude/error-distribution-analysis-aT2cn`

### Objectif
Analyse detaillee de la distribution des erreurs du modele.

### Differences majeures
- Ajout `error_analysis.py` complet
- Analyse FP/FN par:
  - Age
  - Materiau
  - Diametre
  - Decennie d'installation
- Identification des groupes problematiques

### Resultats connus
- Problemes identifies sur ages extremes (>100 ans)
- Segments PEHD/PVC avec 0% capture
- FN concentres sur tuyaux jeunes sans historique

### Risques
- Fusionne via PR

---

## 6. BRANCHE: `claude/water-renewal-planning-app-daIti`

### Objectif
Application Streamlit pour la planification des renouvellements.

### Differences majeures
- Interface web pour visualisation des priorites
- Non pertinent pour le modele ML

### Risques
- Hors scope

---

## PIPELINE END-TO-END CARTOGRAPHIE

```
                    PIPELINE DE MAINTENANCE PREDICTIVE AEP

    +-----------------+     +------------------+     +------------------+
    |   RAW DATA      |     |   FREEZE DATE    |     |     LABELS       |
    | v1_trafic.csv   | --> |  max_obs - H     | --> |  event = 1 si    |
    | historique.csv  |     |  (2023-11-07     |     |  DHS in horizon  |
    +-----------------+     |   pour H=1)      |     +------------------+
                            +------------------+
                                    |
                                    v
    +------------------------------------------------------------------+
    |                     FEATURE ENGINEERING (22 features)            |
    |  - Base: age_at_freeze, diametre, longueur                       |
    |  - Anomalies: n_fuites_*, days_since_last_fuite, leak_rate       |
    |  - Survie: ratio_age_median, overdue_years, over_p75/p90         |
    |  - Interactions: age_x_nfuites, surface_approx, age_x_ratio      |
    |  - Categoriques: materiau_encoded, decade_install_encoded        |
    +------------------------------------------------------------------+
                                    |
                                    v
    +------------------------------------------------------------------+
    |                         MODELISATION                              |
    |  - Split: 80/20 stratifie                                        |
    |  - Modeles: Baseline, Logistic, Cox, HistGB, LightGBM            |
    |  - Best: LightGBM (scale_pos_weight)                             |
    +------------------------------------------------------------------+
                                    |
                                    v
    +------------------------------------------------------------------+
    |                         EVALUATION                                |
    |  - Primary: Lift@K, Capture@K (K=5,10,20%)                       |
    |  - Calibration: Brier, ECE, reliability curve                    |
    |  - Stabilite: capture par segment (materiau, decennie)           |
    |  - Confusion Top-K: TP, FP, FN, TN                               |
    +------------------------------------------------------------------+
                                    |
                                    v
    +------------------------------------------------------------------+
    |                         SCORING PROD                              |
    |  - predict_risk.py: calcul features + inference                  |
    |  - API FastAPI: endpoints batch/single                           |
    |  - Sortie: risk_score, risk_category                             |
    +------------------------------------------------------------------+
```

---

## DECISION: BRANCHES A FUSIONNER

1. **PRIORITE 1**: `claude/improve-water-renewal-model-LiFVo`
   - Contient les meilleures ameliorations
   - A reproduire et valider

2. **PRIORITE 2**: Verifier que tout de `codex/ameliorer...` est dans main
   - Deja fusionne via PRs

3. **HORS SCOPE**: `claude/water-renewal-planning-app-daIti`
   - Application UI, pas de code modele

---

## PROCHAINES ETAPES

1. Checkout et merge du code de `improve-water-renewal-model-LiFVo`
2. Reproduire la baseline avec metriques completes
3. Executer phase2_full.py et valider les resultats
4. Ajouter tests anti-leakage automatises
5. Iterer pour ameliorer encore si possible
