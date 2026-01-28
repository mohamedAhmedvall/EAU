# Positionnement des Approches - Maintenance Predictive AEP

*Date: 2026-01-28*

---

## 1. APPROCHES EVALUEES

### 1.1 Regles Metier (Baseline heuristique)

**Description**: Priorisation basee sur des regles simples

```
score = age × coef_materiau × (1 + n_anomalies)
```

**Avantages**:
- Interpretable
- Pas d'entrainement
- Validation metier facile

**Inconvenients**:
- Lift@10% ~ 2-3 (faible)
- Pas de calibration
- Ignore les interactions

**Verdict**: Insuffisant pour optimisation sous contraintes

---

### 1.2 Classification Binaire (Baseline ML)

**Description**: LightGBM binaire avec class_weight

**Configuration testee**:
- 22 features
- scale_pos_weight = n_neg/n_pos
- Train/test 80/20 stratifie

**Resultats (M0_baseline)**:
| Metrique | Valeur |
|----------|--------|
| Lift@10% | 5.88 |
| Capture@10% | 58.8% |
| ROC-AUC | 0.864 |
| Brier | 0.117 |
| ECE | 0.218 |

**Avantages**:
- Simple
- Bon ranking
- Industrialisable

**Inconvenients**:
- Mal calibre
- Ne distingue pas defaillances vs abandons

**Verdict**: Bon pour ranking, pas pour optimisation quantitative

---

### 1.3 Survival Analysis (Cox, AFT, RSF)

**Description**: Modeles de survie pour time-to-event

**Approches disponibles**:
- Cox Proportional Hazards
- Accelerated Failure Time (AFT)
- Random Survival Forest (RSF)
- Gradient Boosted Survival

**Resultats (anterieurs - non reproduits ici)**:
| Modele | ROC-AUC | Lift@10% | Note |
|--------|---------|----------|------|
| Cox PH | 0.68 | 1.39 | Hypothese PH violee |
| AFT | ~0.70 | ~1.5 | Parametrique limite |
| RSF | ~0.80 | ~4.0 | Meilleur mais lent |

**Avantages**:
- Gere la censure explicitement
- Interpretable (hazard ratios)
- Theoriquement adapte

**Inconvenients**:
- Hypotheses souvent violees (Cox PH)
- Lift inferieur aux methodes ML
- Plus complexe a industrialiser

**Verdict**: Utile pour comprendre, pas optimal pour le ranking

---

### 1.4 Competing Risks (RETENU)

**Description**: Modele multiclass distinguant:
- Classe 0: Pas d'evenement (survie)
- Classe 1: Defaillance reelle
- Classe 2: Abandon preventif

**Configuration testee**:
- LightGBM multiclass
- 27 features etendues
- objective="multiclass", num_class=3

**Resultats (M1_competing, M4_competing_iso)**:
| Metrique | M1_competing | M4_competing_iso |
|----------|--------------|------------------|
| Lift@10% | 6.16 | **6.21** |
| Capture@10% | 61.6% | **62.1%** |
| Brier | 0.010 | 0.010 |
| ECE | 0.001 | 0.005 |
| FN | 152 | **150** |

**Avantages**:
- Meilleur Lift (+5.6% vs baseline)
- Excellente calibration native
- Separation semantique des evenements
- Reduction FN

**Inconvenients**:
- Heuristique d'abandon approximative
- Complexite interpretative

**Verdict**: APPROCHE RETENUE - meilleur compromis performance/calibration

---

### 1.5 Calibration Post-hoc

**Description**: Ajustement des probabilites apres entrainement

**Methodes testees**:
- Isotonic Regression (non-parametrique)
- Platt Scaling (logistique)

**Resultats**:
| Base | Calibration | Brier avant | Brier apres | Lift preserve? |
|------|-------------|-------------|-------------|----------------|
| M0 | Isotonic | 0.117 | ~0.012 | Oui |
| M1 | Isotonic | 0.010 | 0.010 | Oui |
| M2 | Isotonic | 0.134 | 0.011 | Oui |

**Avantages**:
- Ameliore drastiquement la calibration
- Ne degrade pas le ranking (si bien fait)
- Simple a implementer

**Inconvenients**:
- Necessite un set de calibration separe
- Peut introduire du bruit si trop peu de donnees

**Verdict**: INDISPENSABLE pour l'optimisation

---

### 1.6 Contraintes Monotones

**Description**: Forcer des relations monotones (ex: age ↑ → risque ↑)

**Configuration testee**:
- +1: age, ratio_age, n_fuites, leak_rate
- -1: days_since_last_fuite
- 0: categoriques, interactions

**Resultats (M2_mono_clean, M3_mono_reg)**:
| Metrique | Sans monotone | Avec monotone |
|----------|---------------|---------------|
| Lift@10% | 5.88 | 4.72-4.92 |
| Interpretabilite | Moyenne | Haute |

**Avantages**:
- Garantit la coherence metier
- Evite les artefacts

**Inconvenients**:
- Degrade significativement le Lift dans notre cas
- Trop contraignant sur donnees complexes

**Verdict**: REJETE pour ce dataset - le modele sans contraintes performe mieux

---

## 2. MATRICE DE POSITIONNEMENT

| Approche | Lift@10% | Calibration | Stabilite | Complexite | Industrialisable |
|----------|----------|-------------|-----------|------------|------------------|
| Regles metier | Faible | Non | Haute | Tres faible | Oui |
| Classification binaire | **Bon** | Non | Moyenne | Faible | **Oui** |
| Survie (Cox/AFT) | Faible | Oui | Haute | Moyenne | Moyen |
| Survie (RSF/GB) | Moyen | Oui | Moyenne | Haute | Moyen |
| **Competing risks** | **Excellent** | **Oui** | **Bonne** | Moyenne | **Oui** |
| Monotone constraints | Faible | Non | Haute | Faible | Oui |

---

## 3. RECOMMANDATION FINALE

### Approche selectionnee: Competing Risks + Calibration Isotonique

**Justification**:

1. **Performance top-K maximale**: Lift@10% = 6.21, le plus eleve teste
2. **Calibration utilisable**: Brier = 0.010, pret pour optimisation
3. **Semantique claire**: Distingue defaillances des abandons
4. **Industrialisable**: LightGBM standard + isotonic = facile a deployer

### Architecture finale

```
                    +-------------------+
                    |  Features (27)    |
                    +-------------------+
                            |
                            v
                    +-------------------+
                    | LightGBM Multiclass|
                    | (3 classes)       |
                    +-------------------+
                            |
                            v
                    +-------------------+
                    | P(defaillance)    |
                    | = proba[:, 1]     |
                    +-------------------+
                            |
                            v
                    +-------------------+
                    | Isotonic Calibration|
                    +-------------------+
                            |
                            v
                    +-------------------+
                    | proba_calibree    |
                    +-------------------+
                            |
                            v
                    +-------------------+
                    | risque_attendu =  |
                    | proba × longueur  |
                    +-------------------+
```

### Alternatives futures a explorer

1. **Deep Survival Analysis**: Si plus de donnees disponibles
2. **Temporal backtesting**: Valider sur plusieurs fenetres temporelles
3. **Competing risks explicites**: Fine-Grayou cause-specific hazards
4. **Ensemble**: Combiner plusieurs modeles

---

## 4. FREEZE DATE ANALYSIS

### Date actuelle: 2023-11-07

**Calcul**:
- Max observation: 2024-11-07
- Horizon: 1 an
- Freeze date: 2024-11-07 - 1 an = 2023-11-07

### Proposition de modification?

**NON** - La freeze date actuelle est correcte:
- Permet d'avoir un horizon complet
- Evite le leakage
- Donnees suffisantes pour statistiques de survie

**Alternative a tester dans le futur**:
- Rolling validation avec freeze dates multiples (2022, 2021)
- Temporal cross-validation

---

## 5. CONCLUSION

| Question | Reponse |
|----------|---------|
| Meilleure approche? | Competing risks + isotonic |
| Pret pour optimisation? | **OUI** |
| Freeze date OK? | OUI |
| Leakage? | ZERO detecte |
| Prochaines etapes? | Monitoring + backtesting temporel |
