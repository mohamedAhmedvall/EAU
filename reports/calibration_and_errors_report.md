# Rapport: Calibration des Probabilités et Analyse des Erreurs

*Généré le 2026-01-23 11:39:38*

---

## 1. Calibration des Probabilités

### Objectif
Vérifier si une probabilité prédite de 70% correspond réellement à un risque de 70%.

### Métriques Globales

- **Brier Score**: 0.1120 (0 = parfait, plus bas est mieux)
- **Log Loss**: 0.3375 (plus bas est mieux)
- **Expected Calibration Error (ECE)**: 0.4464 (0 = parfaitement calibré)

### Interprétation

❌ **Calibration médiocre** - Recalibration recommandée

### Calibration par Bin de Probabilité

| bin     |   n_samples |   mean_predicted |   actual_rate |   difference |
|:--------|------------:|-----------------:|--------------:|-------------:|
| 0.0-0.1 |       87842 |            0.031 |         0.000 |        0.030 |
| 0.1-0.2 |       24409 |            0.142 |         0.001 |        0.141 |
| 0.2-0.3 |       15125 |            0.247 |         0.003 |        0.244 |
| 0.3-0.4 |       10187 |            0.349 |         0.006 |        0.343 |
| 0.4-0.5 |        8877 |            0.448 |         0.007 |        0.441 |
| 0.5-0.6 |        7606 |            0.549 |         0.015 |        0.534 |
| 0.6-0.7 |        8252 |            0.649 |         0.022 |        0.627 |
| 0.7-0.8 |        9081 |            0.750 |         0.046 |        0.704 |
| 0.8-0.9 |        5711 |            0.847 |         0.109 |        0.738 |
| 0.9-1.0 |        1603 |            0.928 |         0.266 |        0.662 |

**Lecture du tableau**:
- `mean_predicted`: Probabilité moyenne prédite dans ce bin
- `actual_rate`: Taux de défaillance réel observé
- `difference`: Écart (positif = surestimation, négatif = sous-estimation)

### Recommandations

- ⚠️ **Surestimation** dans les bins: 0.1-0.2, 0.2-0.3, 0.3-0.4, 0.4-0.5, 0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, 0.9-1.0
  → Le modèle prédit des probabilités trop élevées

**Solutions**:
1. Appliquer Isotonic Regression pour recalibrer
2. Utiliser Platt Scaling (régression logistique)
3. Ajuster le seuil de décision selon les coûts métier

---

## 2. Analyse des Faux Négatifs (FN)

### Caractéristiques des FN vs TP

{df_fn_comparison.to_markdown(index=False)}

### Insights Clés - Faux Négatifs

1. **Les FN sont plus jeunes** (42.9 ans vs 54.9 ans)
   - Le modèle sous-estime le risque des tronçons jeunes qui défaillent
   - Possiblement des défaillances dues à défauts de fabrication/pose

2. **Score moyen des FN**: 0.313
   - Ces tronçons étaient juste sous le seuil de 0.5
   - Un seuil plus bas (ex: 0.4) permettrait de les détecter


### Actions Recommandées

1. **Baisser le seuil de décision** à 0.4 pour capturer plus de défaillances
2. **Enrichir les features** pour mieux prédire les défaillances précoces
3. **Analyse par expertise** des FN pour identifier patterns manquants

---

## 3. Analyse des Faux Positifs (FP)

### Caractéristiques des FP vs TN

{df_fp_comparison.to_markdown(index=False)}

### Insights Clés - Faux Positifs

1. **Les FP sont plus âgés** (57.1 ans vs 39.8 ans)
   - Le modèle surévalue le risque lié à l'âge seul
   - Certains tronçons anciens sont en bon état malgré leur âge

2. **Score moyen des FP**: 0.697
   - ⚠️ Le modèle est très confiant sur ces FP
   - Problème de calibration ou features manquantes (qualité, entretien)


### Actions Recommandées

1. **Augmenter le seuil** à 0.6 pour réduire les FP (mais augmente les FN)
2. **Ajouter features de qualité** (inspections, maintenance préventive)
3. **Analyse coût-bénéfice** pour trouver le seuil optimal

---

## 4. Sous-groupes Problématiques

Les combinaisons matériau/âge suivantes ont des taux d'erreur élevés:

- **FTG / 70-100 ans**: 54.0% d'erreur (FP: 53.9%, FN: 0.1%) | n=13345
- **FTG / 50-70 ans**: 49.2% d'erreur (FP: 49.1%, FN: 0.1%) | n=17343
- **POLY / 50-70 ans**: 44.0% d'erreur (FP: 44.0%, FN: 0.0%) | n=2575
- **FT / 50-70 ans**: 42.9% d'erreur (FP: 42.7%, FN: 0.1%) | n=16631
- **POLY / 70-100 ans**: 25.0% d'erreur (FP: 25.0%, FN: 0.0%) | n=296

**Action**: Analyser ces sous-groupes spécifiquement et ajuster le modèle

---

## 5. Synthèse et Plan d'Action

### Priorités Immédiates

1. **Optimiser le seuil de décision**
   - Analyser courbe coût FP vs FN
   - Tester seuils entre 0.4 et 0.6

2. **Recalibrer le modèle**
   - Appliquer Isotonic Regression ou Platt Scaling
   - Améliorer la fiabilité des probabilités

3. **Enrichir les features**
   - Ajouter historique d'anomalies
   - Intégrer données de maintenance

### Impact Attendu

- **Réduction des FP** → Moins d'inspections inutiles
- **Réduction des FN** → Moins de défaillances surprises
- **Meilleure calibration** → Priorisation plus fiable

---

## 6. Fichiers Générés

1. `calibration_curve.png` - Courbe de calibration
2. `misclassified_deep_dive.png` - Analyse détaillée FP/FN
3. `fn_details.csv` - Détails des faux négatifs
4. `fp_details.csv` - Détails des faux positifs
5. `problematic_groups.csv` - Sous-groupes problématiques

---

*Analyse générée automatiquement*
