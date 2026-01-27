# Rapport d'Analyse de Distribution des Erreurs

## Métadonnées du Modèle

- **Modèle**: LightGBM
- **Horizon de prédiction**: 1 an(s)
- **Date de freeze**: 2023-11-07 00:00:00
- **Date de création**: 2026-01-19 19:39:43

---

## 1. Vue d'Ensemble - Matrice de Confusion

| | Prédit Négatif | Prédit Positif |
|---|---|---|
| **Réel Négatif** | 146,224 (TN) | 30,487 (FP) |
| **Réel Positif** | 216 (FN) | 1,766 (TP) |

### Métriques Globales

- **Total de tronçons**: 178,693
- **Accuracy**: 82.82%
- **Precision**: 5.48% (Parmi les prédictions positives, quelle proportion est correcte)
- **Recall (Sensibilité)**: 89.10% (Parmi les événements réels, quelle proportion est détectée)
- **F1-Score**: 0.103
- **Taux de Faux Positifs (FPR)**: 17.25%
- **Taux de Faux Négatifs (FNR)**: 10.90%

### Répartition

- **Vrais Positifs (TP)**: 1,766 (1.0%)
- **Vrais Négatifs (TN)**: 146,224 (81.8%)
- **Faux Positifs (FP)**: 30,487 (17.1%)
  - *Tronçons prédits à risque mais qui n'ont pas eu de défaillance*
- **Faux Négatifs (FN)**: 216 (0.1%)
  - *Tronçons prédits sûrs mais qui ont eu une défaillance*

---

## 2. Distribution des Erreurs par Matériau

Les 10 matériaux les plus fréquents:

| materiau   |   n_total |   n_events |   event_rate |   TP |    TN |    FP |   FN |   precision |   recall |   fpr |
|:-----------|----------:|-----------:|-------------:|-----:|------:|------:|-----:|------------:|---------:|------:|
| FT         |     96730 |       1019 |         0.01 |  881 | 83121 | 12590 |  138 |        0.07 |     0.86 |  0.13 |
| FTG        |     42725 |        816 |         0.02 |  759 | 25734 | 16175 |   57 |        0.04 |     0.93 |  0.39 |
| FTVI       |     11245 |         23 |         0.00 |   20 | 10905 |   317 |    3 |        0.06 |     0.87 |  0.03 |
| PEHD       |      7763 |         19 |         0.00 |   12 |  7648 |    96 |    7 |        0.11 |     0.63 |  0.01 |
| POLY       |      5940 |         72 |         0.01 |   67 |  4622 |  1246 |    5 |        0.05 |     0.93 |  0.21 |
| AUTRE      |      5641 |          1 |         0.00 |    1 |  5639 |     1 |    0 |        0.50 |     1.00 |  0.00 |
| PVC        |      3656 |         13 |         0.00 |   10 |  3607 |    36 |    3 |        0.22 |     0.77 |  0.01 |
| BTM        |      2080 |         13 |         0.01 |   12 |  2059 |     8 |    1 |        0.60 |     0.92 |  0.00 |
| ACIE       |      1418 |          5 |         0.00 |    4 |  1408 |     5 |    1 |        0.44 |     0.80 |  0.00 |
| A.C        |       320 |          0 |         0.00 |    0 |   320 |     0 |    0 |        0.00 |     0.00 |  0.00 |

### Observations par Matériau


**FT**:
- Total: 96,730 tronçons
- Événements réels: 1,019 (1.1%)
- Faux Positifs: 12,590
- Faux Négatifs: 138
- Précision: 6.5%
- Rappel: 86.5%

**FTG**:
- Total: 42,725 tronçons
- Événements réels: 816 (1.9%)
- Faux Positifs: 16,175
- Faux Négatifs: 57
- Précision: 4.5%
- Rappel: 93.0%

**FTVI**:
- Total: 11,245 tronçons
- Événements réels: 23 (0.2%)
- Faux Positifs: 317
- Faux Négatifs: 3
- Précision: 5.9%
- Rappel: 87.0%

**PEHD**:
- Total: 7,763 tronçons
- Événements réels: 19 (0.2%)
- Faux Positifs: 96
- Faux Négatifs: 7
- Précision: 11.1%
- Rappel: 63.2%

**POLY**:
- Total: 5,940 tronçons
- Événements réels: 72 (1.2%)
- Faux Positifs: 1,246
- Faux Négatifs: 5
- Précision: 5.1%
- Rappel: 93.1%

---

## 3. Distribution des Erreurs par Décennie d'Installation

|   decade_install |   n_total |   n_events |   event_rate |     TP |       TN |       FP |    FN |   precision |   recall |   fpr |
|-----------------:|----------:|-----------:|-------------:|-------:|---------:|---------:|------:|------------:|---------:|------:|
|          2010.00 |  28889.00 |      98.00 |         0.00 |  65.00 | 28144.00 |   647.00 | 33.00 |        0.09 |     0.66 |  0.02 |
|          1970.00 |  25274.00 |     438.00 |         0.02 | 386.00 | 20326.00 |  4510.00 | 52.00 |        0.08 |     0.88 |  0.18 |
|          1960.00 |  21831.00 |     565.00 |         0.03 | 556.00 |  9133.00 | 12133.00 |  9.00 |        0.04 |     0.98 |  0.57 |
|          1980.00 |  19909.00 |     124.00 |         0.01 |  94.00 | 17927.00 |  1858.00 | 30.00 |        0.05 |     0.76 |  0.09 |
|          2000.00 |  19098.00 |     135.00 |         0.01 | 106.00 | 17220.00 |  1743.00 | 29.00 |        0.06 |     0.79 |  0.09 |
|          2020.00 |  18585.00 |      50.00 |         0.00 |  43.00 | 18073.00 |   462.00 |  7.00 |        0.09 |     0.86 |  0.02 |
|          1900.00 |  16142.00 |      40.00 |         0.00 |  24.00 | 15983.00 |   119.00 | 16.00 |        0.17 |     0.60 |  0.01 |
|          1990.00 |  11678.00 |      78.00 |         0.01 |  58.00 | 10955.00 |   645.00 | 20.00 |        0.08 |     0.74 |  0.06 |
|          1940.00 |   9404.00 |     306.00 |         0.03 | 294.00 |  2570.00 |  6528.00 | 12.00 |        0.04 |     0.96 |  0.72 |
|          1950.00 |   7117.00 |     128.00 |         0.02 | 120.00 |  5249.00 |  1740.00 |  8.00 |        0.06 |     0.94 |  0.25 |
|          1930.00 |    723.00 |      10.00 |         0.01 |  10.00 |   626.00 |    87.00 |  0.00 |        0.10 |     1.00 |  0.12 |
|          1920.00 |     40.00 |      10.00 |         0.25 |  10.00 |    17.00 |    13.00 |  0.00 |        0.43 |     1.00 |  0.43 |

---

## 4. Caractéristiques Comparatives des Erreurs

Comparaison des caractéristiques moyennes entre les différents types de prédictions:

| Caractéristique        | Faux Positifs (FP)   | Faux Négatifs (FN)   | Vrais Positifs (TP)   |
|:-----------------------|:---------------------|:---------------------|:----------------------|
| Âge moyen (années)     | 57.1                 | 42.9                 | 54.9                  |
| Diamètre moyen (mm)    | 135.4                | 123.2                | 143.8                 |
| Longueur moyenne (m)   | 20.3                 | 39.0                 | 20.7                  |
| Nombre de fuites moyen | 0.07                 | 0.25                 | 0.14                  |
| Score de risque moyen  | 0.697                | 0.313                | 0.811                 |
| Matériau dominant      | FTG                  | FT                   | FT                    |

### Insights Clés


1. **Faux Positifs (FP)**:
   - Le modèle prédit des défaillances pour 30,487 tronçons qui ne défaillent pas
   - Âge moyen: 57.1 ans
   - Impact: Surcoût de maintenance préventive inutile

2. **Faux Négatifs (FN)**:
   - Le modèle manque 216 défaillances réelles
   - Âge moyen: 42.9 ans
   - Impact: Risque de défaillances non anticipées

3. **Vrais Positifs (TP)**:
   - Le modèle détecte correctement 1,766 défaillances
   - Âge moyen: 54.9 ans
   - Performance: 89.1% des défaillances sont capturées

---

## 5. Recommandations

### Actions pour Réduire les Faux Positifs

1. **Affiner le seuil de décision**: Actuellement à 0.5, augmenter le seuil pour plus de spécificité
2. **Améliorer les features**: Les FP ont des caractéristiques qui les rendent similaires aux TP
3. **Calibration du modèle**: Améliorer la calibration des probabilités

### Actions pour Réduire les Faux Négatifs

1. **Enrichir les données**: Identifier les features manquantes qui caractérisent les FN
2. **Rééquilibrage**: Augmenter le poids des événements rares dans l'entraînement
3. **Analyse approfondie**: Comprendre pourquoi certains tronçons défaillants ne sont pas détectés

### Validation Business

- **Coût FP**: Inspection/renouvellement de 30,487 tronçons non nécessaires
- **Coût FN**: Défaillances non anticipées sur 216 tronçons
- **Ratio Coût FP/FN**: À évaluer selon les coûts de maintenance vs réparation d'urgence

---

## 6. Visualisations Générées

Les graphiques suivants ont été générés dans `reports/plots/`:

1. `confusion_matrix.png` - Matrice de confusion complète
2. `error_distribution.png` - Distribution des types d'erreurs
3. `errors_by_material.png` - Analyse détaillée par matériau
4. `errors_by_age.png` - Analyse par tranche d'âge
5. `errors_by_diameter.png` - Analyse par diamètre

---

*Rapport généré automatiquement le 2026-01-23 10:31:48*
