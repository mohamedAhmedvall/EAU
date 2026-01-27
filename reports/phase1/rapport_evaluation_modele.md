# Rapport d'évaluation du modèle de priorisation (Phase 1)
*Généré le 2026-01-27 – modèle LightGBM horizon 1 an, freeze_date = 2023-11-07 – aucun ré-entraînement effectué*

---

## 1. Résumé exécutif (métier)
- **À quoi ça sert** : classer les tronçons AEP selon leur risque de défaillance à 12 mois pour orienter les renouvellements annuels.
- **Décisions possibles aujourd’hui** : cibler le **top 10 %** des tronçons (3 574 sur 35 739 test) pour capturer **60 %** des défaillances attendues, ordonner les chantiers par score.
- **Ce que ça ne permet pas encore** : arbitrer seul les choix budgétaires, ni garantir une probabilité calibrée de casse tronçon par tronçon; segments PEHD/FTVI/PVC sous-couverts.
- **Pertinence pour l’optimisation** : fournit un **score de risque monotone** et un **lift x6** à 10 %, exploitable comme bénéfice dans un solveur sous contraintes (risque traité = score × longueur × coût).

## 2. Rappel du problème & méthodologie
- Problème de **priorisation d’événements rares** avec censure : événement = DHS dans les 12 mois suivant la **freeze_date** (07/11/2023).
- **Population éligible** : tronçons posés avant freeze_date et encore en service (DHS > freeze_date ou vide).
- **Label (event)** : 1 si DHS ∈ (freeze_date, freeze_date + 1 an], sinon 0.
- **Validation anti-leakage** : feature engineering strictement ≤ freeze_date, contrôles dans `dataset_builder.py` (anomalies, DHS, survie) et reconstruction du split test identique à l’entraînement (80/20 stratifié, seed 42).

## 3. Performance globale du modèle
- **ROC-AUC** : 0.866 (utile pour hiérarchie globale mais peu informatif métier à cause du déséquilibre 1.1 %).
- **Top-k business**

| K | Capture@K | Precision@K | Lift@K |
|---|-----------|-------------|--------|
| 5 % | 44.4 % | 9.8 % | 8.89 |
| 10 % | 60.1 % | 6.66 % | 6.01 |
| 20 % | 76.0 % | 4.21 % | 3.80 |

Graphiques : courbe de gain cumulative (`reports/phase1/plots/gain_curve.png`), courbe de lift (`.../lift_curve.png`), distribution globale des scores (`.../score_distribution.png`).

## 4. Matrice de confusion & erreurs métier (TOP 10 %)
- **TP 238** : tronçons priorisés qui défaillent (capture 60 % des casses).
- **FP 3 336** : tronçons priorisés qui ne défaillent pas (coût d’opportunité/inspection).
- **FN 158** : tronçons non priorisés qui défaillent (risque résiduel).
- **TN 32 007** : tronçons non priorisés qui ne défaillent pas.
- Precision@10 % = 6.7 %, Recall@10 % = 60.1 %. Cette granularité reflète la rareté de l’événement; le modèle sert à **classer**, pas à décider isolément.

## 5. Distribution des erreurs (obligatoire)
- **TP vs FN** : les FN se concentrent sur des scores plus bas (histogramme `score_errors.png`), confirmant une séparation partielle mais non parfaite des événements.
- **FP vs TN** : FP situés dans la partie médiane/haute des scores; vigilance sur segments peu fréquents.
- **Résidus (y − score)** : distribution centrée légèrement négative (`residuals.png`), indiquant une tendance à sous-prédire la probabilité brute (cohérent avec faible base rate).
- **Déciles de score** : taux de casse passe de **0.03 % (décile 1, scores les plus bas)** à **6.66 % (décile 10, scores les plus élevés)**. Capture cumulée atteint 60 % au décile 10 et 76 % au décile 9 (`decile_event_rate.png`). Monotonie globalement respectée (légère irrégularité sur déciles 3–4 mais sans inversion majeure).

## 6. Analyse par segments métier (Capture@10 % / FN rate)
- **Matériau** (`segment_capture_materiau.png`) :
  - FT : 67 % capture (207 événements), bonne discrimination.
  - FTG : 56 % capture.
  - PEHD, FTVI, PVC : 0 % capture (tous les événements passent en FN) → segments sous-appris à surveiller.
- **Décennie de pose** (`segment_capture_decade.png`) :
  - 1960–1970 : ~70 % capture, cohérent avec vieillissement.
  - 1940 : 56 % capture, vieillissement extrême mais moins bien capté.
  - 2010–2020 : capture très faible (≤25 %) mais event_rate <0.5 %, impact limité.
- **Diamètre** (`segment_capture_diametre.png`) :
  - 300 mm : 77 % capture.
  - 80–150 mm : 53–66 % capture.
  - 200 mm : 45 % capture (sur-estimation possible des grands diamètres hors 300).
- **Longueur** (`segment_capture_longueur.png`) :
  - <50 m : 66 % capture (bulk du réseau).
  - 50–300 m : ~23 % capture, peu d’événements mais risque de FN sur tronçons longs dispersés.

## 7. Explicabilité & cohérence métier
- **Importances LightGBM (top 5)** : longueur, âge, surface_approx (diamètre×longueur), diamètre, ratio_age_median (`feature_importance.png` / `tables/feature_importance.csv`).
- **Signes attendus** : facteurs d’âge, longueur et historiques de fuite (leak_rate) sortent parmi les plus explicatifs → cohérent avec la physique réseau (“les fuites appellent les fuites”, vieillissement).
- **Points rassurants** : dominance de variables structurelles stables (âge, diamètre, longueur) limite le drift court terme.
- **Points de vigilance** : certaines features fuite (n_fuites_1y/3y/5y, has_recent_fuite) ont une importance faible ou nulle, possiblement à cause de sparsité; segments sans capture (PEHD/FTVI) peuvent cacher des effets non appris.

## 8. Intérêt du modèle pour l’optimisation
- Le score est un **rang de risque** à maximiser. Utilisable comme bénéfice dans un solveur : `bénéfice = score × longueur (ou coût évité)`.
- **Calibration** : precision@10 % = 6.7 % → ne pas interpréter le score comme probabilité absolue; utiliser le rang ou normaliser.
- **Stabilité** : lift@K élevé et courbe de gain régulière assurent un comportement monotone requis pour des arbitrages budgétaires; surveiller la stabilité par segment avant d’imposer des contraintes (ex. PEHD).
- **Limites avant optimisation** : segments non couverts, faible capture sur tronçons longs 50–300 m, absence de prise en compte des coûts/chantiers; la fonction objectif devra intégrer ces incertitudes.

## 9. Limites identifiées (sans solution)
- **Biais de survivants** : tronçons très anciens encore en service (1940–1950) partiellement sous-capturés.
- **Segments faibles (PEHD/FTVI/PVC)** : aucun événement capturé à 10 %, risque de sous-estimation systémique.
- **Dépendance forte à la longueur/âge** : possible sur-priorisation de tronçons longs/anciens au détriment d’autres facteurs locaux.
- **Probabilités non calibrées** : precision basse due au taux d’événement 1.1 % → score à utiliser en relatif uniquement.
- **Risques d’interprétation** : FP élevés peuvent générer des interventions non nécessaires si le score est pris comme certitude.

## 10. Conclusion
- **Forces** : lift@10 % = 6.0, capture 60 % des casses sur 10 % du linéaire, courbe de gain monotone, explicabilité alignée avec les mécanismes physiques.
- **Limites** : calibration absolue faible, segments PEHD/FTVI/PVC non maîtrisés, couverture moyenne des tronçons longs intermédiaires.
- **Go / No-Go pour optimisation sous contrainte** : **Go conditionnel** – modèle apte à fournir un rang de risque fiable pour un solveur, à condition d’exposer ses zones d’incertitude (segments faibles) et de garder le score en relatif.
- **Recommandation** : utiliser ce score comme entrée du moteur d’optimisation, tout en appliquant des garde-fous métier sur les segments sous-appris et en validant la solution sur un échantillon terrain.

---

### Références de sortie
- Figures : `reports/phase1/plots/*.png`
- Tables : `reports/phase1/tables/*.csv`, `summary.json`
- Script de reproduction : `src/phase1_model_evaluation.py` (aucune modification du modèle, simple réévaluation)
