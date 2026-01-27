# Rapport d'Analyse des Anomalies et Corrélation avec Mise Hors Service

*Généré le 2026-01-23 11:07:31*

---

## 1. Vue d'Ensemble des Données

### Statistiques Globales

- **Total d'anomalies**: 30,775
- **Tronçons concernés**: 14,276
- **Tronçons avec info complète**: 28,504
- **Anomalies menant à mise hors service**: 10,791 (37.9%)

### Période Couverte

- **Date première anomalie**: 1960-01-01
- **Date dernière anomalie**: 2033-05-26
- **Années de données**: 73.4 ans

---

## 2. Classification des Anomalies par Sévérité

### Distribution

- **CRITIQUE**: 30,138 anomalies (97.9%)
- **AUTRE**: 637 anomalies (2.1%)

### Définition des Niveaux de Sévérité

1. **CRITIQUE**: Anomalies nécessitant une intervention immédiate
   - Fuites détectées, ruptures, cassures, éclatements
   - Impact immédiat sur le service

2. **MAJEURE**: Anomalies avec risque élevé de défaillance
   - Fuites importantes, affaissements, déformations, corrosion avancée
   - Nécessite intervention rapide

3. **MOYENNE**: Anomalies nécessitant surveillance
   - Fuites légères, corrosion, usure
   - Intervention programmée

4. **MINEURE**: Maintenance préventive
   - Fuites accessoires, défauts mineurs
   - Surveillance régulière

---

## 3. Corrélation Anomalies ↔ Mise Hors Service

### Taux de Mise Hors Service par Sévérité

- **CRITIQUE**: 37.9% (10,791/30,138 tronçons)
- **AUTRE**: 0.0% (0/637 tronçons)

### Délai entre Anomalie et Mise Hors Service

Sur les 10,594 anomalies ayant conduit à une mise hors service avec délai mesurable:

- **Délai moyen**: 11.83 ans
- **Délai médian**: 6.89 ans
- **Écart-type**: 12.27 ans
- **Minimum**: 0.00 ans
- **Maximum**: 58.23 ans

#### Délai Moyen par Sévérité

- **CRITIQUE**: 11.83 ans (médiane: 6.89 ans) | n=10,594

### Insights Clés

1. **Les anomalies critiques** ont le délai le plus court avant mise hors service
2. **Corrélation forte** entre sévérité et probabilité de mise hors service
3. **Les fuites détectées** sont les anomalies les plus fréquentes et critiques

---

## 4. Top Types d'Anomalies

### Top 10 par Fréquence

1. **FUITE_SIGNAL_TR**: 25,768 (83.7%)
2. **FUITE_DETECT_TR**: 3,951 (12.8%)
3. **FUITE**: 364 (1.2%)
4. **DEFAUT_PRESSION**: 61 (0.2%)
5. **INA_VENTMANU**: 43 (0.1%)
6. **PRBREGARD_VIDBORGNE**: 39 (0.1%)
7. **PRBREGARD_VIDANGE**: 38 (0.1%)
8. **PRBTAMPON_VIDANGE**: 38 (0.1%)
9. **NONMAN_VENTOUSE**: 35 (0.1%)
10. **ENGORGE_VAIR200**: 23 (0.1%)

### Top 10 par Taux de Mise Hors Service

1. **FUITE_SIGNAL_TR**: 10085 HS (41.0%) sur 24622 total
2. **FUITE_DETECT_TR**: 706 HS (18.2%) sur 3880 total
3. **DEFAUT_PRESSION**: 0 HS (nan%) sur 0 total
4. **PRBTAMPON_VIDRACC**: 0 HS (nan%) sur 0 total
5. **PRBTAMPON_VIDANGE**: 0 HS (nan%) sur 0 total
6. **PRBTAMPON_VENTMANU**: 0 HS (nan%) sur 0 total
7. **PRBTAMPON_VAIR1000**: 0 HS (nan%) sur 0 total
8. **PRBTAMPON_HYDROSTAB**: 0 HS (nan%) sur 0 total
9. **PRBREGARD_VIDRACC**: 0 HS (nan%) sur 0 total
10. **PRBREGARD_VIDBORGNE**: 0 HS (nan%) sur 0 total

---

## 5. Analyse par Matériau

### Taux de Mise Hors Service

- **FTG**: 4948 HS (44.1%) sur 11228 anomalies
- **PEHD**: 2266 HS (35.4%) sur 6395 anomalies
- **POLY**: 2112 HS (52.9%) sur 3993 anomalies
- **FT**: 919 HS (18.7%) sur 4924 anomalies
- **PVC**: 226 HS (30.1%) sur 751 anomalies
- **ACIE**: 125 HS (24.9%) sur 503 anomalies
- **BTM**: 91 HS (33.2%) sur 274 anomalies
- **PB**: 32 HS (25.0%) sur 128 anomalies
- **VP**: 24 HS (72.7%) sur 33 anomalies
- **A.C**: 21 HS (14.4%) sur 146 anomalies

---

## 6. Recommandations

### Actions Prioritaires

1. **Surveillance renforcée** des tronçons avec anomalies critiques
   - Délai moyen très court avant défaillance
   - Intervention préventive recommandée

2. **Programme de maintenance préventive**
   - Cibler les matériaux à risque élevé
   - Planifier interventions selon sévérité

3. **Système d'alerte**
   - Créer des alertes automatiques pour anomalies critiques
   - Suivi des délais par sévérité

4. **Priorisation des renouvellements**
   - Intégrer les anomalies dans le score de risque
   - Pondérer par sévérité et délai historique

### Optimisation du Modèle de Prédiction

Les anomalies historiques peuvent enrichir le modèle:
- **Feature engineering**: Nombre d'anomalies par sévérité
- **Pondération**: Anomalies critiques comme prédicteur fort
- **Temporalité**: Délai depuis dernière anomalie

---

## 7. Fichiers Générés

Les visualisations suivantes ont été créées dans `reports/plots/`:

1. `anomaly_severity_distribution.png` - Distribution par sévérité
2. `failure_rate_by_severity.png` - Taux de mise HS par sévérité
3. `delay_to_failure_analysis.png` - Analyse temporelle détaillée
4. `top_anomaly_types.png` - Types d'anomalies principaux

---

*Rapport généré automatiquement par l'analyse des anomalies*
