# Rapport de Qualité des Données
*Date de génération : 2026-02-03 11:45:44*

## 1. Vue d'ensemble des fichiers
| Fichier | Lignes | Colonnes | Taille mémoire |
|---------|--------|----------|----------------|
| v1_trafic_prepared.csv | 218,141 | 10 | 61.62 MB |
| commune.xlsx | 83,686 | 4 | 7.34 MB |
| TronconDT.xlsx | 210,875 | 5 | 17.47 MB |
| arretEau.xlsx | 22,401 | 19 | 4.53 MB |
| branchement.xlsx | 381,449 | 14 | 130.90 MB |
| Anomalie.xlsx | 61,637 | 17 | 21.70 MB |

## 2. Structure détaillée des tables

### 2.1 Table `troncons`
- **Dimensions**: 218,141 lignes × 10 colonnes
- **Types de données**: str: 4, float64: 4, int64: 2

| Colonne | Type | Non-null | % Manquant | Exemple |
|---------|------|----------|------------|----------|
| GID | int64 | 218,141 | 0.0% | 410966779 |
| DDP | str | 218,141 | 0.0% | 2009-01-01T00:00:00.000Z |
| DDP_year | int64 | 218,141 | 0.0% | 2009 |
| DHS | str | 38,604 | 82.3% | 2004-12-31T23:00:00.000Z |
| DHS_year | float64 | 38,604 | 82.3% | 2004.0 |
| MAT | str | 218,141 | 0.0% | FT |
| DIAMETRE | float64 | 218,076 | 0.0% | 100.0 |
| STATUT_OBJET | str | 218,141 | 0.0% | EN SERVICE |
| LNG | float64 | 218,141 | 0.0% | 82.726 |
| age | float64 | 215,798 | 1.1% | 6188.0 |


### 2.2 Table `commune`
- **Dimensions**: 83,686 lignes × 4 colonnes
- **Types de données**: float64: 3, str: 1

| Colonne | Type | Non-null | % Manquant | Exemple |
|---------|------|----------|------------|----------|
| GID_TRONCON | float64 | 83,670 | 0.0% | 410047802.0 |
| GID_COMMUNE | float64 | 83,684 | 0.0% | 20000614.0 |
| GID_RESEAU | float64 | 83,684 | 0.0% | 400000006.0 |
| NATURE_RESEAU | str | 83,684 | 0.0% | EAU POTABLE |


### 2.3 Table `troncon_dt`
- **Dimensions**: 210,875 lignes × 5 colonnes
- **Types de données**: float64: 3, int64: 1, str: 1

| Colonne | Type | Non-null | % Manquant | Exemple |
|---------|------|----------|------------|----------|
| DT_GID_TRONCON | int64 | 210,875 | 0.0% | 410001208 |
| DT_NB_LOGEMENT | float64 | 162,360 | 23.0% | 1.0 |
| DT_NB_ABONNE | float64 | 162,360 | 23.0% | 2.0 |
| DT_FLUX_CIRCULATION | float64 | 111,909 | 46.9% | 3.0 |
| DT_MATERIAU_PROPOSE_TR | str | 174,832 | 17.1% | FT |


### 2.4 Table `arret_eau`
- **Dimensions**: 22,401 lignes × 19 colonnes
- **Types de données**: float64: 9, datetime64[us]: 6, int64: 3, str: 1

| Colonne | Type | Non-null | % Manquant | Exemple |
|---------|------|----------|------------|----------|
| GID | int64 | 22,401 | 0.0% | 690512578 |
| GID_RESEAU | int64 | 22,401 | 0.0% | 400000003 |
| NATURE_RESEAU | str | 22,401 | 0.0% | EAU POTABLE |
| GID_COMMUNE | int64 | 22,401 | 0.0% | 20000613 |
| DATE_PREVUE_DEBUT | datetime64[us] | 22,401 | 0.0% | 2011-01-10 05:00:00 |
| DATE_PREVUE_FIN | datetime64[us] | 22,401 | 0.0% | 2011-01-10 14:00:00 |
| DATE_PREVUE_FIN_INITIALE | datetime64[us] | 3,549 | 84.2% | 2013-02-21 17:00:00 |
| PROLONGATION_STATUT | float64 | 0 | 100.0% | N/A |
| PROLONGATION_DATE | float64 | 0 | 100.0% | N/A |
| DATE_EXECUTION_DEBUT | datetime64[us] | 19,807 | 11.6% | 2011-01-10 05:45:00 |
| DATE_EXECUTION_FIN | datetime64[us] | 19,908 | 11.1% | 2011-01-10 11:50:00 |
| DIAMETRE_PREMIER_TRONCON | float64 | 20,979 | 6.3% | 100.0 |
| DIAMETRE_MAXI | float64 | 13,942 | 37.8% | 150.0 |
| TAMPONNE | float64 | 40 | 99.8% | 0.0 |
| REMISE_EN_SERVICE | float64 | 0 | 100.0% | N/A |
| NBRE_PT_DESSERTE | float64 | 17,319 | 22.7% | 2.0 |
| VOLUME_DE_SERVICE | float64 | 4 | 100.0% | 0.0 |
| MAJ_DATE_HEURE | datetime64[us] | 22,401 | 0.0% | 2014-06-14 10:37:32 |
| NB_LITRE_EAU_A_DISTRIBUER | float64 | 104 | 99.5% | 90.0 |


### 2.5 Table `branchement`
- **Dimensions**: 381,449 lignes × 14 colonnes
- **Types de données**: float64: 6, str: 5, int64: 2, datetime64[us]: 1

| Colonne | Type | Non-null | % Manquant | Exemple |
|---------|------|----------|------------|----------|
| GID | int64 | 381,449 | 0.0% | 510006083 |
| NATURE_RESEAU | str | 381,449 | 0.0% | EAU POTABLE |
| SECTEUR | str | 333,350 | 12.6% | SAUSSET1 |
| GID_COMMUNE | int64 | 381,449 | 0.0% | 20000657 |
| GID_VOIE | float64 | 381,446 | 0.0% | 50013015.0 |
| TYPE_OBJET | str | 373,873 | 2.0% | ABONNE_TU |
| DIAMETRE | float64 | 327,801 | 14.1% | 32.0 |
| MATERIAU | str | 327,434 | 14.2% | POLY |
| LONGUEUR_SIG | float64 | 381,447 | 0.0% | 3.469 |
| LONGUEUR_MESUREE | float64 | 14,532 | 96.2% | 4.0 |
| GID_AMONT | float64 | 335,483 | 12.1% | 410082466.0 |
| GID_AVAL | float64 | 333,171 | 12.7% | 610801511.0 |
| DATE_RENOVATION | datetime64[us] | 58 | 100.0% | 2022-10-06 00:00:00 |
| ETAT | str | 12 | 100.0% | OUVERT |


### 2.6 Table `anomalie`
- **Dimensions**: 61,637 lignes × 17 colonnes
- **Types de données**: float64: 8, str: 5, datetime64[us]: 3, int64: 1

| Colonne | Type | Non-null | % Manquant | Exemple |
|---------|------|----------|------------|----------|
| GID_OBJET | int64 | 61,637 | 0.0% | 510930406 |
| TYPE_ANOMALIE | str | 61,637 | 0.0% | FUITE_DETECT_BR |
| GID_VOIE | float64 | 60,179 | 2.4% | 50003139.0 |
| NO_VOIRIE | str | 12,250 | 80.1% | 24 |
| LOCALISATION | str | 32,845 | 46.7% | FENTE |
| CARACTERISATION | str | 32,341 | 47.5% | SUITE_AE |
| DATE_DETECTION | datetime64[us] | 61,637 | 0.0% | 2016-10-12 00:00:00 |
| DATE_REPARATION | datetime64[us] | 50,824 | 17.5% | 2016-10-12 00:00:00 |
| DATE_FIN_FUITE | datetime64[us] | 16,930 | 72.5% | 2016-10-12 00:00:00 |
| OBSERVATIONS | str | 25,088 | 59.3% | FRS
CONDUITE
FRS
 |
| GID_CHANTIER | float64 | 35,508 | 42.4% | 100013503.0 |
| CAUSE_FUITE | float64 | 0 | 100.0% | N/A |
| DIAMETRE_FUITE | float64 | 0 | 100.0% | N/A |
| MATERIAUX_FUITE | float64 | 0 | 100.0% | N/A |
| PROFONDEUR_FUITE | float64 | 0 | 100.0% | N/A |
| ETAT_FUITE | float64 | 0 | 100.0% | N/A |
| DOMAINE | float64 | 0 | 100.0% | N/A |

## 3. Clés de jointure identifiées
| Clé | Tables concernées | Cardinalités |
|-----|-------------------|---------------|
| GID | troncons, arret_eau, branchement | troncons: 218,141, arret_eau: 22,401, branchement: 381,449 |
| GID_TRONCON | commune | commune: 71,817 |
| DT_GID_TRONCON | troncon_dt | troncon_dt: 210,875 |
| GID_OBJET | anomalie | anomalie: 38,122 |
| GID_COMMUNE | commune, arret_eau, branchement | commune: 97, arret_eau: 89, branchement: 110 |
| GID_VOIE | branchement, anomalie | branchement: 20,738, anomalie: 11,933 |
| GID_RESEAU | commune, arret_eau | commune: 391, arret_eau: 286 |

## 4. Analyse des valeurs manquantes

### Table `troncons` - Colonnes avec valeurs manquantes
| Colonne | Manquants | % |
|---------|-----------|---|
| DHS_year | 179,537 | 82.3% |
| DHS | 179,537 | 82.3% |
| age | 2,343 | 1.1% |
| DIAMETRE | 65 | 0.0% |

### Table `commune` - Colonnes avec valeurs manquantes
| Colonne | Manquants | % |
|---------|-----------|---|
| GID_TRONCON | 16 | 0.0% |

### Table `troncon_dt` - Colonnes avec valeurs manquantes
| Colonne | Manquants | % |
|---------|-----------|---|
| DT_FLUX_CIRCULATION | 98,966 | 46.9% |
| DT_NB_ABONNE | 48,515 | 23.0% |
| DT_NB_LOGEMENT | 48,515 | 23.0% |
| DT_MATERIAU_PROPOSE_TR | 36,043 | 17.1% |

### Table `arret_eau` - Colonnes avec valeurs manquantes
| Colonne | Manquants | % |
|---------|-----------|---|
| REMISE_EN_SERVICE | 22,401 | 100.0% |
| PROLONGATION_STATUT | 22,401 | 100.0% |
| PROLONGATION_DATE | 22,401 | 100.0% |
| VOLUME_DE_SERVICE | 22,397 | 100.0% |
| TAMPONNE | 22,361 | 99.8% |
| NB_LITRE_EAU_A_DISTRIBUER | 22,297 | 99.5% |
| DATE_PREVUE_FIN_INITIALE | 18,852 | 84.2% |
| DIAMETRE_MAXI | 8,459 | 37.8% |
| NBRE_PT_DESSERTE | 5,082 | 22.7% |
| DATE_EXECUTION_DEBUT | 2,594 | 11.6% |
| DATE_EXECUTION_FIN | 2,493 | 11.1% |
| DIAMETRE_PREMIER_TRONCON | 1,422 | 6.3% |

### Table `branchement` - Colonnes avec valeurs manquantes
| Colonne | Manquants | % |
|---------|-----------|---|
| ETAT | 381,437 | 100.0% |
| DATE_RENOVATION | 381,391 | 100.0% |
| LONGUEUR_MESUREE | 366,917 | 96.2% |
| MATERIAU | 54,015 | 14.2% |
| DIAMETRE | 53,648 | 14.1% |
| GID_AVAL | 48,278 | 12.7% |
| SECTEUR | 48,099 | 12.6% |
| GID_AMONT | 45,966 | 12.1% |
| TYPE_OBJET | 7,576 | 2.0% |

### Table `anomalie` - Colonnes avec valeurs manquantes
| Colonne | Manquants | % |
|---------|-----------|---|
| MATERIAUX_FUITE | 61,637 | 100.0% |
| CAUSE_FUITE | 61,637 | 100.0% |
| PROFONDEUR_FUITE | 61,637 | 100.0% |
| DOMAINE | 61,637 | 100.0% |
| ETAT_FUITE | 61,637 | 100.0% |
| DIAMETRE_FUITE | 61,637 | 100.0% |
| NO_VOIRIE | 49,387 | 80.1% |
| DATE_FIN_FUITE | 44,707 | 72.5% |
| OBSERVATIONS | 36,549 | 59.3% |
| CARACTERISATION | 29,296 | 47.5% |
| LOCALISATION | 28,792 | 46.7% |
| GID_CHANTIER | 26,129 | 42.4% |
| DATE_REPARATION | 10,813 | 17.5% |
| GID_VOIE | 1,458 | 2.4% |

## 5. Analyse des doublons
| Table | Lignes | Doublons exacts | % |
|-------|--------|-----------------|---|
| troncons | 218,141 | 0 | 0.00% |
| commune | 83,686 | 11,856 | 14.17% |
| troncon_dt | 210,875 | 0 | 0.00% |
| arret_eau | 22,401 | 0 | 0.00% |
| branchement | 381,449 | 0 | 0.00% |
| anomalie | 61,637 | 641 | 1.04% |

## 6. Analyse des colonnes de dates

### Colonne DDP (Date De Pose) - troncons
- Valides: 218,141
- Plage: 1900-01-01 00:00:00+00:00 → 2024-11-04 00:00:00+00:00
- Dates aberrantes: 0

### Colonne DATE_DETECTION - anomalie
- Valides: 61,637
- Plage: 1899-12-29 00:00:00 → 2024-11-07 23:49:23

### Colonne DATE_REPARATION - anomalie
- Valides: 50,824
- Plage: 1899-12-29 00:00:00 → 2024-11-07 18:39:49

### Colonne DATE_FIN_FUITE - anomalie
- Valides: 16,930
- Plage: 1899-12-29 00:00:00 → 2312-08-11 12:00:00

### Colonne DATE_PREVUE_DEBUT - arret_eau
- Valides: 22,401
- Plage: 1900-01-01 00:00:00 → 2024-12-19 09:00:00

### Colonne DATE_EXECUTION_DEBUT - arret_eau
- Valides: 19,807
- Plage: 2001-12-05 16:00:00 → 2024-11-07 15:00:00

### Colonne DATE_EXECUTION_FIN - arret_eau
- Valides: 19,908
- Plage: 2011-01-04 11:10:00 → 2024-11-07 16:30:00

## 7. Dictionnaire de données

### Table `troncons`
| Colonne | Description |
|---------|-------------|
| GID | Identifiant unique du tronçon |
| DDP | Date De Pose du tronçon |
| DDP_year | Année de pose |
| DHS | Date Hors Service |
| DHS_year | Année de mise hors service |
| MAT | Matériau de la canalisation (FT=Fonte, AC=Acier, PVC, etc.) |
| DIAMETRE | Diamètre intérieur en mm |
| STATUT_OBJET | Statut du tronçon (EN SERVICE, HORS SERVICE) |
| LNG | Longueur du tronçon en mètres |
| age | Âge du tronçon en jours |

### Table `commune`
| Colonne | Description |
|---------|-------------|
| GID_TRONCON | Identifiant du tronçon (clé de jointure) |
| GID_COMMUNE | Identifiant de la commune |
| GID_RESEAU | Identifiant du réseau |
| NATURE_RESEAU | Type de réseau (EP=Eau Potable, EU=Eaux Usées) |

### Table `troncon_dt`
| Colonne | Description |
|---------|-------------|
| DT_GID_TRONCON | Identifiant du tronçon |
| DT_NB_LOGEMENT | Nombre de logements desservis |
| DT_NB_ABONNE | Nombre d'abonnés |
| DT_FLUX_CIRCULATION | Indice de trafic routier |
| DT_MATERIAU_PROPOSE_TR | Matériau proposé pour renouvellement |

### Table `anomalie`
| Colonne | Description |
|---------|-------------|
| GID_OBJET | Identifiant du tronçon concerné |
| TYPE_ANOMALIE | Type d'anomalie (FUITE, CASSE, etc.) |
| DATE_DETECTION | Date de détection de l'anomalie |
| DATE_REPARATION | Date de réparation |
| DATE_FIN_FUITE | Date de fin de la fuite |
| CAUSE_FUITE | Cause identifiée de la fuite |
| CARACTERISATION | Caractérisation de l'anomalie |
| OBSERVATIONS | Notes et observations |

### Table `arret_eau`
| Colonne | Description |
|---------|-------------|
| GID | Identifiant de l'arrêt d'eau |
| DATE_PREVUE_DEBUT | Date prévue de début d'intervention |
| DATE_EXECUTION_DEBUT | Date réelle de début |
| DATE_EXECUTION_FIN | Date réelle de fin |
| DIAMETRE_PREMIER_TRONCON | Diamètre du premier tronçon |
| NBRE_PT_DESSERTE | Nombre de points de desserte affectés |

### Table `branchement`
| Colonne | Description |
|---------|-------------|
| GID | Identifiant du branchement |
| DIAMETRE | Diamètre du branchement |
| MATERIAU | Matériau du branchement |
| LONGUEUR_SIG | Longueur SIG |
| LONGUEUR_MESUREE | Longueur mesurée sur terrain |
| GID_AMONT | Tronçon amont |
| GID_AVAL | Tronçon aval |
