# Schema des donnees

## Table patrimoine (`v1_trafic_prepared.csv`)

| Colonne | Type | Obligatoire | Description |
|---------|------|-------------|-------------|
| `GID` | int | OUI | Identifiant unique du troncon |
| `DDP` | datetime (ISO) | OUI | Date de pose (installation) |
| `DHS` | datetime (ISO) | NON | Date hors service (null si en service) |
| `MAT` | string | OUI | Code materiau (FT, FTG, PEHD, PVC, BTM...) |
| `DIAMETRE` | float | OUI | Diametre en mm |
| `LNG` | float | OUI | Longueur en metres |
| `DDP_year` | int | NON | Annee de pose (derive de DDP) |
| `DHS_year` | int | NON | Annee hors service (derive de DHS) |
| `STATUT_OBJET` | string | NON | Statut (EN SERVICE, ABANDONNE...) |
| `age` | int | NON | Age en jours (calcule) |

### Colonnes geometrie (optionnelles)

Si disponibles, ces colonnes activent le mode carte :

| Colonne | Type | Description |
|---------|------|-------------|
| `geometry` | WKT string | Geometrie au format WKT (LineString ou MultiLineString) |
| `geom` | WKT string | Alternative pour geometrie WKT |
| `wkt` | WKT string | Alternative pour geometrie WKT |
| `latitude` / `longitude` | float | Coordonnees du centroide (mode points) |

## Table anomalies (`historiqueanomalie.csv`)

| Colonne | Type | Obligatoire | Description |
|---------|------|-------------|-------------|
| `GID_OBJET` | int | OUI | Identifiant du troncon (jointure avec GID) |
| `DATE_DETECTION_parsed` | datetime (ISO) | OUI | Date de detection de l'anomalie |
| `TYPE_ANOMALIE` | string | NON | Type (FUITE_DETECT_TR, FUITE_SIGNAL_TR...) |
| `annee_Anomalie` | int | NON | Annee de detection |
| `DATE_REPARATION` | string | NON | Date de reparation (texte) |
| `DATE_REPARATION_parsed` | datetime (ISO) | NON | Date de reparation (parsee) |
| `OBJET_DEPOSE_OU_ABANDONNE` | int | NON | Flag depose/abandonne |

## Table life stats (`life_stats_h{horizon}.csv`)

| Colonne | Type | Description |
|---------|------|-------------|
| `MAT` | string | Code materiau |
| `median_life` | float | Duree de vie mediane (annees) |
| `mean_life` | float | Duree de vie moyenne (annees) |
| `p25_life` | float | Percentile 25 duree de vie |
| `p75_life` | float | Percentile 75 duree de vie |
| `p90_life` | float | Percentile 90 duree de vie |
| `n_failed` | int | Nombre de defaillances observees |

## Features calculees (22 features)

### Features numeriques (20)

| Feature | Calcul | Source |
|---------|--------|--------|
| `age_at_freeze` | (freeze_date - DDP) / 365.25 | DDP |
| `diametre` | DIAMETRE | DIAMETRE |
| `longueur` | LNG | LNG |
| `longueur_km` | LNG / 1000 | LNG |
| `log_longueur` | log1p(LNG) | LNG |
| `log_diametre` | log1p(DIAMETRE) | DIAMETRE |
| `n_fuites_total` | count anomalies <= freeze | Anomalies |
| `n_fuites_1y` | count anomalies derniere annee | Anomalies |
| `n_fuites_3y` | count anomalies 3 dernieres annees | Anomalies |
| `n_fuites_5y` | count anomalies 5 dernieres annees | Anomalies |
| `days_since_last_fuite` | jours depuis derniere anomalie (cap 20 ans) | Anomalies |
| `has_recent_fuite` | days_since_last_fuite < 3*365 | Anomalies |
| `leak_rate_per_year` | n_fuites_total / age_at_freeze | Derive |
| `leak_rate_per_km` | n_fuites_total / longueur_km | Derive |
| `ratio_age_median` | age_at_freeze / median_life | Life stats |
| `overdue_years` | max(0, age - median_life) | Life stats |
| `over_p75_life` | age > p75_life (binaire) | Life stats |
| `over_p90_life` | age > p90_life (binaire) | Life stats |
| `age_x_nfuites` | age * n_fuites_total | Interaction |
| `surface_approx` | diametre * longueur | Interaction |

### Features categorielles (2)

| Feature | Calcul | Encodage |
|---------|--------|----------|
| `materiau` | MAT | LabelEncoder |
| `decade_install` | (DDP.year // 10) * 10 | LabelEncoder |

## Sortie du scoring

| Colonne | Type | Description |
|---------|------|-------------|
| `GID` | int | Identifiant du troncon |
| `risk_score` | float [0,1] | Score de risque (1 = plus risque) |
| `rank` | int | Rang (1 = plus risque) |
| `risk_category` | string | FAIBLE / MOYEN / ELEVE |
| `cost` | float | Cout proxy de renouvellement (EUR) |
| `selected` | bool | Selectionne dans le plan (export) |
