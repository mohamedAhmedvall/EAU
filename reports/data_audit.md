# Rapport d'Audit des Données

*Généré le: 2026-01-19 19:31:42*


## 1. Résumé Exécutif

- **Table Patrimoine (v1_trafic_prepared.csv)**: 218,141 tronçons
- **Table Anomalies (historiqueanomalie.csv)**: 30,775 anomalies
- **Tronçons avec au moins une anomalie**: 12,763 (5.9%)
- **Date maximale d'observation**: 2024-11-07
- **Taux de tronçons avec DHS (défaillants/abandonnés)**: 38,604 (17.7%)

## 2. Table Patrimoine (v1_trafic_prepared.csv)

### 2.1 Structure
- Nombre de lignes: 218,141
- Nombre de colonnes: 12
- Colonnes: `GID, DDP, DDP_year, DHS, DHS_year, MAT, DIAMETRE, STATUT_OBJET, LNG, age, DDP_parsed, DHS_parsed`
- Identifiants uniques (GID): 218,141
- Doublons sur GID: 0

### 2.2 Dates
**Date de Pose (DDP):**
- Valides: 218,141
- Manquantes: 0
- Plage: 1900-01-01 → 2024-11-04
- Dates futures (implausibles): 0

**Date Hors Service (DHS):**
- Valides: 38,604
- Manquantes (= en service): 179,537
- Plage: 1899-12-31 → 2024-11-04
- Dates futures (implausibles): 0

**Incohérences:**
- DDP après DHS: 1321

### 2.3 Matériaux
| Matériau | Nombre | % |
|----------|--------|---|
| FT | 111,025 | 50.9% |
| FTG | 61,082 | 28.0% |
| FTVI | 12,307 | 5.6% |
| PEHD | 8,934 | 4.1% |
| POLY | 8,152 | 3.7% |
| AUTRE | 5,681 | 2.6% |
| PVC | 4,632 | 2.1% |
| BTM | 2,393 | 1.1% |
| ACIE | 1,859 | 0.9% |
| A.C | 486 | 0.2% |
| FTTT | 318 | 0.1% |
| BIOR | 276 | 0.1% |
| FTBLU | 180 | 0.1% |
| PB | 160 | 0.1% |
| FER | 141 | 0.1% |
| PRV | 107 | 0.0% |
| FTTTVI | 101 | 0.0% |
| VP | 80 | 0.0% |
| BA | 60 | 0.0% |
| PEBD | 47 | 0.0% |
| AFCO | 41 | 0.0% |
| COMP | 32 | 0.0% |
| CENT | 15 | 0.0% |
| GALV | 13 | 0.0% |
| INOX | 13 | 0.0% |
| GRES | 2 | 0.0% |
| CUIT | 2 | 0.0% |
| GALERIE | 2 | 0.0% |

### 2.4 Statuts
| Statut | Nombre | % |
|--------|--------|---|
| EN SERVICE | 179,533 | 82.3% |
| ABANDONNE | 38,604 | 17.7% |
| CHANTIER | 4 | 0.0% |

### 2.5 Caractéristiques Physiques
**Diamètre (mm):**
- Min: 0.0
- Max: 3200.0
- Moyenne: 143.0
- Médiane: 100.0

**Longueur (m):**
- Min: 0.0
- Max: 5701102.0
- Moyenne: 81.1
- Total réseau: 17684.4 km

## 3. Table Anomalies (historiqueanomalie.csv)

### 3.1 Structure
- Nombre de lignes (anomalies): 30,775
- Tronçons uniques concernés: 14,276
- Lignes dupliquées exactes: 1286

### 3.2 Dates de détection
- Valides: 30,775
- Manquantes: 0
- Plage: 1960-01-01 → 2033-05-26
- Dates futures (implausibles): 1

### 3.3 Types d'anomalies
| Type | Nombre | % |
|------|--------|---|
| FUITE_SIGNAL_TR | 25,768 | 83.7% |
| FUITE_DETECT_TR | 3,951 | 12.8% |
| FUITE | 364 | 1.2% |
| DEFAUT_PRESSION | 61 | 0.2% |
| INA_VENTMANU | 43 | 0.1% |
| PRBREGARD_VIDBORGNE | 39 | 0.1% |
| PRBREGARD_VIDANGE | 38 | 0.1% |
| PRBTAMPON_VIDANGE | 38 | 0.1% |
| NONMAN_VENTOUSE | 35 | 0.1% |
| ENGORGE_VAIR200 | 23 | 0.1% |
| ENGORGE_VENTOUSE | 22 | 0.1% |
| FUITE_VIDANGE | 21 | 0.1% |
| NONMAN_HYDRANT | 20 | 0.1% |
| VETUSTE_VAIR200 | 18 | 0.1% |
| HS_VENTMANU | 15 | 0.0% |
| VETUSTE_EQUIPPUB | 13 | 0.0% |
| ENGORGE_VIDBORGNE | 12 | 0.0% |
| HS_VAIR500 | 12 | 0.0% |
| PRESENCE_AIR | 12 | 0.0% |
| OBSTRUCTION | 11 | 0.0% |
| HS_VAIR1000 | 11 | 0.0% |
| INTROU_EQUIPPUB | 10 | 0.0% |
| INTROU_VAIR1000 | 10 | 0.0% |
| NONMAN_VAIR200 | 10 | 0.0% |
| NONMAN_EQUIPPUB | 8 | 0.0% |
| INA_VIDRACC | 8 | 0.0% |
| ENTRETIEN_CRT_REGARD | 8 | 0.0% |
| PRBREGARD_VIDRACC | 7 | 0.0% |
| REVETSOL_VENTOUSE | 7 | 0.0% |
| VOL_EAU | 7 | 0.0% |
| NONMAN_VIDBORGNE | 7 | 0.0% |
| PRBTAMPON_VENTMANU | 7 | 0.0% |
| INTROU_HYDRANT | 7 | 0.0% |
| FUITE_MONOVAR | 6 | 0.0% |
| PRBTAMPON_VIDBORGNE | 6 | 0.0% |
| FUITE_VAIR500 | 6 | 0.0% |
| INA_VAIR500 | 5 | 0.0% |
| INA_EQUIPPUB | 5 | 0.0% |
| VETUSTE_VIDANGE | 5 | 0.0% |
| ENGORGE_VENTMANU | 5 | 0.0% |
| PRBREGARD_HYDROSTAB | 5 | 0.0% |
| PRBREGARD_VAIR1000 | 5 | 0.0% |
| FUITE_VIDBORGNE | 5 | 0.0% |
| VETUSTE_SECMESDEB | 4 | 0.0% |
| FUITE_SECMESDEB | 4 | 0.0% |
| NONMAN_VENTMANU | 4 | 0.0% |
| REVETSOL_VIDANGE | 4 | 0.0% |
| VETUSTE_MONOVAR | 4 | 0.0% |
| VETUSTE_VENTMANU | 4 | 0.0% |
| FUITE_DETENDEUR | 4 | 0.0% |
| PRBTAMPON_HYDROSTAB | 4 | 0.0% |
| FUITE_VAIR1000 | 3 | 0.0% |
| HS_VIDBORGNE | 3 | 0.0% |
| FUITE_VENTMANU | 3 | 0.0% |
| ENGORGE_VIDRACC | 3 | 0.0% |
| HS_PURGE | 3 | 0.0% |
| NONMAN_VIDRACC | 3 | 0.0% |
| INTROU_PURGE | 3 | 0.0% |
| HS_VIDRACC | 3 | 0.0% |
| UFERMETURE_REGARD | 3 | 0.0% |
| REVETSOL_HYDRANT | 3 | 0.0% |
| PRBREGARD_VAIR500 | 2 | 0.0% |
| INTROU_MVENT | 2 | 0.0% |
| VETUSTE_HYDROSTAB | 2 | 0.0% |
| NONMAN_MVENT | 2 | 0.0% |
| PRBTAMPON_VIDRACC | 2 | 0.0% |
| INA_DETENDEUR | 2 | 0.0% |
| FUITE_VIDRACC | 2 | 0.0% |
| FUITE_HYDROSTAB | 1 | 0.0% |
| INA_PURGE | 1 | 0.0% |
| INTROU_SECMESDEB | 1 | 0.0% |
| ENGORGE_VAIR1000 | 1 | 0.0% |
| HS_SECMESDEB | 1 | 0.0% |
| NONMAN_VAIR500 | 1 | 0.0% |
| INA_MVENT | 1 | 0.0% |
| PRBTAMPON_VAIR1000 | 1 | 0.0% |
| REVETSOL_HYDROSTAB | 1 | 0.0% |
| VETUSTE_DETENDEUR | 1 | 0.0% |
| VETUSTE_VIDRACC | 1 | 0.0% |
| VETUSTE_VAIR500 | 1 | 0.0% |
| REVETSOL_VIDBORGNE | 1 | 0.0% |
| VETUSTE_VIDBORGNE | 1 | 0.0% |
| PRBREGARD_PURGE | 1 | 0.0% |
| REVETSOL_EQUIPPUB | 1 | 0.0% |
| UARB_MEN_ESPACE_VERT | 1 | 0.0% |
| ENTC_ESPACE_VERT | 1 | 0.0% |
| PRBREGARD_DETENDEUR | 1 | 0.0% |
| DESOR_GC_SOUTERRAIN | 1 | 0.0% |
| NONMAN_VAIR1000 | 1 | 0.0% |
| DESOR_GC_CIEL_OUVERT | 1 | 0.0% |
| DESOR_GC_REGARD | 1 | 0.0% |

### 3.4 Distribution des anomalies par tronçon
- Min: 1
- Max: 138
- Moyenne: 2.16
- Médiane: 1
- 75ème percentile: 2
- 90ème percentile: 4
- 99ème percentile: 17

## 4. Alignement des Tables

- GID dans patrimoine: 218,141
- GID dans anomalies: 14,276
- GID communs: 12,763
- GID uniquement dans patrimoine: 205,378
- GID uniquement dans anomalies (orphelins): 1,513
- % de tronçons avec anomalies: 5.9%
- % d'anomalies matchées: 89.4%

**Échantillon de GID orphelins (anomalies sans tronçon):**
`[np.int64(410140679), np.int64(410009621), np.int64(410009622), np.int64(410009627), np.int64(410009628), np.int64(500957221), np.int64(410034216), np.int64(410034219), np.int64(410034221), np.int64(410034222)]`

## 5. Décisions de Nettoyage

1. **Dates futures**: Seront exclues pour le calcul de FREEZE_DATE
2. **DDP > DHS**: Ces tronçons seront signalés et potentiellement exclus
3. **Anomalies orphelines**: Seront ignorées (pas de jointure possible)
4. **DHS manquant**: Interprété comme tronçon toujours en service (censuré)

## 6. Définition des Horizons

**Date maximale d'observation retenue**: 2024-11-07

**FREEZE_DATE par horizon:**
- Horizon 1 an(s): FREEZE_DATE = 2023-11-07
- Horizon 3 an(s): FREEZE_DATE = 2021-11-07
- Horizon 5 an(s): FREEZE_DATE = 2019-11-07