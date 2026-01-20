# Analyse Descriptive de Survie (Kaplan-Meier)

*Généré le: 2026-01-19 19:34:11*


## 1. Statistiques Globales

- **Nombre total de tronçons analysés**: 211,806
- **Nombre d'événements (DHS observé)**: 32,269
- **Taux d'événement global**: 15.2%
- **Durée de vie médiane (Kaplan-Meier)**: inf ans
- **Durée d'observation moyenne**: 43.7 ans

### Interprétation
La durée de vie médiane représente l'âge auquel 50% des tronçons sont défaillants/abandonnés.
Un taux d'événement de 84.8% signifie que la majorité des tronçons sont encore en service (censurés à droite).

## 2. Survie par Matériau

![Courbes KM par matériau](km_by_mat.png)

| Matériau | N total | N événements | Taux événement | Durée médiane (ans) |
|----------|---------|--------------|----------------|---------------------|
| FTVI | 12,307 | 184 | 1.5% | 53.3 |
| A.C | 407 | 87 | 21.4% | 114.9 |
| BTM | 2,390 | 323 | 13.5% | 117.3 |
| ACIE | 1,752 | 339 | 19.3% | inf |
| AUTRE | 5,681 | 41 | 0.7% | inf |
| FT | 109,448 | 12,089 | 11.0% | inf |
| FTG | 57,942 | 16,033 | 27.7% | inf |
| PEHD | 8,604 | 686 | 8.0% | inf |
| POLY | 7,611 | 1,742 | 22.9% | inf |
| PVC | 4,198 | 555 | 13.2% | inf |

### Observations par matériau
- **Matériau le plus fragile**: FTVI (médiane: 53.3 ans)
- **Matériau le plus durable**: PVC (médiane: inf ans)

## 3. Survie par Décennie d'Installation

![Courbes KM par décennie](km_by_decade_install_str.png)

| Décennie | N total | N événements | Taux événement | Durée médiane (ans) |
|----------|---------|--------------|----------------|---------------------|
| 1900s | 17,099 | 997 | 5.8% | inf |
| 1940s | 17,043 | 7,945 | 46.6% | inf |
| 1950s | 10,161 | 3,172 | 31.2% | inf |
| 1960s | 28,125 | 6,859 | 24.4% | inf |
| 1970s | 29,195 | 4,359 | 14.9% | inf |
| 1980s | 21,744 | 1,959 | 9.0% | inf |
| 1990s | 13,171 | 1,571 | 11.9% | inf |
| 2000s | 22,670 | 3,707 | 16.4% | inf |
| 2010s | 30,131 | 1,340 | 4.4% | inf |
| 2020s | 21,564 | 203 | 0.9% | inf |

### Observations temporelles
- Les tronçons anciens ont un biais de survivant: ceux qui sont encore en service ont "survécu" plus longtemps.
- Les tronçons récents sont censurés (pas assez de recul pour observer leur défaillance).

## 4. Survie par Historique d'Anomalies

![Courbes KM par groupe d'anomalies](km_by_anomaly_group.png)

| Groupe anomalies | N total | N événements | Taux événement | Durée médiane (ans) |
|------------------|---------|--------------|----------------|---------------------|
| 0 | 199,499 | 28,678 | 14.4% | inf |
| 1 | 8,010 | 2,068 | 25.8% | 107.6 |
| 2-3 | 2,967 | 1,009 | 34.0% | 75.1 |
| 4+ | 1,330 | 514 | 38.6% | 71.1 |

### Observations sur les anomalies
- Les tronçons avec plus d'anomalies historiques ont généralement une durée de vie plus courte.
- Cela confirme l'utilité de l'historique des fuites comme prédicteur de risque.

## 5. Survie par Diamètre

![Courbes KM par diamètre](km_by_diametre_bin.png)

| Diamètre | N total | N événements | Taux événement | Durée médiane (ans) |
|----------|---------|--------------|----------------|---------------------|
| 100-150 | 48,089 | 6,953 | 14.5% | inf |
| 150-200 | 20,130 | 2,921 | 14.5% | inf |
| 200-300 | 16,000 | 3,105 | 19.4% | 85.2 |
| 300-1000 | 8,940 | 924 | 10.3% | 108.8 |
| 80-100 | 72,614 | 10,481 | 14.4% | inf |
| >1000 | 317 | 26 | 8.2% | inf |
| ≤80 | 41,824 | 7,799 | 18.6% | inf |

## 6. Considérations Méthodologiques

### Biais potentiels identifiés
1. **Biais de survivant**: Les tronçons très anciens encore en service sont par définition des "survivants"
2. **Censure à droite importante**: 84.8% des tronçons n'ont pas encore d'événement observé
3. **Troncature à gauche**: Les tronçons posés avant le début de l'observation des anomalies peuvent avoir un historique incomplet

### Implications pour la modélisation
- Utiliser une approche freeze+horizon pour éviter la fuite de données temporelles
- Les features basées sur l'historique des anomalies doivent être calculées strictement avant FREEZE_DATE
- Les durées de vie médianes par matériau seront utilisées comme features de référence