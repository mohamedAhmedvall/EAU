# Rapport d'Explicabilite

*Genere le: 2026-01-19 19:44:25*


## Horizon 1 an(s)

**Modele**: LightGBM

### Top 15 Features (par importance)
| Rang | Feature | Importance |
|------|---------|------------|
| 3 | longueur | 1851.0000 |
| 1 | age_at_freeze | 1301.0000 |
| 19 | surface_approx | 1220.0000 |
| 2 | diametre | 1209.0000 |
| 14 | ratio_age_median | 1025.0000 |
| 20 | age_x_ratio | 655.0000 |
| 15 | overdue_years | 387.0000 |
| 13 | leak_rate_per_km | 271.0000 |
| 21 | materiau_encoded | 265.0000 |
| 10 | days_since_last_fuite | 263.0000 |
| 12 | leak_rate_per_year | 185.0000 |
| 18 | age_x_nfuites | 120.0000 |
| 17 | over_p90_life | 26.0000 |
| 22 | decade_install_encoded | 23.0000 |
| 8 | n_fuites_3y | 14.0000 |

### Verification metier

**Features attendues a impact positif (plus de risque):**
- age_at_freeze: importance = 1301.0000
- n_fuites_total: importance = 11.0000
- n_fuites_1y: importance = 2.0000
- n_fuites_3y: importance = 14.0000
- n_fuites_5y: importance = 7.0000
- ratio_age_median: importance = 1025.0000
- overdue_years: importance = 387.0000
- over_p75_life: importance = 8.0000
- over_p90_life: importance = 26.0000
- leak_rate_per_year: importance = 185.0000
- leak_rate_per_km: importance = 271.0000
- has_recent_fuite: importance = 0.0000

**Features attendues a impact negatif (moins de risque):**
- days_since_last_fuite: importance = 263.0000

[OK] Aucune anomalie de signe detectee.

## Horizon 3 an(s)

**Modele**: LightGBM

### Top 15 Features (par importance)
| Rang | Feature | Importance |
|------|---------|------------|
| 1 | age_at_freeze | 1714.0000 |
| 3 | longueur | 1417.0000 |
| 2 | diametre | 1273.0000 |
| 14 | ratio_age_median | 1084.0000 |
| 19 | surface_approx | 957.0000 |
| 20 | age_x_ratio | 624.0000 |
| 15 | overdue_years | 378.0000 |
| 21 | materiau_encoded | 319.0000 |
| 10 | days_since_last_fuite | 237.0000 |
| 12 | leak_rate_per_year | 199.0000 |
| 13 | leak_rate_per_km | 187.0000 |
| 18 | age_x_nfuites | 129.0000 |
| 9 | n_fuites_5y | 27.0000 |
| 6 | n_fuites_total | 26.0000 |
| 22 | decade_install_encoded | 24.0000 |

### Verification metier

**Features attendues a impact positif (plus de risque):**
- age_at_freeze: importance = 1714.0000
- n_fuites_total: importance = 26.0000
- n_fuites_1y: importance = 17.0000
- n_fuites_3y: importance = 16.0000
- n_fuites_5y: importance = 27.0000
- ratio_age_median: importance = 1084.0000
- overdue_years: importance = 378.0000
- over_p75_life: importance = 2.0000
- over_p90_life: importance = 23.0000
- leak_rate_per_year: importance = 199.0000
- leak_rate_per_km: importance = 187.0000
- has_recent_fuite: importance = 0.0000

**Features attendues a impact negatif (moins de risque):**
- days_since_last_fuite: importance = 237.0000

[OK] Aucune anomalie de signe detectee.

## Horizon 5 an(s)

**Modele**: LightGBM

### Top 15 Features (par importance)
| Rang | Feature | Importance |
|------|---------|------------|
| 1 | age_at_freeze | 1620.0000 |
| 3 | longueur | 1391.0000 |
| 2 | diametre | 1270.0000 |
| 14 | ratio_age_median | 1209.0000 |
| 19 | surface_approx | 1047.0000 |
| 20 | age_x_ratio | 786.0000 |
| 21 | materiau_encoded | 356.0000 |
| 15 | overdue_years | 292.0000 |
| 10 | days_since_last_fuite | 238.0000 |
| 13 | leak_rate_per_km | 192.0000 |
| 12 | leak_rate_per_year | 130.0000 |
| 18 | age_x_nfuites | 128.0000 |
| 8 | n_fuites_3y | 37.0000 |
| 9 | n_fuites_5y | 30.0000 |
| 17 | over_p90_life | 27.0000 |

### Verification metier

**Features attendues a impact positif (plus de risque):**
- age_at_freeze: importance = 1620.0000
- n_fuites_total: importance = 16.0000
- n_fuites_1y: importance = 6.0000
- n_fuites_3y: importance = 37.0000
- n_fuites_5y: importance = 30.0000
- ratio_age_median: importance = 1209.0000
- overdue_years: importance = 292.0000
- over_p75_life: importance = 2.0000
- over_p90_life: importance = 27.0000
- leak_rate_per_year: importance = 130.0000
- leak_rate_per_km: importance = 192.0000
- has_recent_fuite: importance = 0.0000

**Features attendues a impact negatif (moins de risque):**
- days_since_last_fuite: importance = 238.0000

[OK] Aucune anomalie de signe detectee.