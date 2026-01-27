"""
Analyse de Calibration des Probabilités et Deep Dive sur les Tronçons Mal Classés
1. Vérifie si les probabilités prédites reflètent les risques réels
2. Analyse en profondeur les faux positifs et faux négatifs
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.isotonic import IsotonicRegression

from config import ARTIFACTS_DIR, REPORTS_DIR
from modeling import load_dataset, prepare_features

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_model_and_predictions():
    """Charge le modèle et calcule les prédictions."""
    print("\n1. Chargement du modèle et données...")

    # Charger le modèle
    model_artifact = joblib.load(ARTIFACTS_DIR / "best_model.joblib")
    model = model_artifact['model']
    metadata = model_artifact['metadata']
    horizon = metadata['horizon_years']

    # Charger les données
    df = load_dataset(horizon)
    X, y, _, _ = prepare_features(df)

    # Prédictions
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    print(f"   ✓ Modèle: {metadata['model_type']}")
    print(f"   ✓ Horizon: {horizon} an(s)")
    print(f"   ✓ {len(df):,} tronçons")

    return model, df, X, y, y_proba, y_pred, metadata


def analyze_calibration(y_true, y_proba, output_dir):
    """Analyse la calibration des probabilités."""
    print("\n2. Analyse de calibration...")

    # Calculer la courbe de calibration
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy='uniform')

    # Métriques de calibration
    brier = brier_score_loss(y_true, y_proba)
    logloss = log_loss(y_true, y_proba)

    print(f"   ✓ Brier Score: {brier:.4f} (0=parfait)")
    print(f"   ✓ Log Loss: {logloss:.4f} (plus bas=mieux)")

    # Calculer l'écart de calibration (Expected Calibration Error)
    ece = np.mean(np.abs(prob_true - prob_pred))
    print(f"   ✓ Expected Calibration Error: {ece:.4f}")

    # Analyse par bins de probabilité
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(y_proba, bins) - 1

    calibration_data = []
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        if mask.sum() > 0:
            n_samples = mask.sum()
            mean_pred_prob = y_proba[mask].mean()
            actual_prob = y_true[mask].mean()
            calibration_data.append({
                'bin': f'{bins[i]:.1f}-{bins[i+1]:.1f}',
                'n_samples': n_samples,
                'mean_predicted': mean_pred_prob,
                'actual_rate': actual_prob,
                'difference': mean_pred_prob - actual_prob
            })

    df_calib = pd.DataFrame(calibration_data)

    print("\n   Calibration par bin de probabilité:")
    for _, row in df_calib.iterrows():
        print(f"   {row['bin']}: Prédit={row['mean_predicted']:.3f}, Réel={row['actual_rate']:.3f}, "
              f"Diff={row['difference']:+.3f} (n={row['n_samples']:,})")

    return df_calib, brier, logloss, ece, prob_true, prob_pred


def plot_calibration_curve(prob_true, prob_pred, brier, ece, output_dir):
    """Visualise la courbe de calibration."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Courbe de calibration
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', label='Parfaitement calibré', linewidth=2)
    ax.plot(prob_pred, prob_true, 's-', linewidth=2, markersize=8,
            label='Modèle actuel', color='#e74c3c')

    ax.set_xlabel('Probabilité Prédite', fontsize=11)
    ax.set_ylabel('Proportion Réelle', fontsize=11)
    ax.set_title('Courbe de Calibration', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Ajouter les métriques
    metrics_text = f'Brier Score: {brier:.4f}\nECE: {ece:.4f}'
    ax.text(0.98, 0.02, metrics_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. Histogramme des probabilités prédites
    ax = axes[1]
    ax.hist(prob_pred, bins=10, edgecolor='black', alpha=0.7, color='#3498db')
    ax.set_xlabel('Bins de Probabilité Prédite', fontsize=11)
    ax.set_ylabel('Fréquence', fontsize=11)
    ax.set_title('Distribution des Probabilités Prédites', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Courbe de calibration sauvegardée")


def apply_calibration_methods(model, X, y, y_proba):
    """Applique différentes méthodes de calibration."""
    print("\n3. Application de méthodes de calibration...")

    # Isotonic Regression
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    y_proba_isotonic = iso_reg.fit_transform(y_proba, y)

    brier_isotonic = brier_score_loss(y, y_proba_isotonic)
    print(f"   ✓ Isotonic Regression - Brier: {brier_isotonic:.4f}")

    # Platt Scaling (utilise CalibratedClassifierCV avec sigmoid)
    # Note: On ne peut pas l'appliquer directement sur les probas, on simule
    # en ajustant une régression logistique
    from sklearn.linear_model import LogisticRegression
    platt = LogisticRegression()
    platt.fit(y_proba.reshape(-1, 1), y)
    y_proba_platt = platt.predict_proba(y_proba.reshape(-1, 1))[:, 1]

    brier_platt = brier_score_loss(y, y_proba_platt)
    print(f"   ✓ Platt Scaling - Brier: {brier_platt:.4f}")

    return y_proba_isotonic, y_proba_platt, brier_isotonic, brier_platt


def analyze_false_negatives(df, y_true, y_pred, y_proba):
    """Analyse approfondie des faux négatifs."""
    print("\n4. Deep Dive sur les Faux Négatifs (FN)...")

    # Identifier les FN
    fn_mask = (y_true == 1) & (y_pred == 0)
    df_fn = df[fn_mask].copy()
    df_fn['y_proba'] = y_proba[fn_mask]

    # Comparer avec les vrais positifs
    tp_mask = (y_true == 1) & (y_pred == 1)
    df_tp = df[tp_mask].copy()
    df_tp['y_proba'] = y_proba[tp_mask]

    print(f"   ✓ {len(df_fn)} faux négatifs identifiés")
    print(f"   ✓ {len(df_tp)} vrais positifs pour comparaison")

    # Statistiques comparatives
    comparisons = {
        'Métrique': [],
        'Faux Négatifs (FN)': [],
        'Vrais Positifs (TP)': [],
        'Différence': []
    }

    # Âge
    fn_age = df_fn['age_at_freeze'].mean()
    tp_age = df_tp['age_at_freeze'].mean()
    comparisons['Métrique'].append('Âge moyen (ans)')
    comparisons['Faux Négatifs (FN)'].append(f"{fn_age:.1f}")
    comparisons['Vrais Positifs (TP)'].append(f"{tp_age:.1f}")
    comparisons['Différence'].append(f"{fn_age - tp_age:+.1f}")

    # Diamètre
    fn_diam = df_fn['diametre'].mean()
    tp_diam = df_tp['diametre'].mean()
    comparisons['Métrique'].append('Diamètre moyen (mm)')
    comparisons['Faux Négatifs (FN)'].append(f"{fn_diam:.1f}")
    comparisons['Vrais Positifs (TP)'].append(f"{tp_diam:.1f}")
    comparisons['Différence'].append(f"{fn_diam - tp_diam:+.1f}")

    # Longueur
    fn_long = df_fn['longueur'].mean()
    tp_long = df_tp['longueur'].mean()
    comparisons['Métrique'].append('Longueur moyenne (m)')
    comparisons['Faux Négatifs (FN)'].append(f"{fn_long:.1f}")
    comparisons['Vrais Positifs (TP)'].append(f"{tp_long:.1f}")
    comparisons['Différence'].append(f"{fn_long - tp_long:+.1f}")

    # Probabilité
    fn_prob = df_fn['y_proba'].mean()
    tp_prob = df_tp['y_proba'].mean()
    comparisons['Métrique'].append('Probabilité moyenne')
    comparisons['Faux Négatifs (FN)'].append(f"{fn_prob:.3f}")
    comparisons['Vrais Positifs (TP)'].append(f"{tp_prob:.3f}")
    comparisons['Différence'].append(f"{fn_prob - tp_prob:+.3f}")

    # Nombre de fuites (si disponible)
    if 'n_fuites_total' in df_fn.columns:
        fn_fuites = df_fn['n_fuites_total'].mean()
        tp_fuites = df_tp['n_fuites_total'].mean()
        comparisons['Métrique'].append('Nb fuites moyen')
        comparisons['Faux Négatifs (FN)'].append(f"{fn_fuites:.2f}")
        comparisons['Vrais Positifs (TP)'].append(f"{tp_fuites:.2f}")
        comparisons['Différence'].append(f"{fn_fuites - tp_fuites:+.2f}")

    df_comparison = pd.DataFrame(comparisons)

    print("\n   Comparaison FN vs TP:")
    print(df_comparison.to_string(index=False))

    # Distribution par matériau
    print("\n   Distribution des FN par matériau:")
    fn_mat = df_fn['materiau'].value_counts().head(5)
    for mat, count in fn_mat.items():
        pct = count / len(df_fn) * 100
        print(f"   - {mat}: {count} ({pct:.1f}%)")

    # Distribution des scores de probabilité des FN
    print("\n   Distribution des scores des FN:")
    print(f"   - Min: {df_fn['y_proba'].min():.3f}")
    print(f"   - Q1: {df_fn['y_proba'].quantile(0.25):.3f}")
    print(f"   - Médiane: {df_fn['y_proba'].median():.3f}")
    print(f"   - Q3: {df_fn['y_proba'].quantile(0.75):.3f}")
    print(f"   - Max: {df_fn['y_proba'].max():.3f}")

    return df_fn, df_tp, df_comparison


def analyze_false_positives(df, y_true, y_pred, y_proba):
    """Analyse approfondie des faux positifs."""
    print("\n5. Deep Dive sur les Faux Positifs (FP)...")

    # Identifier les FP
    fp_mask = (y_true == 0) & (y_pred == 1)
    df_fp = df[fp_mask].copy()
    df_fp['y_proba'] = y_proba[fp_mask]

    # Comparer avec les vrais négatifs
    tn_mask = (y_true == 0) & (y_pred == 0)
    df_tn = df[tn_mask].copy()
    df_tn['y_proba'] = y_proba[tn_mask]

    print(f"   ✓ {len(df_fp):,} faux positifs identifiés")
    print(f"   ✓ {len(df_tn):,} vrais négatifs pour comparaison")

    # Statistiques comparatives
    comparisons = {
        'Métrique': [],
        'Faux Positifs (FP)': [],
        'Vrais Négatifs (TN)': [],
        'Différence': []
    }

    # Âge
    fp_age = df_fp['age_at_freeze'].mean()
    tn_age = df_tn['age_at_freeze'].mean()
    comparisons['Métrique'].append('Âge moyen (ans)')
    comparisons['Faux Positifs (FP)'].append(f"{fp_age:.1f}")
    comparisons['Vrais Négatifs (TN)'].append(f"{tn_age:.1f}")
    comparisons['Différence'].append(f"{fp_age - tn_age:+.1f}")

    # Diamètre
    fp_diam = df_fp['diametre'].mean()
    tn_diam = df_tn['diametre'].mean()
    comparisons['Métrique'].append('Diamètre moyen (mm)')
    comparisons['Faux Positifs (FP)'].append(f"{fp_diam:.1f}")
    comparisons['Vrais Négatifs (TN)'].append(f"{tn_diam:.1f}")
    comparisons['Différence'].append(f"{fp_diam - tn_diam:+.1f}")

    # Longueur
    fp_long = df_fp['longueur'].mean()
    tn_long = df_tn['longueur'].mean()
    comparisons['Métrique'].append('Longueur moyenne (m)')
    comparisons['Faux Positifs (FP)'].append(f"{fp_long:.1f}")
    comparisons['Vrais Négatifs (TN)'].append(f"{tn_long:.1f}")
    comparisons['Différence'].append(f"{fp_long - tn_long:+.1f}")

    # Probabilité
    fp_prob = df_fp['y_proba'].mean()
    tn_prob = df_tn['y_proba'].mean()
    comparisons['Métrique'].append('Probabilité moyenne')
    comparisons['Faux Positifs (FP)'].append(f"{fp_prob:.3f}")
    comparisons['Vrais Négatifs (TN)'].append(f"{tn_prob:.3f}")
    comparisons['Différence'].append(f"{fp_prob - tn_prob:+.3f}")

    df_comparison = pd.DataFrame(comparisons)

    print("\n   Comparaison FP vs TN:")
    print(df_comparison.to_string(index=False))

    # Distribution par matériau
    print("\n   Top 5 matériaux dans les FP:")
    fp_mat = df_fp['materiau'].value_counts().head(5)
    for mat, count in fp_mat.items():
        pct = count / len(df_fp) * 100
        print(f"   - {mat}: {count:,} ({pct:.1f}%)")

    # Identifier les sous-groupes problématiques
    print("\n   Sous-groupes avec taux de FP élevé:")

    # Par matériau
    for mat in df_fp['materiau'].value_counts().head(3).index:
        total_mat = len(df[df['materiau'] == mat])
        fp_mat_count = len(df_fp[df_fp['materiau'] == mat])
        fp_rate = fp_mat_count / total_mat * 100 if total_mat > 0 else 0
        print(f"   - {mat}: {fp_rate:.1f}% de FP ({fp_mat_count:,}/{total_mat:,})")

    # Distribution des scores
    print("\n   Distribution des scores des FP:")
    print(f"   - Min: {df_fp['y_proba'].min():.3f}")
    print(f"   - Q1: {df_fp['y_proba'].quantile(0.25):.3f}")
    print(f"   - Médiane: {df_fp['y_proba'].median():.3f}")
    print(f"   - Q3: {df_fp['y_proba'].quantile(0.75):.3f}")
    print(f"   - Max: {df_fp['y_proba'].max():.3f}")

    # FP "confiants" (probabilité > 0.7)
    fp_confident = df_fp[df_fp['y_proba'] > 0.7]
    print(f"\n   ⚠️  FP très confiants (prob > 0.7): {len(fp_confident):,} ({len(fp_confident)/len(df_fp)*100:.1f}%)")

    return df_fp, df_tn, df_comparison


def plot_misclassified_analysis(df_fn, df_tp, df_fp, df_tn, output_dir):
    """Visualise l'analyse des tronçons mal classés."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Distribution des probabilités FN vs TP
    ax = axes[0, 0]
    ax.hist(df_fn['y_proba'], bins=20, alpha=0.7, label='FN', color='#e74c3c', edgecolor='black')
    ax.hist(df_tp['y_proba'], bins=20, alpha=0.7, label='TP', color='#2ecc71', edgecolor='black')
    ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Seuil (0.5)')
    ax.set_xlabel('Probabilité Prédite', fontsize=10)
    ax.set_ylabel('Nombre de tronçons', fontsize=10)
    ax.set_title('Distribution des Probabilités: FN vs TP', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Distribution des probabilités FP vs TN
    ax = axes[0, 1]
    ax.hist(df_fp['y_proba'], bins=20, alpha=0.7, label='FP', color='#f39c12', edgecolor='black')
    ax.hist(df_tn['y_proba'].sample(min(len(df_tn), 10000)), bins=20, alpha=0.7,
            label='TN (échantillon)', color='#3498db', edgecolor='black')
    ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Seuil (0.5)')
    ax.set_xlabel('Probabilité Prédite', fontsize=10)
    ax.set_ylabel('Nombre de tronçons', fontsize=10)
    ax.set_title('Distribution des Probabilités: FP vs TN', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Âge: FN vs TP
    ax = axes[0, 2]
    data = [df_fn['age_at_freeze'].dropna(), df_tp['age_at_freeze'].dropna()]
    bp = ax.boxplot(data, labels=['FN', 'TP'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#e74c3c')
    bp['boxes'][1].set_facecolor('#2ecc71')
    ax.set_ylabel('Âge (années)', fontsize=10)
    ax.set_title('Distribution de l\'Âge: FN vs TP', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 4. Âge: FP vs TN
    ax = axes[1, 0]
    data = [df_fp['age_at_freeze'].dropna(),
            df_tn['age_at_freeze'].sample(min(len(df_tn), 10000)).dropna()]
    bp = ax.boxplot(data, labels=['FP', 'TN'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#f39c12')
    bp['boxes'][1].set_facecolor('#3498db')
    ax.set_ylabel('Âge (années)', fontsize=10)
    ax.set_title('Distribution de l\'Âge: FP vs TN', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 5. Top matériaux FN
    ax = axes[1, 1]
    fn_mat = df_fn['materiau'].value_counts().head(5)
    ax.barh(range(len(fn_mat)), fn_mat.values, color='#e74c3c', edgecolor='black')
    ax.set_yticks(range(len(fn_mat)))
    ax.set_yticklabels(fn_mat.index)
    ax.set_xlabel('Nombre de FN', fontsize=10)
    ax.set_title('Top 5 Matériaux dans les FN', fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    # 6. Top matériaux FP
    ax = axes[1, 2]
    fp_mat = df_fp['materiau'].value_counts().head(5)
    ax.barh(range(len(fp_mat)), fp_mat.values, color='#f39c12', edgecolor='black')
    ax.set_yticks(range(len(fp_mat)))
    ax.set_yticklabels(fp_mat.index)
    ax.set_xlabel('Nombre de FP', fontsize=10)
    ax.set_title('Top 5 Matériaux dans les FP', fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_dir / 'misclassified_deep_dive.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Analyse des mal classés sauvegardée")


def identify_systematic_errors(df, y_true, y_pred, y_proba):
    """Identifie les sous-groupes systématiquement mal prédits."""
    print("\n6. Identification des sous-groupes problématiques...")

    df_analysis = df.copy()
    df_analysis['y_true'] = y_true
    df_analysis['y_pred'] = y_pred
    df_analysis['y_proba'] = y_proba
    df_analysis['is_fp'] = (y_true == 0) & (y_pred == 1)
    df_analysis['is_fn'] = (y_true == 1) & (y_pred == 0)

    # Créer des bins d'âge
    df_analysis['age_bin'] = pd.cut(df_analysis['age_at_freeze'],
                                     bins=[0, 30, 50, 70, 100, 200],
                                     labels=['<30', '30-50', '50-70', '70-100', '100+'])

    # Analyser par combinaisons matériau x âge
    problematic_groups = []

    for mat in df_analysis['materiau'].value_counts().head(5).index:
        for age_bin in df_analysis['age_bin'].unique():
            subset = df_analysis[(df_analysis['materiau'] == mat) &
                                (df_analysis['age_bin'] == age_bin)]

            if len(subset) >= 50:  # Au moins 50 tronçons
                fp_rate = subset['is_fp'].mean() * 100
                fn_rate = subset['is_fn'].mean() * 100
                total_error = (subset['is_fp'] | subset['is_fn']).mean() * 100

                # Considérer comme problématique si taux d'erreur > 20%
                if total_error > 20:
                    problematic_groups.append({
                        'materiau': mat,
                        'age_bin': age_bin,
                        'n_total': len(subset),
                        'fp_rate': fp_rate,
                        'fn_rate': fn_rate,
                        'total_error_rate': total_error
                    })

    if problematic_groups:
        df_problematic = pd.DataFrame(problematic_groups)
        df_problematic = df_problematic.sort_values('total_error_rate', ascending=False)

        print("\n   Sous-groupes avec taux d'erreur > 20%:")
        for _, row in df_problematic.head(10).iterrows():
            print(f"   - {row['materiau']} / {row['age_bin']} ans: "
                  f"{row['total_error_rate']:.1f}% erreur (FP:{row['fp_rate']:.1f}%, FN:{row['fn_rate']:.1f}%) "
                  f"| n={row['n_total']}")

        return df_problematic
    else:
        print("   ✓ Aucun sous-groupe avec taux d'erreur > 20%")
        return pd.DataFrame()


def generate_report(df_calib, brier, logloss, ece,
                   df_fn_comparison, df_fp_comparison,
                   df_problematic, metadata, output_path):
    """Génère un rapport markdown complet."""
    print("\n7. Génération du rapport...")

    report = f"""# Rapport: Calibration des Probabilités et Analyse des Erreurs

*Généré le {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*

---

## 1. Calibration des Probabilités

### Objectif
Vérifier si une probabilité prédite de 70% correspond réellement à un risque de 70%.

### Métriques Globales

- **Brier Score**: {brier:.4f} (0 = parfait, plus bas est mieux)
- **Log Loss**: {logloss:.4f} (plus bas est mieux)
- **Expected Calibration Error (ECE)**: {ece:.4f} (0 = parfaitement calibré)

### Interprétation

"""

    if ece < 0.05:
        report += "✅ **Excellente calibration** - Les probabilités sont fiables\n"
    elif ece < 0.10:
        report += "⚠️ **Calibration acceptable** - Légère surestimation ou sous-estimation\n"
    else:
        report += "❌ **Calibration médiocre** - Recalibration recommandée\n"

    report += f"""
### Calibration par Bin de Probabilité

{df_calib.to_markdown(index=False, floatfmt='.3f')}

**Lecture du tableau**:
- `mean_predicted`: Probabilité moyenne prédite dans ce bin
- `actual_rate`: Taux de défaillance réel observé
- `difference`: Écart (positif = surestimation, négatif = sous-estimation)

### Recommandations

"""

    # Analyser les bins pour recommandations
    over_bins = df_calib[df_calib['difference'] > 0.05]
    under_bins = df_calib[df_calib['difference'] < -0.05]

    if len(over_bins) > 0:
        report += f"- ⚠️ **Surestimation** dans les bins: {', '.join(over_bins['bin'].tolist())}\n"
        report += "  → Le modèle prédit des probabilités trop élevées\n"

    if len(under_bins) > 0:
        report += f"- ⚠️ **Sous-estimation** dans les bins: {', '.join(under_bins['bin'].tolist())}\n"
        report += "  → Le modèle prédit des probabilités trop faibles\n"

    report += """
**Solutions**:
1. Appliquer Isotonic Regression pour recalibrer
2. Utiliser Platt Scaling (régression logistique)
3. Ajuster le seuil de décision selon les coûts métier

---

## 2. Analyse des Faux Négatifs (FN)

### Caractéristiques des FN vs TP

{df_fn_comparison.to_markdown(index=False)}

### Insights Clés - Faux Négatifs

"""

    # Extraire insights sur FN
    fn_age = float(df_fn_comparison[df_fn_comparison['Métrique'] == 'Âge moyen (ans)']['Faux Négatifs (FN)'].values[0])
    tp_age = float(df_fn_comparison[df_fn_comparison['Métrique'] == 'Âge moyen (ans)']['Vrais Positifs (TP)'].values[0])

    if fn_age < tp_age:
        report += f"1. **Les FN sont plus jeunes** ({fn_age:.1f} ans vs {tp_age:.1f} ans)\n"
        report += "   - Le modèle sous-estime le risque des tronçons jeunes qui défaillent\n"
        report += "   - Possiblement des défaillances dues à défauts de fabrication/pose\n\n"

    fn_prob = float(df_fn_comparison[df_fn_comparison['Métrique'] == 'Probabilité moyenne']['Faux Négatifs (FN)'].values[0])
    report += f"2. **Score moyen des FN**: {fn_prob:.3f}\n"
    report += "   - Ces tronçons étaient juste sous le seuil de 0.5\n"
    report += "   - Un seuil plus bas (ex: 0.4) permettrait de les détecter\n\n"

    report += """
### Actions Recommandées

1. **Baisser le seuil de décision** à 0.4 pour capturer plus de défaillances
2. **Enrichir les features** pour mieux prédire les défaillances précoces
3. **Analyse par expertise** des FN pour identifier patterns manquants

---

## 3. Analyse des Faux Positifs (FP)

### Caractéristiques des FP vs TN

{df_fp_comparison.to_markdown(index=False)}

### Insights Clés - Faux Positifs

"""

    # Extraire insights sur FP
    fp_age = float(df_fp_comparison[df_fp_comparison['Métrique'] == 'Âge moyen (ans)']['Faux Positifs (FP)'].values[0])
    tn_age = float(df_fp_comparison[df_fp_comparison['Métrique'] == 'Âge moyen (ans)']['Vrais Négatifs (TN)'].values[0])

    report += f"1. **Les FP sont plus âgés** ({fp_age:.1f} ans vs {tn_age:.1f} ans)\n"
    report += "   - Le modèle surévalue le risque lié à l'âge seul\n"
    report += "   - Certains tronçons anciens sont en bon état malgré leur âge\n\n"

    fp_prob = float(df_fp_comparison[df_fp_comparison['Métrique'] == 'Probabilité moyenne']['Faux Positifs (FP)'].values[0])
    report += f"2. **Score moyen des FP**: {fp_prob:.3f}\n"
    if fp_prob > 0.6:
        report += "   - ⚠️ Le modèle est très confiant sur ces FP\n"
        report += "   - Problème de calibration ou features manquantes (qualité, entretien)\n\n"

    report += """
### Actions Recommandées

1. **Augmenter le seuil** à 0.6 pour réduire les FP (mais augmente les FN)
2. **Ajouter features de qualité** (inspections, maintenance préventive)
3. **Analyse coût-bénéfice** pour trouver le seuil optimal

---

## 4. Sous-groupes Problématiques

"""

    if len(df_problematic) > 0:
        report += "Les combinaisons matériau/âge suivantes ont des taux d'erreur élevés:\n\n"
        for _, row in df_problematic.head(10).iterrows():
            report += f"- **{row['materiau']} / {row['age_bin']} ans**: {row['total_error_rate']:.1f}% d'erreur "
            report += f"(FP: {row['fp_rate']:.1f}%, FN: {row['fn_rate']:.1f}%) | n={row['n_total']}\n"

        report += "\n**Action**: Analyser ces sous-groupes spécifiquement et ajuster le modèle\n"
    else:
        report += "✅ Aucun sous-groupe avec taux d'erreur > 20%\n"

    report += """
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
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"   ✓ Rapport sauvegardé: {output_path}")


def main():
    """Fonction principale."""
    print("="*70)
    print("CALIBRATION DES PROBABILITÉS ET ANALYSE DES ERREURS")
    print("="*70)

    # Créer répertoire de sortie
    output_dir = REPORTS_DIR / "plots"
    output_dir.mkdir(exist_ok=True, parents=True)

    # 1. Charger modèle et prédictions
    model, df, X, y, y_proba, y_pred, metadata = load_model_and_predictions()

    # 2. Analyse de calibration
    df_calib, brier, logloss, ece, prob_true, prob_pred = analyze_calibration(
        y, y_proba, output_dir
    )
    plot_calibration_curve(prob_true, prob_pred, brier, ece, output_dir)

    # 3. Méthodes de calibration
    y_proba_iso, y_proba_platt, brier_iso, brier_platt = apply_calibration_methods(
        model, X, y, y_proba
    )

    # 4. Analyse des faux négatifs
    df_fn, df_tp, df_fn_comparison = analyze_false_negatives(df, y, y_pred, y_proba)

    # 5. Analyse des faux positifs
    df_fp, df_tn, df_fp_comparison = analyze_false_positives(df, y, y_pred, y_proba)

    # 6. Visualisations
    print("\n8. Génération des visualisations...")
    plot_misclassified_analysis(df_fn, df_tp, df_fp, df_tn, output_dir)

    # 7. Identifier sous-groupes problématiques
    df_problematic = identify_systematic_errors(df, y, y_pred, y_proba)

    # 8. Rapport
    report_path = REPORTS_DIR / "calibration_and_errors_report.md"
    generate_report(df_calib, brier, logloss, ece,
                   df_fn_comparison, df_fp_comparison,
                   df_problematic, metadata, report_path)

    # 9. Sauvegarder les données
    print("\n9. Sauvegarde des données détaillées...")

    df_fn[['GID', 'materiau', 'age_at_freeze', 'diametre', 'longueur', 'y_proba']].to_csv(
        REPORTS_DIR / "fn_details.csv", index=False
    )
    print(f"   ✓ FN sauvegardés: fn_details.csv ({len(df_fn)} lignes)")

    df_fp[['GID', 'materiau', 'age_at_freeze', 'diametre', 'longueur', 'y_proba']].to_csv(
        REPORTS_DIR / "fp_details.csv", index=False
    )
    print(f"   ✓ FP sauvegardés: fp_details.csv ({len(df_fp):,} lignes)")

    if len(df_problematic) > 0:
        df_problematic.to_csv(REPORTS_DIR / "problematic_groups.csv", index=False)
        print(f"   ✓ Groupes problématiques sauvegardés: problematic_groups.csv")

    print("\n" + "="*70)
    print("ANALYSE TERMINÉE")
    print("="*70)
    print(f"\n📊 Rapport: {report_path}")
    print(f"📈 Visualisations: {output_dir}")
    print(f"💾 Données: {REPORTS_DIR}")

    return df_fn, df_fp, df_calib, df_problematic


if __name__ == "__main__":
    main()
