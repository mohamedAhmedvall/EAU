"""
Analyse de la Distribution des Erreurs du Modèle de Prédiction
Analyse détaillée des faux positifs, faux négatifs, et distribution des erreurs
par différentes dimensions (matériau, âge, diamètre, etc.)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from config import ARTIFACTS_DIR, REPORTS_DIR
from modeling import load_dataset, prepare_features

# Configuration des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_best_model_and_data():
    """Charge le meilleur modèle et les données de test."""
    # Charger le modèle
    model_artifact = joblib.load(ARTIFACTS_DIR / "best_model.joblib")
    model = model_artifact['model']
    encoders = model_artifact['encoders']
    metadata = model_artifact['metadata']

    # Charger les données pour l'horizon du meilleur modèle
    horizon = metadata['horizon_years']
    df = load_dataset(horizon)

    print(f"Modèle chargé: {metadata['model_type']}")
    print(f"Horizon: {horizon} an(s)")
    print(f"Dataset: {len(df):,} tronçons")

    return model, encoders, metadata, df, horizon


def compute_predictions_and_errors(model, df, threshold=0.5):
    """Calcule les prédictions et identifie les erreurs."""
    # Préparer les features
    X, y, _, _ = prepare_features(df)

    # Prédictions
    if hasattr(model, 'predict_proba'):
        y_scores = model.predict_proba(X)[:, 1]
    else:
        y_scores = model.predict(X)

    y_pred = (y_scores >= threshold).astype(int)

    # Classification des prédictions
    df_result = df.copy()
    df_result['y_true'] = y
    df_result['y_pred'] = y_pred
    df_result['y_score'] = y_scores

    # Types d'erreurs
    df_result['prediction_type'] = 'TN'  # True Negative (par défaut)
    df_result.loc[(y == 1) & (y_pred == 1), 'prediction_type'] = 'TP'  # True Positive
    df_result.loc[(y == 0) & (y_pred == 1), 'prediction_type'] = 'FP'  # False Positive
    df_result.loc[(y == 1) & (y_pred == 0), 'prediction_type'] = 'FN'  # False Negative

    return df_result, y_scores


def compute_confusion_matrix_stats(df_result):
    """Calcule les statistiques de la matrice de confusion."""
    TP = len(df_result[df_result['prediction_type'] == 'TP'])
    TN = len(df_result[df_result['prediction_type'] == 'TN'])
    FP = len(df_result[df_result['prediction_type'] == 'FP'])
    FN = len(df_result[df_result['prediction_type'] == 'FN'])

    total = TP + TN + FP + FN

    # Métriques
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0

    stats = {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'total': total,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'fpr': fpr,  # False Positive Rate
        'fnr': fnr   # False Negative Rate
    }

    return stats


def analyze_errors_by_dimension(df_result, dimension_col):
    """Analyse la distribution des erreurs par une dimension donnée."""
    results = []

    for dim_value in df_result[dimension_col].unique():
        if pd.isna(dim_value):
            continue

        df_dim = df_result[df_result[dimension_col] == dim_value]

        if len(df_dim) < 10:  # Ignorer les groupes trop petits
            continue

        TP = len(df_dim[df_dim['prediction_type'] == 'TP'])
        TN = len(df_dim[df_dim['prediction_type'] == 'TN'])
        FP = len(df_dim[df_dim['prediction_type'] == 'FP'])
        FN = len(df_dim[df_dim['prediction_type'] == 'FN'])

        total = len(df_dim)
        n_events = df_dim['y_true'].sum()
        event_rate = n_events / total if total > 0 else 0

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

        results.append({
            dimension_col: dim_value,
            'n_total': total,
            'n_events': n_events,
            'event_rate': event_rate,
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN,
            'precision': precision,
            'recall': recall,
            'fpr': fpr
        })

    df_analysis = pd.DataFrame(results)
    df_analysis = df_analysis.sort_values('n_total', ascending=False)

    return df_analysis


def plot_confusion_matrix(stats, output_dir):
    """Visualise la matrice de confusion."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Matrice
    conf_matrix = np.array([
        [stats['TN'], stats['FP']],
        [stats['FN'], stats['TP']]
    ])

    # Heatmap
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Prédit Négatif', 'Prédit Positif'],
                yticklabels=['Réel Négatif', 'Réel Positif'],
                ax=ax, cbar_kws={'label': 'Nombre de tronçons'})

    ax.set_title('Matrice de Confusion', fontsize=14, fontweight='bold')
    ax.set_ylabel('Valeur Réelle', fontsize=12)
    ax.set_xlabel('Valeur Prédite', fontsize=12)

    # Ajouter les métriques
    metrics_text = f"""
    Accuracy: {stats['accuracy']:.2%}
    Precision: {stats['precision']:.2%}
    Recall: {stats['recall']:.2%}
    F1-Score: {stats['f1_score']:.3f}
    """
    plt.figtext(0.15, 0.02, metrics_text, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Matrice de confusion sauvegardée")


def plot_error_distribution(df_result, output_dir):
    """Visualise la distribution des types d'erreurs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Distribution absolue
    counts = df_result['prediction_type'].value_counts()
    colors = {'TP': '#2ecc71', 'TN': '#3498db', 'FP': '#e74c3c', 'FN': '#f39c12'}
    color_list = [colors[x] for x in counts.index]

    axes[0].bar(counts.index, counts.values, color=color_list, edgecolor='black', linewidth=1.5)
    axes[0].set_title('Distribution des Prédictions', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Nombre de tronçons', fontsize=10)
    axes[0].set_xlabel('Type de Prédiction', fontsize=10)

    # Ajouter les valeurs sur les barres
    for i, (idx, val) in enumerate(counts.items()):
        axes[0].text(i, val, f'{val:,}\n({val/len(df_result)*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')

    # 2. Distribution relative (%)
    pcts = (counts / len(df_result) * 100).sort_index()
    axes[1].pie(pcts.values, labels=pcts.index, autopct='%1.1f%%',
                colors=[colors[x] for x in pcts.index],
                startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    axes[1].set_title('Répartition en %', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Distribution des erreurs sauvegardée")


def plot_errors_by_material(df_analysis_mat, output_dir):
    """Visualise les erreurs par matériau."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Trier par nombre total
    df_plot = df_analysis_mat.sort_values('n_total', ascending=False).head(10)

    # 1. Nombre d'erreurs par type
    ax = axes[0, 0]
    x = np.arange(len(df_plot))
    width = 0.2

    ax.bar(x - width*1.5, df_plot['TP'], width, label='TP (Vrai Positif)', color='#2ecc71')
    ax.bar(x - width*0.5, df_plot['FP'], width, label='FP (Faux Positif)', color='#e74c3c')
    ax.bar(x + width*0.5, df_plot['FN'], width, label='FN (Faux Négatif)', color='#f39c12')
    ax.bar(x + width*1.5, df_plot['TN'], width, label='TN (Vrai Négatif)', color='#3498db', alpha=0.5)

    ax.set_xlabel('Matériau', fontsize=10)
    ax.set_ylabel('Nombre de tronçons', fontsize=10)
    ax.set_title('Distribution des Erreurs par Matériau', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot['materiau'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Taux d'erreur (FPR)
    ax = axes[0, 1]
    ax.barh(df_plot['materiau'], df_plot['fpr'] * 100, color='#e74c3c', edgecolor='black')
    ax.set_xlabel('Taux de Faux Positifs (%)', fontsize=10)
    ax.set_title('Taux de Faux Positifs par Matériau', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # 3. Precision
    ax = axes[1, 0]
    ax.barh(df_plot['materiau'], df_plot['precision'] * 100, color='#2ecc71', edgecolor='black')
    ax.set_xlabel('Précision (%)', fontsize=10)
    ax.set_title('Précision par Matériau', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # 4. Recall
    ax = axes[1, 1]
    ax.barh(df_plot['materiau'], df_plot['recall'] * 100, color='#3498db', edgecolor='black')
    ax.set_xlabel('Rappel (%)', fontsize=10)
    ax.set_title('Rappel par Matériau', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'errors_by_material.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Analyse par matériau sauvegardée")


def plot_errors_by_age(df_result, output_dir):
    """Visualise les erreurs par tranche d'âge."""
    # Créer des tranches d'âge
    df_result['age_bin'] = pd.cut(df_result['age_at_freeze'],
                                    bins=[0, 20, 40, 60, 80, 100, 200],
                                    labels=['0-20', '20-40', '40-60', '60-80', '80-100', '100+'])

    df_age = analyze_errors_by_dimension(df_result, 'age_bin')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Distribution des erreurs par âge
    ax = axes[0]
    x = np.arange(len(df_age))
    width = 0.2

    ax.bar(x - width*1.5, df_age['TP'], width, label='TP', color='#2ecc71')
    ax.bar(x - width*0.5, df_age['FP'], width, label='FP', color='#e74c3c')
    ax.bar(x + width*0.5, df_age['FN'], width, label='FN', color='#f39c12')

    ax.set_xlabel('Tranche d\'âge (années)', fontsize=10)
    ax.set_ylabel('Nombre de tronçons', fontsize=10)
    ax.set_title('Distribution des Erreurs par Âge', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_age['age_bin'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Taux d'événements réels par âge
    ax = axes[1]
    ax.plot(df_age['age_bin'], df_age['event_rate'] * 100, marker='o',
            linewidth=2, markersize=8, color='#e74c3c')
    ax.set_xlabel('Tranche d\'âge (années)', fontsize=10)
    ax.set_ylabel('Taux de défaillance réel (%)', fontsize=10)
    ax.set_title('Taux de Défaillance par Âge', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xticklabels(df_age['age_bin'], rotation=45)

    # 3. Performance du modèle par âge
    ax = axes[2]
    ax.plot(df_age['age_bin'], df_age['precision'] * 100, marker='o',
            label='Précision', linewidth=2, markersize=8)
    ax.plot(df_age['age_bin'], df_age['recall'] * 100, marker='s',
            label='Rappel', linewidth=2, markersize=8)
    ax.set_xlabel('Tranche d\'âge (années)', fontsize=10)
    ax.set_ylabel('Performance (%)', fontsize=10)
    ax.set_title('Performance du Modèle par Âge', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticklabels(df_age['age_bin'], rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'errors_by_age.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Analyse par âge sauvegardée")


def plot_errors_by_diameter(df_result, output_dir):
    """Visualise les erreurs par diamètre."""
    # Créer des tranches de diamètre
    df_result['diam_bin'] = pd.cut(df_result['diametre'],
                                     bins=[0, 100, 150, 200, 300, 1000],
                                     labels=['<100', '100-150', '150-200', '200-300', '300+'])

    df_diam = analyze_errors_by_dimension(df_result, 'diam_bin')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Distribution des erreurs
    ax = axes[0]
    x = np.arange(len(df_diam))
    width = 0.25

    ax.bar(x - width, df_diam['FP'], width, label='FP (Faux Positif)', color='#e74c3c')
    ax.bar(x, df_diam['FN'], width, label='FN (Faux Négatif)', color='#f39c12')
    ax.bar(x + width, df_diam['TP'], width, label='TP (Vrai Positif)', color='#2ecc71')

    ax.set_xlabel('Diamètre (mm)', fontsize=10)
    ax.set_ylabel('Nombre de tronçons', fontsize=10)
    ax.set_title('Distribution des Erreurs par Diamètre', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_diam['diam_bin'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Taux de faux positifs par diamètre
    ax = axes[1]
    ax.bar(df_diam['diam_bin'], df_diam['fpr'] * 100, color='#e74c3c', edgecolor='black')
    ax.set_xlabel('Diamètre (mm)', fontsize=10)
    ax.set_ylabel('Taux de Faux Positifs (%)', fontsize=10)
    ax.set_title('Taux de Faux Positifs par Diamètre', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'errors_by_diameter.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Analyse par diamètre sauvegardée")


def analyze_error_characteristics(df_result):
    """Analyse les caractéristiques des erreurs (FP vs FN)."""
    fp_data = df_result[df_result['prediction_type'] == 'FP']
    fn_data = df_result[df_result['prediction_type'] == 'FN']
    tp_data = df_result[df_result['prediction_type'] == 'TP']

    comparison = {
        'Caractéristique': [],
        'Faux Positifs (FP)': [],
        'Faux Négatifs (FN)': [],
        'Vrais Positifs (TP)': []
    }

    # Âge moyen
    comparison['Caractéristique'].append('Âge moyen (années)')
    comparison['Faux Positifs (FP)'].append(f"{fp_data['age_at_freeze'].mean():.1f}")
    comparison['Faux Négatifs (FN)'].append(f"{fn_data['age_at_freeze'].mean():.1f}")
    comparison['Vrais Positifs (TP)'].append(f"{tp_data['age_at_freeze'].mean():.1f}")

    # Diamètre moyen
    comparison['Caractéristique'].append('Diamètre moyen (mm)')
    comparison['Faux Positifs (FP)'].append(f"{fp_data['diametre'].mean():.1f}")
    comparison['Faux Négatifs (FN)'].append(f"{fn_data['diametre'].mean():.1f}")
    comparison['Vrais Positifs (TP)'].append(f"{tp_data['diametre'].mean():.1f}")

    # Longueur moyenne
    comparison['Caractéristique'].append('Longueur moyenne (m)')
    comparison['Faux Positifs (FP)'].append(f"{fp_data['longueur'].mean():.1f}")
    comparison['Faux Négatifs (FN)'].append(f"{fn_data['longueur'].mean():.1f}")
    comparison['Vrais Positifs (TP)'].append(f"{tp_data['longueur'].mean():.1f}")

    # Nombre de fuites moyen
    if 'n_fuites_total' in df_result.columns:
        comparison['Caractéristique'].append('Nombre de fuites moyen')
        comparison['Faux Positifs (FP)'].append(f"{fp_data['n_fuites_total'].mean():.2f}")
        comparison['Faux Négatifs (FN)'].append(f"{fn_data['n_fuites_total'].mean():.2f}")
        comparison['Vrais Positifs (TP)'].append(f"{tp_data['n_fuites_total'].mean():.2f}")

    # Score de risque moyen
    comparison['Caractéristique'].append('Score de risque moyen')
    comparison['Faux Positifs (FP)'].append(f"{fp_data['y_score'].mean():.3f}")
    comparison['Faux Négatifs (FN)'].append(f"{fn_data['y_score'].mean():.3f}")
    comparison['Vrais Positifs (TP)'].append(f"{tp_data['y_score'].mean():.3f}")

    # Matériau le plus fréquent
    comparison['Caractéristique'].append('Matériau dominant')
    comparison['Faux Positifs (FP)'].append(fp_data['materiau'].mode()[0] if len(fp_data) > 0 else 'N/A')
    comparison['Faux Négatifs (FN)'].append(fn_data['materiau'].mode()[0] if len(fn_data) > 0 else 'N/A')
    comparison['Vrais Positifs (TP)'].append(tp_data['materiau'].mode()[0] if len(tp_data) > 0 else 'N/A')

    return pd.DataFrame(comparison)


def generate_error_report(stats, df_analysis_mat, df_analysis_decade,
                          df_comparison, metadata, output_path):
    """Génère un rapport markdown détaillé."""
    report = f"""# Rapport d'Analyse de Distribution des Erreurs

## Métadonnées du Modèle

- **Modèle**: {metadata['model_type']}
- **Horizon de prédiction**: {metadata['horizon_years']} an(s)
- **Date de freeze**: {metadata['freeze_date']}
- **Date de création**: {metadata['created_at']}

---

## 1. Vue d'Ensemble - Matrice de Confusion

| | Prédit Négatif | Prédit Positif |
|---|---|---|
| **Réel Négatif** | {stats['TN']:,} (TN) | {stats['FP']:,} (FP) |
| **Réel Positif** | {stats['FN']:,} (FN) | {stats['TP']:,} (TP) |

### Métriques Globales

- **Total de tronçons**: {stats['total']:,}
- **Accuracy**: {stats['accuracy']:.2%}
- **Precision**: {stats['precision']:.2%} (Parmi les prédictions positives, quelle proportion est correcte)
- **Recall (Sensibilité)**: {stats['recall']:.2%} (Parmi les événements réels, quelle proportion est détectée)
- **F1-Score**: {stats['f1_score']:.3f}
- **Taux de Faux Positifs (FPR)**: {stats['fpr']:.2%}
- **Taux de Faux Négatifs (FNR)**: {stats['fnr']:.2%}

### Répartition

- **Vrais Positifs (TP)**: {stats['TP']:,} ({stats['TP']/stats['total']*100:.1f}%)
- **Vrais Négatifs (TN)**: {stats['TN']:,} ({stats['TN']/stats['total']*100:.1f}%)
- **Faux Positifs (FP)**: {stats['FP']:,} ({stats['FP']/stats['total']*100:.1f}%)
  - *Tronçons prédits à risque mais qui n'ont pas eu de défaillance*
- **Faux Négatifs (FN)**: {stats['FN']:,} ({stats['FN']/stats['total']*100:.1f}%)
  - *Tronçons prédits sûrs mais qui ont eu une défaillance*

---

## 2. Distribution des Erreurs par Matériau

Les 10 matériaux les plus fréquents:

{df_analysis_mat.head(10).to_markdown(index=False, floatfmt='.2f')}

### Observations par Matériau

"""

    # Ajouter les observations par matériau
    for _, row in df_analysis_mat.head(5).iterrows():
        report += f"\n**{row['materiau']}**:\n"
        report += f"- Total: {row['n_total']:,} tronçons\n"
        report += f"- Événements réels: {row['n_events']:,} ({row['event_rate']*100:.1f}%)\n"
        report += f"- Faux Positifs: {row['FP']:,}\n"
        report += f"- Faux Négatifs: {row['FN']:,}\n"
        report += f"- Précision: {row['precision']*100:.1f}%\n"
        report += f"- Rappel: {row['recall']*100:.1f}%\n"

    report += f"""
---

## 3. Distribution des Erreurs par Décennie d'Installation

{df_analysis_decade.to_markdown(index=False, floatfmt='.2f')}

---

## 4. Caractéristiques Comparatives des Erreurs

Comparaison des caractéristiques moyennes entre les différents types de prédictions:

{df_comparison.to_markdown(index=False)}

### Insights Clés

"""

    # Calculer quelques insights
    fp_age = float(df_comparison[df_comparison['Caractéristique'] == 'Âge moyen (années)']['Faux Positifs (FP)'].values[0])
    fn_age = float(df_comparison[df_comparison['Caractéristique'] == 'Âge moyen (années)']['Faux Négatifs (FN)'].values[0])
    tp_age = float(df_comparison[df_comparison['Caractéristique'] == 'Âge moyen (années)']['Vrais Positifs (TP)'].values[0])

    report += f"""
1. **Faux Positifs (FP)**:
   - Le modèle prédit des défaillances pour {stats['FP']:,} tronçons qui ne défaillent pas
   - Âge moyen: {fp_age:.1f} ans
   - Impact: Surcoût de maintenance préventive inutile

2. **Faux Négatifs (FN)**:
   - Le modèle manque {stats['FN']:,} défaillances réelles
   - Âge moyen: {fn_age:.1f} ans
   - Impact: Risque de défaillances non anticipées

3. **Vrais Positifs (TP)**:
   - Le modèle détecte correctement {stats['TP']:,} défaillances
   - Âge moyen: {tp_age:.1f} ans
   - Performance: {stats['recall']*100:.1f}% des défaillances sont capturées

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

- **Coût FP**: Inspection/renouvellement de {stats['FP']:,} tronçons non nécessaires
- **Coût FN**: Défaillances non anticipées sur {stats['FN']:,} tronçons
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

*Rapport généré automatiquement le {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n✓ Rapport sauvegardé: {output_path}")


def main():
    """Fonction principale d'analyse."""
    print("="*60)
    print("ANALYSE DE LA DISTRIBUTION DES ERREURS")
    print("="*60)

    # Créer le répertoire de sortie
    output_dir = REPORTS_DIR / "plots"
    output_dir.mkdir(exist_ok=True, parents=True)

    # 1. Charger le modèle et les données
    print("\n1. Chargement du modèle et des données...")
    model, encoders, metadata, df, horizon = load_best_model_and_data()

    # 2. Calculer les prédictions et erreurs
    print("\n2. Calcul des prédictions et identification des erreurs...")
    df_result, y_scores = compute_predictions_and_errors(model, df, threshold=0.5)

    # 3. Statistiques de confusion
    print("\n3. Calcul de la matrice de confusion...")
    stats = compute_confusion_matrix_stats(df_result)

    print(f"\n   Matrice de Confusion:")
    print(f"   TP: {stats['TP']:,} | FP: {stats['FP']:,}")
    print(f"   FN: {stats['FN']:,} | TN: {stats['TN']:,}")
    print(f"\n   Accuracy: {stats['accuracy']:.2%}")
    print(f"   Precision: {stats['precision']:.2%}")
    print(f"   Recall: {stats['recall']:.2%}")

    # 4. Visualisations
    print("\n4. Génération des visualisations...")
    plot_confusion_matrix(stats, output_dir)
    plot_error_distribution(df_result, output_dir)

    # 5. Analyse par dimensions
    print("\n5. Analyse par dimensions...")

    # Par matériau
    if 'materiau' in df_result.columns:
        df_analysis_mat = analyze_errors_by_dimension(df_result, 'materiau')
        plot_errors_by_material(df_analysis_mat, output_dir)
    else:
        df_analysis_mat = pd.DataFrame()

    # Par âge
    plot_errors_by_age(df_result, output_dir)

    # Par diamètre
    plot_errors_by_diameter(df_result, output_dir)

    # Par décennie
    if 'decade_install' in df_result.columns:
        df_analysis_decade = analyze_errors_by_dimension(df_result, 'decade_install')
    else:
        df_analysis_decade = pd.DataFrame()

    # 6. Comparaison des caractéristiques
    print("\n6. Analyse comparative des erreurs...")
    df_comparison = analyze_error_characteristics(df_result)
    print("\n" + df_comparison.to_string(index=False))

    # 7. Génération du rapport
    print("\n7. Génération du rapport markdown...")
    report_path = REPORTS_DIR / "error_distribution_analysis.md"
    generate_error_report(stats, df_analysis_mat, df_analysis_decade,
                         df_comparison, metadata, report_path)

    # 8. Sauvegarder les données détaillées
    print("\n8. Sauvegarde des données détaillées...")
    df_result[['GID', 'materiau', 'age_at_freeze', 'diametre', 'longueur',
               'y_true', 'y_pred', 'y_score', 'prediction_type']].to_csv(
        REPORTS_DIR / "error_details.csv", index=False
    )
    print(f"   ✓ Détails sauvegardés: error_details.csv")

    print("\n" + "="*60)
    print("ANALYSE TERMINÉE")
    print("="*60)
    print(f"\nRapport principal: {report_path}")
    print(f"Visualisations: {output_dir}")
    print(f"Détails complets: {REPORTS_DIR / 'error_details.csv'}")

    return df_result, stats, df_analysis_mat


if __name__ == "__main__":
    main()
