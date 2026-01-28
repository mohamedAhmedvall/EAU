#!/usr/bin/env python3
"""
Generation complete de tous les graphiques pour le modele final v3
==================================================================
- Courbes de lift
- Calibration (reliability diagram, ECE)
- Distributions d'erreurs (FP, FN, TP, TN)
- Analyse par segment
- Feature importance
- ROC/PR curves
"""

import sys, os, warnings
from pathlib import Path

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))
os.chdir(SRC_DIR)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    brier_score_loss, average_precision_score
)
from sklearn.calibration import calibration_curve

from config import ARTIFACTS_DIR, REPORTS_DIR, RANDOM_SEED

# Output directory
PLOTS_DIR = REPORTS_DIR / "phase4_final" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'TP': '#2ecc71', 'FP': '#e74c3c', 'FN': '#9b59b6', 'TN': '#3498db'}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    df = pd.read_pickle(ARTIFACTS_DIR / "dataset_freeze_h1.pkl")

    for col in ["materiau", "decade_install"]:
        if col in df.columns and col + "_enc" not in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    df["age_cap"] = df["age_at_freeze"].clip(upper=110)
    df["age_extreme"] = (df["age_at_freeze"] > 110).astype(int)
    p99 = df["age_at_freeze"].quantile(0.99)
    df["age_winsor"] = df["age_at_freeze"].clip(upper=p99)

    dec_med = df.groupby("decade_install")["age_at_freeze"].median()
    df["decade_med"] = df["decade_install"].map(dec_med)
    df["age_ratio_dec"] = df["age_at_freeze"] / df["decade_med"].replace(0, np.nan)
    df["age_ratio_dec"] = df["age_ratio_dec"].fillna(1.0)

    mat_med = df.groupby("materiau")["age_at_freeze"].median()
    df["mat_med"] = df["materiau"].map(mat_med)
    df["age_resid_mat"] = df["age_at_freeze"] - df["mat_med"].fillna(df["age_at_freeze"].median())

    return df

FEATURES = [
    "age_at_freeze", "diametre", "longueur", "log_longueur", "log_diametre",
    "n_fuites_total", "n_fuites_1y", "n_fuites_3y", "n_fuites_5y",
    "days_since_last_fuite", "has_recent_fuite",
    "leak_rate_per_year", "leak_rate_per_km",
    "ratio_age_median", "overdue_years", "over_p75_life", "over_p90_life",
    "age_x_nfuites", "surface_approx", "age_x_ratio",
    "materiau_enc", "decade_install_enc",
    "age_cap", "age_extreme", "age_winsor", "age_ratio_dec", "age_resid_mat",
]

def get_X(df):
    cols = [c for c in FEATURES if c in df.columns]
    return df[cols].replace([np.inf, -np.inf], np.nan).fillna(0)

# ============================================================================
# 1. COURBES DE LIFT
# ============================================================================

def plot_lift_curves(y_true, scores, fname="lift_curves.png"):
    """Courbes de lift pour différents percentiles."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    y = np.asarray(y_true)
    s = np.asarray(scores)
    n = len(y)
    n_ev = y.sum()
    order = np.argsort(-s)
    y_sorted = y[order]

    # Lift curve
    ax = axes[0]
    pcts = np.arange(1, 51)
    lifts = []
    for p in pcts:
        k = int(np.ceil(n * p / 100))
        cap = y_sorted[:k].sum() / n_ev
        lifts.append(cap / (p / 100))

    ax.plot(pcts, lifts, 'b-', linewidth=2.5, label='Modèle v3_final')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='Aléatoire')
    ax.fill_between(pcts, 1, lifts, alpha=0.3)
    ax.set_xlabel('Top K%', fontsize=12)
    ax.set_ylabel('Lift', fontsize=12)
    ax.set_title('Courbe de Lift', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Annotate key points
    for k_pct in [5, 10, 20]:
        k = int(np.ceil(n * k_pct / 100))
        cap = y_sorted[:k].sum() / n_ev
        lift = cap / (k_pct / 100)
        ax.annotate(f'L@{k_pct}%={lift:.2f}', xy=(k_pct, lift),
                   xytext=(k_pct+3, lift+0.3), fontsize=10,
                   arrowprops=dict(arrowstyle='->', color='red'))

    # Capture curve (cumulative gains)
    ax = axes[1]
    captures = []
    for p in pcts:
        k = int(np.ceil(n * p / 100))
        cap = y_sorted[:k].sum() / n_ev
        captures.append(cap * 100)

    ax.plot(pcts, captures, 'g-', linewidth=2.5, label='Modèle v3_final')
    ax.plot(pcts, pcts, 'gray', linestyle='--', alpha=0.7, label='Aléatoire')
    ax.fill_between(pcts, pcts, captures, alpha=0.3, color='green')
    ax.set_xlabel('Top K%', fontsize=12)
    ax.set_ylabel('Capture (%)', fontsize=12)
    ax.set_title('Courbe de Capture (Gains Cumulés)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Annotate
    for k_pct in [5, 10, 20]:
        k = int(np.ceil(n * k_pct / 100))
        cap = y_sorted[:k].sum() / n_ev * 100
        ax.annotate(f'C@{k_pct}%={cap:.1f}%', xy=(k_pct, cap),
                   xytext=(k_pct+5, cap-5), fontsize=10,
                   arrowprops=dict(arrowstyle='->', color='red'))

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

# ============================================================================
# 2. CALIBRATION
# ============================================================================

def plot_calibration(y_true, scores, fname="calibration.png"):
    """Diagramme de fiabilité et analyse de calibration."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    y = np.asarray(y_true)
    s = np.asarray(scores)

    # Reliability diagram
    ax = axes[0]
    prob_true, prob_pred = calibration_curve(y, s, n_bins=10, strategy='quantile')

    ax.plot([0, 1], [0, 1], 'k--', label='Parfaitement calibré')
    ax.plot(prob_pred, prob_true, 'bo-', markersize=8, linewidth=2, label='Modèle v3_final')
    ax.fill_between(prob_pred, prob_pred, prob_true, alpha=0.3)
    ax.set_xlabel('Probabilité prédite moyenne', fontsize=12)
    ax.set_ylabel('Fraction positive observée', fontsize=12)
    ax.set_title('Diagramme de Fiabilité (Reliability)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Brier score
    brier = brier_score_loss(y, s)
    ax.text(0.05, 0.95, f'Brier Score: {brier:.4f}', transform=ax.transAxes,
            fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

    # ECE by bin
    ax = axes[1]
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_errors = []
    bin_counts = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (s >= lo) & (s < hi)
        if mask.sum() > 0:
            bin_centers.append((lo + hi) / 2)
            pred_mean = s[mask].mean()
            true_mean = y[mask].mean()
            bin_errors.append(abs(pred_mean - true_mean))
            bin_counts.append(mask.sum())

    colors = ['green' if e < 0.02 else 'orange' if e < 0.05 else 'red' for e in bin_errors]
    ax.bar(bin_centers, bin_errors, width=0.08, color=colors, edgecolor='black', alpha=0.7)
    ax.axhline(y=0.02, color='green', linestyle='--', alpha=0.5, label='Excellent (<2%)')
    ax.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Acceptable (<5%)')
    ax.set_xlabel('Probabilité prédite', fontsize=12)
    ax.set_ylabel('Erreur de calibration |pred - obs|', fontsize=12)
    ax.set_title('Erreur de Calibration par Bin', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # ECE total
    ece = sum(c * e for c, e in zip(bin_counts, bin_errors)) / sum(bin_counts)
    ax.text(0.95, 0.95, f'ECE: {ece:.4f}', transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat'))

    # Distribution des scores
    ax = axes[2]
    ax.hist(s[y == 0], bins=50, alpha=0.6, label=f'Négatifs (n={int((y==0).sum()):,})', color='blue', density=True)
    ax.hist(s[y == 1], bins=50, alpha=0.6, label=f'Positifs (n={int((y==1).sum()):,})', color='red', density=True)
    ax.set_xlabel('Score de risque', fontsize=12)
    ax.set_ylabel('Densité', fontsize=12)
    ax.set_title('Distribution des Scores', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

    return brier, ece

# ============================================================================
# 3. DISTRIBUTIONS D'ERREURS
# ============================================================================

def plot_error_distributions(y_true, scores, df_test, fname="error_distributions.png"):
    """Distribution des erreurs par type (TP, FP, FN, TN)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    y = np.asarray(y_true)
    s = np.asarray(scores)
    n = len(y)
    k = int(np.ceil(n * 0.10))
    order = np.argsort(-s)
    top = np.zeros(n, dtype=bool)
    top[order[:k]] = True

    tp = (y == 1) & top
    fp = (y == 0) & top
    fn = (y == 1) & ~top
    tn = (y == 0) & ~top

    # 1. Age distribution
    ax = axes[0, 0]
    for label, mask, color in [('TP', tp, COLORS['TP']), ('FP', fp, COLORS['FP']),
                                ('FN', fn, COLORS['FN'])]:
        if mask.any():
            ax.hist(df_test.loc[mask, 'age_at_freeze'], bins=30, alpha=0.5,
                   label=f'{label} (n={mask.sum()})', color=color, density=True)
    ax.set_xlabel('Âge (années)', fontsize=11)
    ax.set_ylabel('Densité', fontsize=11)
    ax.set_title('Distribution par Âge', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Score distribution
    ax = axes[0, 1]
    for label, mask, color in [('TP', tp, COLORS['TP']), ('FP', fp, COLORS['FP']),
                                ('FN', fn, COLORS['FN'])]:
        if mask.any():
            ax.hist(s[mask], bins=30, alpha=0.5, label=f'{label}', color=color, density=True)
    ax.axvline(x=s[order[k-1]], color='black', linestyle='--', label=f'Seuil Top 10%')
    ax.set_xlabel('Score de risque', fontsize=11)
    ax.set_ylabel('Densité', fontsize=11)
    ax.set_title('Distribution des Scores par Type', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Materiau
    ax = axes[0, 2]
    mat_data = []
    for mat in df_test['materiau'].unique():
        mask_mat = df_test['materiau'] == mat
        if mask_mat.sum() < 10:
            continue
        mat_data.append({
            'Matériau': mat,
            'TP': (tp & mask_mat).sum(),
            'FP': (fp & mask_mat).sum(),
            'FN': (fn & mask_mat).sum(),
        })
    if mat_data:
        mat_df = pd.DataFrame(mat_data).set_index('Matériau')
        mat_df[['TP', 'FP', 'FN']].plot(kind='bar', ax=ax,
                                         color=[COLORS['TP'], COLORS['FP'], COLORS['FN']])
        ax.set_title('Erreurs par Matériau', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    # 4. Décennie
    ax = axes[1, 0]
    dec_data = []
    for dec in sorted(df_test['decade_install'].unique()):
        mask_dec = df_test['decade_install'] == dec
        n_events = (y[mask_dec] == 1).sum()
        if n_events == 0:
            continue
        fn_rate = (fn & mask_dec).sum() / n_events
        capture = 1 - fn_rate
        dec_data.append({
            'Décennie': int(dec),
            'Capture': capture,
            'FN_rate': fn_rate,
            'Events': n_events
        })
    if dec_data:
        dec_df = pd.DataFrame(dec_data)
        colors = ['green' if c > 0.6 else 'orange' if c > 0.4 else 'red' for c in dec_df['Capture']]
        ax.bar(dec_df['Décennie'].astype(str), dec_df['Capture'], color=colors, edgecolor='black')
        ax.axhline(y=0.6, color='green', linestyle='--', alpha=0.5)
        ax.set_xlabel('Décennie d\'installation', fontsize=11)
        ax.set_ylabel('Taux de Capture', fontsize=11)
        ax.set_title('Capture par Décennie', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    # 5. N_fuites vs erreur
    ax = axes[1, 1]
    for label, mask, color in [('TP', tp, COLORS['TP']), ('FN', fn, COLORS['FN'])]:
        if mask.any():
            ax.scatter(df_test.loc[mask, 'n_fuites_total'], s[mask],
                      alpha=0.5, label=label, color=color, s=20)
    ax.set_xlabel('Nombre de fuites historiques', fontsize=11)
    ax.set_ylabel('Score de risque', fontsize=11)
    ax.set_title('Score vs Historique Anomalies', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Confusion matrix style
    ax = axes[1, 2]
    conf_data = [[tp.sum(), fp.sum()], [fn.sum(), tn.sum()]]
    conf_labels = [['TP\n(Détectés)', 'FP\n(Fausses alertes)'],
                   ['FN\n(Manqués)', 'TN\n(Corrects)']]

    im = ax.imshow([[1, 0], [0, 1]], cmap='RdYlGn', alpha=0.3)
    for i in range(2):
        for j in range(2):
            color = 'green' if (i == j and i == 0) or (i == j and i == 1 and j == 1) else 'red' if i != j else 'black'
            ax.text(j, i, f'{conf_labels[i][j]}\n{conf_data[i][j]:,}',
                   ha='center', va='center', fontsize=12, fontweight='bold')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Top 10%', 'Bottom 90%'])
    ax.set_yticklabels(['Événement', 'Pas d\'événement'])
    ax.set_title('Matrice de Confusion @10%', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

# ============================================================================
# 4. ROC & PR CURVES
# ============================================================================

def plot_roc_pr(y_true, scores, fname="roc_pr_curves.png"):
    """ROC et Precision-Recall curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Curve
    ax = axes[0]
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)

    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'Modèle (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Aléatoire')
    ax.fill_between(fpr, 0, tpr, alpha=0.3)
    ax.set_xlabel('Taux de Faux Positifs', fontsize=12)
    ax.set_ylabel('Taux de Vrais Positifs', fontsize=12)
    ax.set_title('Courbe ROC', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # PR Curve
    ax = axes[1]
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    baseline = y_true.mean()

    ax.plot(recall, precision, 'g-', linewidth=2, label=f'Modèle (AP = {ap:.3f})')
    ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.3f})')
    ax.fill_between(recall, baseline, precision, alpha=0.3, color='green')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Courbe Precision-Recall', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

# ============================================================================
# 5. FEATURE IMPORTANCE
# ============================================================================

def plot_feature_importance(model, feature_names, fname="feature_importance.png"):
    """Feature importance from the model."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print("  Model doesn't have feature_importances_, skipping...")
        return

    # Sort
    indices = np.argsort(importances)[::-1][:20]  # Top 20

    # Plot
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(indices)))
    ax.barh(range(len(indices)), importances[indices], color=colors)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Top 20 Features (Importance)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

# ============================================================================
# 6. SEGMENT ANALYSIS
# ============================================================================

def plot_segment_analysis(y_true, scores, df_test, fname="segment_analysis.png"):
    """Analyse détaillée par segment."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    y = np.asarray(y_true)
    s = np.asarray(scores)
    n = len(y)
    k = int(np.ceil(n * 0.10))
    order = np.argsort(-s)
    top = np.zeros(n, dtype=bool)
    top[order[:k]] = True

    # 1. Capture par matériau
    ax = axes[0, 0]
    mat_caps = []
    for mat in df_test['materiau'].unique():
        mask = df_test['materiau'] == mat
        events = (y[mask] == 1).sum()
        if events < 3:
            continue
        captured = ((y[mask] == 1) & top[mask]).sum()
        cap = captured / events
        mat_caps.append({'Matériau': mat, 'Capture': cap, 'Events': events})

    mat_caps = sorted(mat_caps, key=lambda x: x['Capture'], reverse=True)
    mats = [m['Matériau'] for m in mat_caps]
    caps = [m['Capture'] for m in mat_caps]
    colors = ['green' if c > 0.6 else 'orange' if c > 0.3 else 'red' for c in caps]

    bars = ax.barh(mats, caps, color=colors, edgecolor='black')
    ax.axvline(x=0.657, color='blue', linestyle='--', linewidth=2, label='Global (65.7%)')
    ax.set_xlabel('Taux de Capture @10%', fontsize=11)
    ax.set_title('Capture par Matériau', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    # Add event counts
    for bar, m in zip(bars, mat_caps):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
               f'n={m["Events"]}', va='center', fontsize=9)

    # 2. Score moyen par matériau
    ax = axes[0, 1]
    df_test_reset = df_test.reset_index(drop=True)
    df_test_reset['score'] = s
    df_test_reset['event'] = y
    mat_scores = df_test_reset.groupby('materiau').agg(
        score_moyen=('score', 'mean'),
        event_rate=('event', 'mean'),
        n=('score', 'size')
    ).reset_index()
    mat_scores = mat_scores[mat_scores['n'] > 50].sort_values('score_moyen', ascending=False)

    x_pos = np.arange(len(mat_scores))
    width = 0.35
    ax.bar(x_pos - width/2, mat_scores['score_moyen'], width, label='Score moyen', color='steelblue')
    ax.bar(x_pos + width/2, mat_scores['event_rate'], width, label='Taux événement réel', color='coral')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(mat_scores['materiau'], rotation=45, ha='right')
    ax.set_ylabel('Valeur', fontsize=11)
    ax.set_title('Score Prédit vs Taux Réel par Matériau', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Capture par décennie
    ax = axes[1, 0]
    dec_caps = []
    for dec in sorted(df_test['decade_install'].unique()):
        mask = df_test['decade_install'] == dec
        events = (y[mask] == 1).sum()
        if events < 3:
            continue
        captured = ((y[mask] == 1) & top[mask]).sum()
        cap = captured / events
        dec_caps.append({'Décennie': int(dec), 'Capture': cap, 'Events': events})

    decs = [str(d['Décennie']) for d in dec_caps]
    caps = [d['Capture'] for d in dec_caps]
    colors = ['green' if c > 0.6 else 'orange' if c > 0.4 else 'red' for c in caps]

    bars = ax.bar(decs, caps, color=colors, edgecolor='black')
    ax.axhline(y=0.657, color='blue', linestyle='--', linewidth=2, label='Global (65.7%)')
    ax.set_xlabel('Décennie', fontsize=11)
    ax.set_ylabel('Taux de Capture @10%', fontsize=11)
    ax.set_title('Capture par Décennie d\'Installation', fontsize=12, fontweight='bold')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # Add event counts
    for bar, d in zip(bars, dec_caps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'n={d["Events"]}', ha='center', fontsize=8)

    # 4. Heatmap score moyen par materiau x decennie
    ax = axes[1, 1]
    pivot = df_test_reset.pivot_table(
        values='score',
        index='materiau',
        columns='decade_install',
        aggfunc='mean'
    )
    pivot = pivot.dropna(how='all', axis=0).dropna(how='all', axis=1)

    if not pivot.empty:
        sns.heatmap(pivot, ax=ax, cmap='YlOrRd', annot=True, fmt='.3f',
                   cbar_kws={'label': 'Score moyen'})
        ax.set_title('Score Moyen par Matériau × Décennie', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

# ============================================================================
# 7. CALIBRATION POUR OPTIMISATION
# ============================================================================

def plot_calibration_for_optimization(y_true, scores, fname="calibration_optimization.png"):
    """Analyse spécifique calibration pour usage en optimisation."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    y = np.asarray(y_true)
    s = np.asarray(scores)

    # 1. Reliability diagram détaillé (20 bins)
    ax = axes[0, 0]
    prob_true, prob_pred = calibration_curve(y, s, n_bins=20, strategy='quantile')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Parfait')
    ax.scatter(prob_pred, prob_true, c='blue', s=100, zorder=5)
    ax.plot(prob_pred, prob_true, 'b-', linewidth=1, alpha=0.5)

    # Confidence interval approximation
    for pp, pt in zip(prob_pred, prob_true):
        # Wilson score interval approx
        n_bin = int(len(y) / 20)
        se = np.sqrt(pt * (1 - pt) / n_bin)
        ax.errorbar(pp, pt, yerr=1.96*se, color='blue', alpha=0.3, capsize=3)

    ax.set_xlabel('Probabilité prédite', fontsize=12)
    ax.set_ylabel('Probabilité observée', fontsize=12)
    ax.set_title('Diagramme de Fiabilité (20 bins)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Résidus de calibration
    ax = axes[0, 1]
    residuals = prob_true - prob_pred
    ax.bar(range(len(residuals)), residuals, color=['green' if r > 0 else 'red' for r in residuals])
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axhline(y=0.02, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=-0.02, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Bin', fontsize=12)
    ax.set_ylabel('Résidu (observé - prédit)', fontsize=12)
    ax.set_title('Résidus de Calibration', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. Expected vs Observed events par décile
    ax = axes[1, 0]
    deciles = pd.qcut(s, q=10, labels=False, duplicates='drop')
    decile_stats = []
    for d in sorted(set(deciles)):
        mask = deciles == d
        expected = s[mask].sum()  # Sum of probabilities
        observed = y[mask].sum()  # Actual events
        decile_stats.append({
            'Décile': d + 1,
            'Attendu': expected,
            'Observé': observed,
            'Ratio': observed / expected if expected > 0 else 0
        })

    dec_df = pd.DataFrame(decile_stats)
    x = np.arange(len(dec_df))
    width = 0.35
    ax.bar(x - width/2, dec_df['Attendu'], width, label='Événements attendus', color='steelblue')
    ax.bar(x + width/2, dec_df['Observé'], width, label='Événements observés', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels([f'D{d}' for d in dec_df['Décile']])
    ax.set_xlabel('Décile de risque', fontsize=12)
    ax.set_ylabel('Nombre d\'événements', fontsize=12)
    ax.set_title('Événements Attendus vs Observés par Décile', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Ratio O/E par décile
    ax = axes[1, 1]
    colors = ['green' if 0.8 <= r <= 1.2 else 'orange' if 0.6 <= r <= 1.4 else 'red'
              for r in dec_df['Ratio']]
    ax.bar(dec_df['Décile'], dec_df['Ratio'], color=colors, edgecolor='black')
    ax.axhline(y=1, color='black', linewidth=2, label='Parfait (O/E=1)')
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=1.2, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Décile de risque', fontsize=12)
    ax.set_ylabel('Ratio Observé/Attendu', fontsize=12)
    ax.set_title('Ratio O/E par Décile (1 = parfait)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Summary text
    total_expected = s.sum()
    total_observed = y.sum()
    overall_ratio = total_observed / total_expected

    fig.text(0.5, 0.02,
             f'Total: Attendu={total_expected:.1f}, Observé={total_observed}, Ratio O/E={overall_ratio:.3f}',
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

    return overall_ratio

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("GENERATION DE TOUS LES GRAPHIQUES")
    print("="*70)

    # Load data
    print("\nChargement des données...")
    df = load_data()

    # Split (same as training)
    idx = np.arange(len(df))
    i_tr, i_te = train_test_split(idx, test_size=0.2, random_state=RANDOM_SEED,
                                   stratify=df["event"].values)

    df_tr = df.iloc[i_tr]
    df_te = df.iloc[i_te]
    y_tr = df_tr["event"].values
    y_te = df_te["event"].values

    X_tr = get_X(df_tr)
    X_te = get_X(df_te)

    print(f"Test set: {len(df_te):,} samples, {y_te.sum():,} events")

    # Load or train model
    model_path = ARTIFACTS_DIR / "model_v3_final.joblib"
    if model_path.exists():
        print(f"\nChargement du modèle: {model_path}")
        artifact = joblib.load(model_path)
        model = artifact["model"]
        calibrator = artifact["calibrator"]

        # Get scores
        scores_raw = model.predict_proba(X_te)[:, 1]
        scores = calibrator.transform(scores_raw)
    else:
        print("\nModèle non trouvé, entraînement...")
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.isotonic import IsotonicRegression

        model = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.03, max_depth=None,
            min_samples_leaf=50, random_state=RANDOM_SEED
        )
        model.fit(X_tr, y_tr)

        scores_raw = model.predict_proba(X_te)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        scores_tr = model.predict_proba(X_tr)[:, 1]
        calibrator.fit(scores_tr, y_tr)
        scores = calibrator.transform(scores_raw)

    # Generate all plots
    print(f"\nGénération des graphiques dans: {PLOTS_DIR}")
    print("-" * 50)

    # 1. Lift curves
    plot_lift_curves(y_te, scores, "01_lift_curves.png")

    # 2. Calibration
    brier, ece = plot_calibration(y_te, scores, "02_calibration.png")

    # 3. Error distributions
    plot_error_distributions(y_te, scores, df_te, "03_error_distributions.png")

    # 4. ROC & PR
    plot_roc_pr(y_te, scores, "04_roc_pr_curves.png")

    # 5. Feature importance
    plot_feature_importance(model, FEATURES, "05_feature_importance.png")

    # 6. Segment analysis
    plot_segment_analysis(y_te, scores, df_te, "06_segment_analysis.png")

    # 7. Calibration for optimization
    oe_ratio = plot_calibration_for_optimization(y_te, scores, "07_calibration_optimization.png")

    # Summary
    print("\n" + "="*70)
    print("RÉSUMÉ CALIBRATION POUR OPTIMISATION")
    print("="*70)
    print(f"  Brier Score: {brier:.4f} (< 0.02 = excellent)")
    print(f"  ECE: {ece:.4f} (< 0.02 = excellent)")
    print(f"  Ratio O/E global: {oe_ratio:.3f} (1.0 = parfait)")
    print()

    if brier < 0.02 and ece < 0.02 and 0.9 <= oe_ratio <= 1.1:
        print("  >>> CALIBRATION EXCELLENTE - OK POUR OPTIMISATION <<<")
    elif brier < 0.05 and ece < 0.05 and 0.8 <= oe_ratio <= 1.2:
        print("  >>> CALIBRATION BONNE - UTILISABLE POUR OPTIMISATION <<<")
    else:
        print("  >>> ATTENTION: Calibration à améliorer avant optimisation <<<")

    print(f"\nGraphiques sauvegardés: {PLOTS_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
