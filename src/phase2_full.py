#!/usr/bin/env python3
"""
PHASE 2 — Amélioration itérative du modèle de renouvellement AEP
================================================================
M0: Baseline LightGBM (best_model.joblib)
M1: Corrections biais (âges extrêmes + abandons préventifs)
M2: Calibration + contraintes monotones + régularisation
M*: Modèle final retenu

Anti-leakage: toutes les features calculées <= freeze_date (protocole inchangé).
"""

import sys, json, warnings, os
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

import lightgbm as lgb

from config import ARTIFACTS_DIR, REPORTS_DIR, RANDOM_SEED, TOP_K_PERCENTAGES

DEFAULT_HORIZON = 1


def get_output_dirs(horizon: int):
    phase2_dir = REPORTS_DIR / f"phase2_h{horizon}"
    phase2_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = phase2_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return phase2_dir, plots_dir


def horizon_suffix(horizon: int) -> str:
    return "" if horizon == DEFAULT_HORIZON else f"_h{horizon}"

# ===================================================================
# 1. UTILITY FUNCTIONS
# ===================================================================

def topk_metrics(y_true, scores, k_list=TOP_K_PERCENTAGES):
    """Capture@K, Lift@K, Precision@K."""
    y = np.asarray(y_true)
    s = np.asarray(scores)
    n = len(y)
    n_ev = y.sum()
    order = np.argsort(-s)
    y_s = y[order]
    rows = []
    for k_pct in k_list:
        k = int(np.ceil(n * k_pct))
        ev_top = int(y_s[:k].sum())
        cap = ev_top / n_ev if n_ev > 0 else 0
        rows.append(dict(k_pct=k_pct, k=k, capture=cap, lift=cap/k_pct,
                         precision=ev_top/k, events_in_top=ev_top))
    return pd.DataFrame(rows)


def confusion_at_k(y_true, scores, k_pct=0.10):
    y = np.asarray(y_true)
    s = np.asarray(scores)
    n = len(y)
    k = int(np.ceil(n * k_pct))
    order = np.argsort(-s)
    top = np.zeros(n, dtype=bool)
    top[order[:k]] = True
    tp = int(((y == 1) & top).sum())
    fp = int(((y == 0) & top).sum())
    fn = int(((y == 1) & ~top).sum())
    tn = int(((y == 0) & ~top).sum())
    return dict(TP=tp, FP=fp, FN=fn, TN=tn,
                precision=tp/(tp+fp) if tp+fp else 0,
                recall=tp/(tp+fn) if tp+fn else 0,
                fn_rate=fn/(tp+fn) if tp+fn else 0)


def segment_capture(y_true, scores, groups, k_pct=0.10):
    """Capture@k par segment."""
    y = np.asarray(y_true)
    s = np.asarray(scores)
    n = len(y)
    k = int(np.ceil(n * k_pct))
    order = np.argsort(-s)
    top = np.zeros(n, dtype=bool)
    top[order[:k]] = True
    rows = []
    for g in sorted(set(groups)):
        mask = np.array(groups) == g
        ng = mask.sum()
        ev = int(y[mask].sum())
        if ng < 30 or ev == 0:
            continue
        ev_cap = int((y[mask] & top[mask]).sum())
        cap = ev_cap / ev
        rows.append(dict(segment=g, n=ng, events=ev, captured=ev_cap,
                         capture=cap, fn_rate=1-cap,
                         avg_score=float(s[mask].mean()),
                         std_score=float(s[mask].std())))
    return pd.DataFrame(rows)


def score_variance_by_segment(scores, groups):
    """Variance intra-segment du score — mesure de stabilité."""
    s = np.asarray(scores)
    rows = []
    for g in sorted(set(groups)):
        mask = np.array(groups) == g
        if mask.sum() < 30:
            continue
        rows.append(dict(segment=g, n=int(mask.sum()),
                         mean=float(s[mask].mean()),
                         std=float(s[mask].std()),
                         cv=float(s[mask].std()/max(s[mask].mean(), 1e-9))))
    return pd.DataFrame(rows)


def ece_score(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / len(y) * abs(y[mask].mean() - p[mask].mean())
    return ece


# ===================================================================
# 2. DATA LOADING & FEATURE PREPARATION
# ===================================================================

def load_data(horizon: int):
    dataset_path = ARTIFACTS_DIR / f"dataset_freeze_h{horizon}.pkl"
    try:
        df = pd.read_pickle(dataset_path)
    except Exception as exc:
        csv_path = ARTIFACTS_DIR / f"dataset_freeze_h{horizon}.csv"
        print(
            f"[WARN] Impossible de lire {dataset_path} ({exc}). "
            f"Chargement du CSV {csv_path}."
        )
        df = pd.read_csv(csv_path)
        if "event" in df.columns:
            df["event"] = df["event"].astype(int)
    print(
        f"Dataset charge (h{horizon}): {len(df):,} lignes, "
        f"{df['event'].sum():,} evenements ({100*df['event'].mean():.2f}%)"
    )
    return df


def encode_categoricals(df, cat_cols=("materiau", "decade_install", "diametre_bin")):
    """Encode les catégorielles et retourne les encodeurs."""
    df = df.copy()
    encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


BASE_NUM_FEATURES = [
    "age_at_freeze", "diametre", "longueur", "log_longueur", "log_diametre",
    "n_fuites_total", "n_fuites_1y", "n_fuites_3y", "n_fuites_5y",
    "days_since_last_fuite", "has_recent_fuite",
    "leak_rate_per_year", "leak_rate_per_km",
    "ratio_age_median", "overdue_years", "over_p75_life", "over_p90_life",
    "age_x_nfuites", "surface_approx", "age_x_ratio",
]
BASE_CAT_ENCODED = ["materiau_encoded", "decade_install_encoded"]
BASE_FEATURES = BASE_NUM_FEATURES + BASE_CAT_ENCODED


def get_Xy(df, feature_cols, target_col="event"):
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).copy()
    y = df[target_col].astype(int).values
    return X, y


def split_data(df, feature_cols, target_col="event"):
    X, y = get_Xy(df, feature_cols, target_col)
    idx = np.arange(len(df))
    X_tr, X_te, y_tr, y_te, i_tr, i_te = train_test_split(
        X, y, idx, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    return X_tr, X_te, y_tr, y_te, i_tr, i_te


# ===================================================================
# 3. AGE EXTREME FEATURES
# ===================================================================

def add_age_features(df, cap=110):
    """Ajoute features robustes pour âges extrêmes."""
    df = df.copy()
    # Conservative: capping
    df["age_cap"] = df["age_at_freeze"].clip(upper=cap)
    df["age_extreme_flag"] = (df["age_at_freeze"] > cap).astype(int)
    # Winsorisation percentile 99
    p99 = df["age_at_freeze"].quantile(0.99)
    df["age_winsor"] = df["age_at_freeze"].clip(upper=p99)
    # Structurelle: ratio par décennie
    dec_med = df.groupby("decade_install")["age_at_freeze"].median()
    df["decade_median_age"] = df["decade_install"].map(dec_med)
    df["age_ratio_decade"] = df["age_at_freeze"] / df["decade_median_age"].replace(0, np.nan)
    df["age_ratio_decade"] = df["age_ratio_decade"].fillna(1.0)
    # Résidu d'âge (écart à la médiane du groupe matériau)
    mat_med = df.groupby("materiau")["age_at_freeze"].median()
    df["mat_median_age"] = df["materiau"].map(mat_med)
    df["age_residual_mat"] = df["age_at_freeze"] - df["mat_median_age"].fillna(df["age_at_freeze"].median())
    return df


# ===================================================================
# 4. PREVENTIVE ABANDON DETECTION
# ===================================================================

def flag_preventive_abandon(df, ratio_thresh=0.6, min_age=15):
    """
    Heuristique abandon préventif:
    - event=1 (DHS dans l'horizon)
    - AUCUNE anomalie historique
    - ratio_age_median < seuil (jeune par rapport au matériau)
    - age absolu < min_age ans
    - pas de fuite récente
    """
    df = df.copy()
    cond = (
        (df["event"] == 1) &
        (df["n_fuites_total"] == 0) &
        (df["ratio_age_median"] < ratio_thresh) &
        (df["age_at_freeze"] < min_age) &
        (df["has_recent_fuite"] == 0)
    )
    df["is_preventive_abandon"] = cond.astype(int)
    n_prev = cond.sum()
    n_ev = df["event"].sum()
    print(f"  Abandons preventifs detectes: {n_prev} / {n_ev} evenements "
          f"({100*n_prev/max(n_ev,1):.1f}%)")
    return df


# ===================================================================
# 5. TRAINING
# ===================================================================

def train_lgbm(X_tr, y_tr, monotone=None, params_override=None):
    n_neg = (y_tr == 0).sum()
    n_pos = max((y_tr == 1).sum(), 1)
    params = dict(
        n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31,
        subsample=0.9, colsample_bytree=0.9, verbose=-1,
        random_state=RANDOM_SEED, scale_pos_weight=n_neg/n_pos)
    if monotone is not None:
        params["monotone_constraints"] = monotone
    if params_override:
        params.update(params_override)
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr)
    return model


def evaluate_model(model, X_te, y_te, df_te, label="model"):
    """Évalue un modèle et retourne dict complet de métriques."""
    scores = model.predict_proba(X_te)[:, 1]
    tk = topk_metrics(y_te, scores)
    conf = confusion_at_k(y_te, scores, 0.10)
    roc = roc_auc_score(y_te, scores)
    brier = brier_score_loss(y_te, scores)
    ec = ece_score(y_te, scores)

    # Segments
    seg_mat = segment_capture(y_te, scores,
                              df_te["materiau"].values if "materiau" in df_te.columns else ["NA"]*len(y_te))
    seg_dec = segment_capture(y_te, scores,
                              df_te["decade_install"].values if "decade_install" in df_te.columns else [0]*len(y_te))

    lift10 = float(tk.loc[tk["k_pct"]==0.10, "lift"].values[0])
    cap10 = float(tk.loc[tk["k_pct"]==0.10, "capture"].values[0])
    cap5 = float(tk.loc[tk["k_pct"]==0.05, "capture"].values[0])
    cap20 = float(tk.loc[tk["k_pct"]==0.20, "capture"].values[0])

    print(f"  [{label}] ROC={roc:.4f} Lift@10%={lift10:.2f} Cap@10%={cap10:.2%} "
          f"Brier={brier:.4f} ECE={ec:.4f} FN={conf['FN']}")

    return dict(
        label=label, roc_auc=roc, brier=brier, ece=ec,
        lift_5=float(tk.loc[tk["k_pct"]==0.05, "lift"].values[0]),
        lift_10=lift10, lift_20=float(tk.loc[tk["k_pct"]==0.20, "lift"].values[0]),
        cap_5=cap5, cap_10=cap10, cap_20=cap20,
        conf=conf, scores=scores, topk=tk,
        seg_materiau=seg_mat, seg_decade=seg_dec,
        model=model
    )


def cv_evaluate(df, feature_cols, target_col="event", label="cv",
                monotone=None, params_override=None, n_folds=5):
    """Cross-validation stratifiée pour robustesse."""
    X, y = get_Xy(df, feature_cols, target_col)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    lifts, caps, rocs, briers = [], [], [], []
    for fold, (tr_i, te_i) in enumerate(skf.split(X, y)):
        m = train_lgbm(X.iloc[tr_i], y[tr_i], monotone=monotone,
                       params_override=params_override)
        sc = m.predict_proba(X.iloc[te_i])[:, 1]
        tk = topk_metrics(y[te_i], sc)
        lifts.append(float(tk.loc[tk["k_pct"]==0.10, "lift"].values[0]))
        caps.append(float(tk.loc[tk["k_pct"]==0.10, "capture"].values[0]))
        rocs.append(roc_auc_score(y[te_i], sc))
        briers.append(brier_score_loss(y[te_i], sc))
    return dict(
        label=label,
        lift10_mean=np.mean(lifts), lift10_std=np.std(lifts),
        cap10_mean=np.mean(caps), cap10_std=np.std(caps),
        roc_mean=np.mean(rocs), roc_std=np.std(rocs),
        brier_mean=np.mean(briers), brier_std=np.std(briers),
    )


# ===================================================================
# 6. CALIBRATION
# ===================================================================

def calibrate(train_scores, train_labels, test_scores):
    """Isotonic + Platt calibration."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(train_scores, train_labels)
    iso_test = iso.transform(test_scores)

    platt = LogisticRegression(max_iter=1000)
    platt.fit(train_scores.reshape(-1, 1), train_labels)
    platt_test = platt.predict_proba(test_scores.reshape(-1, 1))[:, 1]

    return iso_test, platt_test, iso, platt


# ===================================================================
# 7. PLOTTING
# ===================================================================

def plot_comparison_bar(metrics_list, metric_key, title, fname, plots_dir: Path):
    """Bar chart comparant un metric entre modèles."""
    labels = [m["label"] for m in metrics_list]
    vals = [m[metric_key] for m in metrics_list]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, vals, color=["#2196F3", "#FF9800", "#4CAF50", "#E91E63",
                                        "#9C27B0", "#00BCD4", "#795548"][:len(labels)])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel(metric_key)
    plt.tight_layout()
    plt.savefig(plots_dir / fname, dpi=150)
    plt.close()


def plot_calibration_curves(y_true, scores_dict, fname, plots_dir: Path):
    """Reliability diagram for multiple score sets."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Parfaitement calibre")
    for name, sc in scores_dict.items():
        prob_true, prob_pred = calibration_curve(y_true, sc, n_bins=10, strategy="quantile")
        ax.plot(prob_pred, prob_true, "o-", label=name)
    ax.set_xlabel("Probabilite predite (moyenne par bin)")
    ax.set_ylabel("Fraction positive observee")
    ax.set_title("Courbe de calibration")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / fname, dpi=150)
    plt.close()


def plot_segment_comparison(seg_before, seg_after, col, fname, plots_dir: Path):
    """Comparer capture@10% par segment entre deux modèles."""
    merged = seg_before[["segment", "capture"]].merge(
        seg_after[["segment", "capture"]], on="segment", suffixes=("_before", "_after"))
    if merged.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(merged))
    w = 0.35
    ax.bar(x - w/2, merged["capture_before"], w, label="Avant", color="#2196F3")
    ax.bar(x + w/2, merged["capture_after"], w, label="Apres", color="#4CAF50")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["segment"], rotation=45, ha="right")
    ax.set_ylabel("Capture@10%")
    ax.set_title(f"Capture@10% par {col}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / fname, dpi=150)
    plt.close()


def plot_score_distributions(scores_dict, fname, plots_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, sc in scores_dict.items():
        ax.hist(sc, bins=50, alpha=0.5, label=name, density=True)
    ax.set_xlabel("Score")
    ax.set_ylabel("Densite")
    ax.set_title("Distribution des scores")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / fname, dpi=150)
    plt.close()


def plot_lift_curves(y_true, scores_dict, fname, plots_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    pcts = np.arange(1, 51)
    for name, sc in scores_dict.items():
        y = np.asarray(y_true)
        s = np.asarray(sc)
        n = len(y)
        n_ev = y.sum()
        order = np.argsort(-s)
        y_s = y[order]
        lifts = []
        for p in pcts:
            k = int(np.ceil(n * p / 100))
            cap = y_s[:k].sum() / n_ev if n_ev else 0
            lifts.append(cap / (p/100))
        ax.plot(pcts, lifts, label=name)
    ax.set_xlabel("Top K%")
    ax.set_ylabel("Lift")
    ax.set_title("Courbe de Lift")
    ax.legend()
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / fname, dpi=150)
    plt.close()


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        headers = [str(col) for col in df.columns]
        sep = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        rows = []
        for row in df.itertuples(index=False):
            values = []
            for val in row:
                if isinstance(val, float):
                    values.append(f"{val:.4f}")
                else:
                    values.append(str(val))
            rows.append("| " + " | ".join(values) + " |")
        return "\n".join([sep, sep_line] + rows)


# ===================================================================
# 8. MAIN PIPELINE
# ===================================================================

def main(horizon: int = DEFAULT_HORIZON):
    print("=" * 70)
    print("PHASE 2 — AMELIORATION ITERATIVE DU MODELE")
    print("=" * 70)
    print(f"Horizon analyse: {horizon} an(s)")

    phase2_dir, plots_dir = get_output_dirs(horizon)

    # ----- LOAD DATA -----
    df_raw = load_data(horizon)
    df = add_age_features(df_raw)
    df = flag_preventive_abandon(df)
    df, encoders_base = encode_categoricals(df)

    # =====================================================================
    # DIAGNOSTIC PRE-ITERATION: Analyse ages extremes & abandons
    # =====================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: Ages extremes & abandons preventifs")
    print("=" * 70)

    # Age distribution
    for thresh in [80, 100, 110, 120]:
        n_old = (df["age_at_freeze"] > thresh).sum()
        ev_old = df.loc[df["age_at_freeze"] > thresh, "event"].sum()
        print(f"  Age > {thresh}: {n_old:,} pipes, {ev_old} evenements "
              f"(rate={100*ev_old/max(n_old,1):.2f}%)")

    n_prev = df["is_preventive_abandon"].sum()
    print(f"\n  Abandons preventifs: {n_prev} / {df['event'].sum()} evenements")
    print(f"  Event rate global: {df['event'].mean():.4f}")

    # =====================================================================
    # M0: BASELINE
    # =====================================================================
    print("\n" + "=" * 70)
    print("ITERATION 0 — M0 BASELINE")
    print("=" * 70)

    features_m0 = [f for f in BASE_FEATURES if f in df.columns]
    X_tr, X_te, y_tr, y_te, i_tr, i_te = split_data(df, features_m0, "event")
    df_te = df.iloc[i_te].copy()

    model_m0 = train_lgbm(X_tr, y_tr)
    res_m0 = evaluate_model(model_m0, X_te, y_te, df_te, "M0_baseline")

    # CV for robustness
    cv_m0 = cv_evaluate(df, features_m0, "event", "M0_cv")
    print(f"  [M0 CV] Lift@10%={cv_m0['lift10_mean']:.2f}+/-{cv_m0['lift10_std']:.2f} "
          f"ROC={cv_m0['roc_mean']:.4f}+/-{cv_m0['roc_std']:.4f}")

    # =====================================================================
    # ITERATION 1 — M1: CORRECTIONS BIAIS
    # =====================================================================
    print("\n" + "=" * 70)
    print("ITERATION 1 — M1: CORRECTIONS BIAIS AGES + ABANDONS")
    print("=" * 70)

    # --- M1a: Conservative (capping age + abandon filter) ---
    print("\n--- M1a: Conservative (age_cap + abandon label filter) ---")
    features_m1a = features_m0 + ["age_cap", "age_extreme_flag"]
    features_m1a = [f for f in features_m1a if f in df.columns]
    # Remove duplicates
    features_m1a = list(dict.fromkeys(features_m1a))

    df["event_m1a"] = df["event"].copy()
    df.loc[df["is_preventive_abandon"] == 1, "event_m1a"] = 0

    X_tr1a, X_te1a, y_tr1a, y_te1a, i_tr1a, i_te1a = split_data(
        df, features_m1a, "event_m1a")
    df_te1a = df.iloc[i_te1a].copy()

    model_m1a = train_lgbm(X_tr1a, y_tr1a)
    # Evaluate on ORIGINAL event (not modified) for fair FN comparison
    y_te_orig_1a = df.iloc[i_te1a]["event"].values
    res_m1a = evaluate_model(model_m1a, X_te1a, y_te_orig_1a, df_te1a, "M1a_conservative")

    cv_m1a = cv_evaluate(df, features_m1a, "event_m1a", "M1a_cv")
    print(f"  [M1a CV] Lift@10%={cv_m1a['lift10_mean']:.2f}+/-{cv_m1a['lift10_std']:.2f}")

    # --- M1b: Structural (age_ratio_decade + age_residual + abandon filter) ---
    print("\n--- M1b: Structural (age_ratio_decade + age_residual + abandon filter) ---")
    features_m1b = features_m1a + ["age_ratio_decade", "age_residual_mat", "age_winsor"]
    features_m1b = list(dict.fromkeys([f for f in features_m1b if f in df.columns]))

    X_tr1b, X_te1b, y_tr1b, y_te1b, i_tr1b, i_te1b = split_data(
        df, features_m1b, "event_m1a")
    df_te1b = df.iloc[i_te1b].copy()

    model_m1b = train_lgbm(X_tr1b, y_tr1b)
    y_te_orig_1b = df.iloc[i_te1b]["event"].values
    res_m1b = evaluate_model(model_m1b, X_te1b, y_te_orig_1b, df_te1b, "M1b_structural")

    cv_m1b = cv_evaluate(df, features_m1b, "event_m1a", "M1b_cv")
    print(f"  [M1b CV] Lift@10%={cv_m1b['lift10_mean']:.2f}+/-{cv_m1b['lift10_std']:.2f}")

    # --- M1c: Competing risks (multi-class: 0=no event, 1=real failure, 2=preventive) ---
    print("\n--- M1c: Competing risks (multi-class) ---")
    df["event_class"] = 0
    df.loc[(df["event"] == 1) & (df["is_preventive_abandon"] == 0), "event_class"] = 1
    df.loc[df["is_preventive_abandon"] == 1, "event_class"] = 2

    X_tr1c, X_te1c, y_tr1c, y_te1c, i_tr1c, i_te1c = split_data(
        df, features_m1b, "event_class")
    df_te1c = df.iloc[i_te1c].copy()

    params_mc = dict(n_estimators=260, learning_rate=0.05, max_depth=6,
                     num_leaves=31, subsample=0.9, colsample_bytree=0.9,
                     random_state=RANDOM_SEED, objective="multiclass", num_class=3,
                     verbose=-1)
    model_m1c = lgb.LGBMClassifier(**params_mc)
    model_m1c.fit(X_tr1c, y_tr1c)
    proba_mc = model_m1c.predict_proba(X_te1c)
    scores_m1c = proba_mc[:, 1]  # P(real failure)
    y_te_orig_1c = df.iloc[i_te1c]["event"].values

    # Manual evaluation for competing risks
    tk_1c = topk_metrics(y_te_orig_1c, scores_m1c)
    conf_1c = confusion_at_k(y_te_orig_1c, scores_m1c, 0.10)
    roc_1c = roc_auc_score((df.iloc[i_te1c]["event_class"] == 1).astype(int), scores_m1c)
    lift10_1c = float(tk_1c.loc[tk_1c["k_pct"]==0.10, "lift"].values[0])
    cap10_1c = float(tk_1c.loc[tk_1c["k_pct"]==0.10, "capture"].values[0])
    brier_1c = brier_score_loss(y_te_orig_1c, scores_m1c)
    print(f"  [M1c_competing] ROC={roc_1c:.4f} Lift@10%={lift10_1c:.2f} "
          f"Cap@10%={cap10_1c:.2%} FN={conf_1c['FN']}")

    seg_1c_mat = segment_capture(y_te_orig_1c, scores_m1c,
                                  df_te1c["materiau"].values)
    seg_1c_dec = segment_capture(y_te_orig_1c, scores_m1c,
                                  df_te1c["decade_install"].values)
    res_m1c = dict(label="M1c_competing", roc_auc=roc_1c, brier=brier_1c,
                   ece=ece_score(y_te_orig_1c, scores_m1c),
                   lift_10=lift10_1c, cap_10=cap10_1c,
                   lift_5=float(tk_1c.loc[tk_1c["k_pct"]==0.05, "lift"].values[0]),
                   lift_20=float(tk_1c.loc[tk_1c["k_pct"]==0.20, "lift"].values[0]),
                   cap_5=float(tk_1c.loc[tk_1c["k_pct"]==0.05, "capture"].values[0]),
                   cap_20=float(tk_1c.loc[tk_1c["k_pct"]==0.20, "capture"].values[0]),
                   conf=conf_1c, scores=scores_m1c, topk=tk_1c,
                   seg_materiau=seg_1c_mat, seg_decade=seg_1c_dec,
                   model=model_m1c)

    # --- M1 DECISION ---
    m1_candidates = {"M1a": res_m1a, "M1b": res_m1b, "M1c": res_m1c}
    m1_best_name = max(m1_candidates, key=lambda k: m1_candidates[k]["lift_10"])
    res_m1 = m1_candidates[m1_best_name]
    print(f"\n  >>> M1 retenu: {m1_best_name} (Lift@10%={res_m1['lift_10']:.2f})")

    if res_m1["lift_10"] < res_m0["lift_10"]:
        print(f"  ATTENTION: M1 degrade Lift@10% ({res_m1['lift_10']:.2f} < {res_m0['lift_10']:.2f})")
        print(f"  On garde M1 si amelioration FN rate ou stabilite segments")
        fn_m0 = res_m0["conf"]["FN"]
        fn_m1 = res_m1["conf"]["FN"]
        if fn_m1 < fn_m0:
            print(f"  -> FN ameliore: {fn_m0} -> {fn_m1}. M1 ACCEPTE.")
        else:
            print(f"  -> FN non ameliore. On utilise M0 comme base pour M2.")
            res_m1 = res_m0
            m1_best_name = "M0"

    # Features retained for M2
    if m1_best_name == "M1a":
        features_m1_best = features_m1a
        event_col_m1 = "event_m1a"
    elif m1_best_name == "M1b":
        features_m1_best = features_m1b
        event_col_m1 = "event_m1a"
    elif m1_best_name == "M1c":
        features_m1_best = features_m1b
        event_col_m1 = "event_m1a"  # M2 uses binary, not multi-class
    else:
        features_m1_best = features_m0
        event_col_m1 = "event"

    # =====================================================================
    # ITERATION 2 — M2: CALIBRATION + CONTRAINTES MONOTONES
    # =====================================================================
    print("\n" + "=" * 70)
    print("ITERATION 2 — M2: CALIBRATION + CONTRAINTES MONOTONES")
    print("=" * 70)

    # Build monotone constraints
    mono = []
    for col in features_m1_best:
        if col in ["age_at_freeze", "age_cap", "age_winsor",
                    "ratio_age_median", "overdue_years",
                    "over_p75_life", "over_p90_life",
                    "n_fuites_total", "n_fuites_1y", "n_fuites_3y", "n_fuites_5y",
                    "leak_rate_per_year", "leak_rate_per_km",
                    "age_x_nfuites", "age_x_ratio",
                    "age_ratio_decade", "age_residual_mat",
                    "age_extreme_flag",
                    "surface_approx", "longueur", "diametre",
                    "log_longueur", "log_diametre"]:
            mono.append(1)
        elif col == "days_since_last_fuite":
            mono.append(-1)
        else:
            mono.append(0)

    params_m2 = dict(
        n_estimators=250, learning_rate=0.04, max_depth=5, num_leaves=25,
        subsample=0.85, colsample_bytree=0.85,
        reg_lambda=2.0, reg_alpha=0.3, min_child_samples=50,
        random_state=RANDOM_SEED, verbose=-1)

    X_tr2, X_te2, y_tr2, y_te2, i_tr2, i_te2 = split_data(
        df, features_m1_best, event_col_m1)
    df_te2 = df.iloc[i_te2].copy()
    y_te_orig_2 = df.iloc[i_te2]["event"].values

    # M2a: with monotone constraints
    print("\n--- M2a: Contraintes monotones + regularisation ---")
    model_m2a = train_lgbm(X_tr2, y_tr2, monotone=mono, params_override=params_m2)
    res_m2a = evaluate_model(model_m2a, X_te2, y_te_orig_2, df_te2, "M2a_monotone")

    cv_m2a = cv_evaluate(df, features_m1_best, event_col_m1, "M2a_cv",
                         monotone=mono, params_override=params_m2)
    print(f"  [M2a CV] Lift@10%={cv_m2a['lift10_mean']:.2f}+/-{cv_m2a['lift10_std']:.2f}")

    # M2b: without monotone (just regularisation)
    print("\n--- M2b: Regularisation seule (sans monotone) ---")
    model_m2b = train_lgbm(X_tr2, y_tr2, params_override=params_m2)
    res_m2b = evaluate_model(model_m2b, X_te2, y_te_orig_2, df_te2, "M2b_regul_only")

    # --- Calibration on best M2 raw ---
    m2_raw_candidates = {"M2a": res_m2a, "M2b": res_m2b}
    m2_raw_best_name = max(m2_raw_candidates, key=lambda k: m2_raw_candidates[k]["lift_10"])
    res_m2_raw = m2_raw_candidates[m2_raw_best_name]
    model_m2_raw = res_m2_raw["model"]

    print(f"\n--- Calibration sur {m2_raw_best_name} ---")
    scores_tr_raw = model_m2_raw.predict_proba(X_tr2)[:, 1]
    scores_te_raw = res_m2_raw["scores"]

    iso_scores, platt_scores, iso_model, platt_model = calibrate(
        scores_tr_raw, y_tr2, scores_te_raw)

    # Evaluate calibrated versions
    for cal_name, cal_scores in [("iso", iso_scores), ("platt", platt_scores)]:
        tk_cal = topk_metrics(y_te_orig_2, cal_scores)
        lift10_cal = float(tk_cal.loc[tk_cal["k_pct"]==0.10, "lift"].values[0])
        cap10_cal = float(tk_cal.loc[tk_cal["k_pct"]==0.10, "capture"].values[0])
        brier_cal = brier_score_loss(y_te_orig_2, cal_scores)
        ece_cal = ece_score(y_te_orig_2, cal_scores)
        conf_cal = confusion_at_k(y_te_orig_2, cal_scores, 0.10)
        print(f"  [{m2_raw_best_name}+{cal_name}] Lift@10%={lift10_cal:.2f} "
              f"Brier={brier_cal:.4f} ECE={ece_cal:.4f}")

    # Best M2 (calibrated or raw by Lift@10%)
    cal_results = {
        f"{m2_raw_best_name}_raw": (scores_te_raw, res_m2_raw["brier"], res_m2_raw["ece"]),
        f"{m2_raw_best_name}_iso": (iso_scores,
            brier_score_loss(y_te_orig_2, iso_scores),
            ece_score(y_te_orig_2, iso_scores)),
        f"{m2_raw_best_name}_platt": (platt_scores,
            brier_score_loss(y_te_orig_2, platt_scores),
            ece_score(y_te_orig_2, platt_scores)),
    }

    # Pick calibration that preserves ranking best
    best_cal_name = None
    best_cal_lift = 0
    cal_metrics = {}
    for cname, (cscores, cbrier, cece) in cal_results.items():
        tk_c = topk_metrics(y_te_orig_2, cscores)
        l10 = float(tk_c.loc[tk_c["k_pct"]==0.10, "lift"].values[0])
        c10 = float(tk_c.loc[tk_c["k_pct"]==0.10, "capture"].values[0])
        cal_metrics[cname] = dict(lift_10=l10, cap_10=c10, brier=cbrier, ece=cece,
                                  scores=cscores)
        if l10 > best_cal_lift:
            best_cal_lift = l10
            best_cal_name = cname

    print(f"\n  >>> M2 calibration retenue: {best_cal_name} "
          f"(Lift@10%={best_cal_lift:.2f})")

    # Build final M2 result
    best_cal_scores = cal_metrics[best_cal_name]["scores"]
    seg_m2_mat = segment_capture(y_te_orig_2, best_cal_scores,
                                  df_te2["materiau"].values)
    seg_m2_dec = segment_capture(y_te_orig_2, best_cal_scores,
                                  df_te2["decade_install"].values)
    conf_m2 = confusion_at_k(y_te_orig_2, best_cal_scores, 0.10)

    res_m2 = dict(
        label=best_cal_name,
        roc_auc=roc_auc_score(y_te_orig_2, best_cal_scores),
        brier=cal_metrics[best_cal_name]["brier"],
        ece=cal_metrics[best_cal_name]["ece"],
        lift_10=best_cal_lift,
        cap_10=cal_metrics[best_cal_name]["cap_10"],
        lift_5=float(topk_metrics(y_te_orig_2, best_cal_scores).loc[
            topk_metrics(y_te_orig_2, best_cal_scores)["k_pct"]==0.05, "lift"].values[0]),
        lift_20=float(topk_metrics(y_te_orig_2, best_cal_scores).loc[
            topk_metrics(y_te_orig_2, best_cal_scores)["k_pct"]==0.20, "lift"].values[0]),
        cap_5=float(topk_metrics(y_te_orig_2, best_cal_scores).loc[
            topk_metrics(y_te_orig_2, best_cal_scores)["k_pct"]==0.05, "capture"].values[0]),
        cap_20=float(topk_metrics(y_te_orig_2, best_cal_scores).loc[
            topk_metrics(y_te_orig_2, best_cal_scores)["k_pct"]==0.20, "capture"].values[0]),
        conf=conf_m2, scores=best_cal_scores,
        seg_materiau=seg_m2_mat, seg_decade=seg_m2_dec,
        model=model_m2_raw
    )

    # =====================================================================
    # ITERATION 3 — COMPARAISON FINALE & CHOIX M*
    # =====================================================================
    print("\n" + "=" * 70)
    print("ITERATION 3 — COMPARAISON FINALE M0 vs M1 vs M2")
    print("=" * 70)

    all_models = [res_m0, res_m1, res_m2]
    labels_all = [r["label"] for r in all_models]

    # Summary table
    rows_summary = []
    for r in all_models:
        rows_summary.append(dict(
            Model=r["label"],
            ROC_AUC=r["roc_auc"],
            Lift_5=r.get("lift_5", 0),
            Lift_10=r["lift_10"],
            Lift_20=r.get("lift_20", 0),
            Cap_5=r.get("cap_5", 0),
            Cap_10=r["cap_10"],
            Cap_20=r.get("cap_20", 0),
            Brier=r["brier"],
            ECE=r["ece"],
            FN=r["conf"]["FN"],
            FP=r["conf"]["FP"],
        ))
    df_summary = pd.DataFrame(rows_summary)
    print("\n" + df_summary.to_string(index=False))
    df_summary.to_csv(phase2_dir / "comparison_M0_M1_M2.csv", index=False)

    # --- CHOOSE M* ---
    # Priority: Lift@10% (must not degrade), then Brier, then FN
    best_model_res = max(all_models, key=lambda r: (r["lift_10"], -r["brier"]))
    print(f"\n  >>> MODELE RETENU M*: {best_model_res['label']}")
    print(f"      Lift@10%={best_model_res['lift_10']:.2f} "
          f"Cap@10%={best_model_res['cap_10']:.2%} "
          f"Brier={best_model_res['brier']:.4f} "
          f"FN={best_model_res['conf']['FN']}")

    # =====================================================================
    # SEGMENT STABILITY ANALYSIS
    # =====================================================================
    print("\n" + "=" * 70)
    print("ANALYSE STABILITE PAR SEGMENTS")
    print("=" * 70)

    for seg_name in ["seg_materiau", "seg_decade"]:
        print(f"\n--- {seg_name} ---")
        for r in all_models:
            if seg_name in r and r[seg_name] is not None and not r[seg_name].empty:
                print(f"\n  [{r['label']}]")
                print(r[seg_name][["segment", "n", "events", "capture", "fn_rate",
                                    "avg_score"]].to_string(index=False))

    # =====================================================================
    # PLOTS
    # =====================================================================
    print("\n  Generation des graphiques...")

    # Comparison bars
    plot_comparison_bar(
        all_models,
        "lift_10",
        "Lift@10% — M0 vs M1 vs M2",
        "comparison_lift10.png",
        plots_dir,
    )
    plot_comparison_bar(
        all_models,
        "cap_10",
        "Capture@10% — M0 vs M1 vs M2",
        "comparison_cap10.png",
        plots_dir,
    )
    plot_comparison_bar(
        all_models,
        "brier",
        "Brier Score — M0 vs M1 vs M2",
        "comparison_brier.png",
        plots_dir,
    )
    plot_comparison_bar(
        all_models,
        "ece",
        "ECE — M0 vs M1 vs M2",
        "comparison_ece.png",
        plots_dir,
    )

    # Calibration curves
    plot_calibration_curves(
        y_te_orig_2,
        {
            "M0 (raw)": res_m0["scores"],
            f"{m2_raw_best_name} (raw)": scores_te_raw,
            f"{m2_raw_best_name} (iso)": iso_scores,
            f"{m2_raw_best_name} (platt)": platt_scores,
        },
        "calibration_curves.png",
        plots_dir,
    )

    # Score distributions
    plot_score_distributions(
        {r["label"]: r["scores"] for r in all_models},
        "score_distributions.png",
        plots_dir,
    )

    # Lift curves
    plot_lift_curves(
        y_te_orig_2,
        {r["label"]: r["scores"] for r in all_models},
        "lift_curves.png",
        plots_dir,
    )

    # Segment comparisons
    if not res_m0["seg_materiau"].empty and not best_model_res["seg_materiau"].empty:
        plot_segment_comparison(
            res_m0["seg_materiau"],
            best_model_res["seg_materiau"],
            "materiau",
            "segment_materiau_before_after.png",
            plots_dir,
        )
    if not res_m0["seg_decade"].empty and not best_model_res["seg_decade"].empty:
        plot_segment_comparison(
            res_m0["seg_decade"],
            best_model_res["seg_decade"],
            "decade",
            "segment_decade_before_after.png",
            plots_dir,
        )

    # Score variance by segment
    print("\n  Stabilite intra-segment (variance scores):")
    for r in all_models:
        sv = score_variance_by_segment(r["scores"],
            df.iloc[i_te2 if r["label"] != "M0_baseline" else i_te]["materiau"].values
            if len(r["scores"]) == len(i_te2 if r["label"] != "M0_baseline" else i_te) else
            ["NA"] * len(r["scores"]))
        if not sv.empty:
            print(f"\n  [{r['label']}] Mean CV: {sv['cv'].mean():.3f}")

    # =====================================================================
    # SAVE ARTIFACTS
    # =====================================================================
    print("\n" + "=" * 70)
    print("SAUVEGARDE DES ARTEFACTS")
    print("=" * 70)

    suffix = horizon_suffix(horizon)

    # M0
    artifact_m0 = dict(
        model=model_m0,
        encoders=encoders_base,
        metadata=dict(version="M0_baseline", horizon_years=horizon, feature_cols=features_m0),
    )
    joblib.dump(artifact_m0, ARTIFACTS_DIR / f"model_M0{suffix}.joblib")

    # Best M1
    model_m1_final = res_m1["model"]
    artifact_m1 = dict(
        model=model_m1_final,
        encoders=encoders_base,
        metadata=dict(
            version=m1_best_name,
            horizon_years=horizon,
            feature_cols=features_m1_best,
            abandon_filter=True,
            abandon_heuristic="no_anom + ratio<0.6 + age<15",
        ),
    )
    joblib.dump(artifact_m1, ARTIFACTS_DIR / f"model_M1_best{suffix}.joblib")

    # M2
    calibrator = None
    cal_type = None
    if "iso" in best_cal_name:
        calibrator = iso_model
        cal_type = "isotonic"
    elif "platt" in best_cal_name:
        calibrator = platt_model
        cal_type = "platt"

    artifact_m2 = dict(
        model=model_m2_raw,
        encoders=encoders_base,
        calibrator=dict(type=cal_type, model=calibrator) if calibrator else None,
        metadata=dict(
            version=best_cal_name,
            horizon_years=horizon,
            feature_cols=features_m1_best,
            monotone_constraints=mono if m2_raw_best_name == "M2a" else None,
            calibration=cal_type,
            regularization=params_m2,
        ),
    )
    joblib.dump(artifact_m2, ARTIFACTS_DIR / f"model_M2_best{suffix}.joblib")

    # M* (final)
    if best_model_res["label"] == res_m0["label"]:
        joblib.dump(artifact_m0, ARTIFACTS_DIR / f"model_M_star{suffix}.joblib")
        star_features = features_m0
    elif best_model_res["label"] == res_m1["label"]:
        joblib.dump(artifact_m1, ARTIFACTS_DIR / f"model_M_star{suffix}.joblib")
        star_features = features_m1_best
    else:
        joblib.dump(artifact_m2, ARTIFACTS_DIR / f"model_M_star{suffix}.joblib")
        star_features = features_m1_best

    # Metadata JSON
    metadata_final = dict(
        model_retenu=best_model_res["label"],
        horizon_years=horizon,
        features=star_features,
        calibration=cal_type if best_model_res["label"] == res_m2["label"] else None,
        metrics=dict(
            roc_auc=best_model_res["roc_auc"],
            lift_10=best_model_res["lift_10"],
            cap_10=best_model_res["cap_10"],
            brier=best_model_res["brier"],
            ece=best_model_res["ece"],
            fn=best_model_res["conf"]["FN"],
            fp=best_model_res["conf"]["FP"],
        ),
        risque_attendu="proba_calibree * longueur_km",
        pret_optimisation=True,
    )
    with open(ARTIFACTS_DIR / f"metadata_phase2{suffix}.json", "w") as f:
        json.dump(metadata_final, f, indent=2, default=str)

    print(f"  Artefacts sauvegardes dans {ARTIFACTS_DIR}")

    # =====================================================================
    # GENERATE REPORT
    # =====================================================================
    print("\n  Generation du rapport Phase 2...")

    report = []
    report.append("# Rapport PHASE 2 — Amelioration du modele de renouvellement\n")
    report.append(f"*Genere automatiquement (horizon: {horizon} an(s))*\n")
    report.append("---\n")

    # Section 1: Rappel Phase 1
    report.append("## 1. Rappel des constats PHASE 1\n")
    report.append("| Constat | Impact |")
    report.append("|---------|--------|")
    report.append("| Survivorship bias ages extremes (1900-1920) | FP sur vieux tuyaux, FN sur jeunes |")
    report.append("| Calibration mediocre (ECE=0.45) | Scores non utilisables en optimisation |")
    report.append("| Segments PEHD/PVC sous-couverts | 0% capture sur materiaux recents |")
    report.append("| Abandons preventifs polluent le signal | Bruit dans le label y=1 |")
    report.append("")

    # Section 2: Iteration 1
    report.append("## 2. Iteration 1 — Corrections biais (M1)\n")
    report.append("### A) Ages extremes\n")
    report.append("**Strategies testees:**")
    report.append("1. **Conservative (M1a)**: capping age a 110 ans + flag age extreme")
    report.append("2. **Structurelle (M1b)**: ratio age/decennie + residual age/materiau + winsorisation P99")
    report.append("")
    report.append("### B) Abandons preventifs\n")
    report.append("**Definition operationnelle:**")
    report.append("- event=1 ET zero anomalie historique ET ratio_age_median < 0.6 ET age < 15 ans ET pas de fuite recente")
    report.append(f"- Detectes: {n_prev} / {df['event'].sum()} evenements ({100*n_prev/max(df['event'].sum(),1):.1f}%)")
    report.append("")
    report.append("**Approches testees:**")
    report.append("- A) Filtrage label (event -> 0 pour abandons)")
    report.append("- B) Competing risks (multi-class: 0=survie, 1=defaillance, 2=abandon)")
    report.append("")

    report.append("### Resultats Iteration 1\n")
    report.append("| Modele | Lift@10% | Cap@10% | ROC-AUC | FN | Decision |")
    report.append("|--------|----------|---------|---------|----| ---------|")
    for name, r in m1_candidates.items():
        dec = "RETENU" if name == m1_best_name else "rejete"
        report.append(f"| {name} | {r['lift_10']:.2f} | {r['cap_10']:.2%} | "
                      f"{r['roc_auc']:.4f} | {r['conf']['FN']} | {dec} |")
    report.append("")

    # Section 3: Iteration 2
    report.append("## 3. Iteration 2 — Calibration & Stabilite (M2)\n")
    report.append("### Contraintes monotones\n")
    report.append("- age, ratio_age_median, overdue_years: monotone croissant (+1)")
    report.append("- days_since_last_fuite: monotone decroissant (-1)")
    report.append("- Regularisation: lambda=2.0, alpha=0.3, min_child=50")
    report.append("")
    report.append("### Calibration\n")
    report.append("| Variante | Lift@10% | Brier | ECE |")
    report.append("|----------|----------|-------|-----|")
    for cname, cm in cal_metrics.items():
        report.append(f"| {cname} | {cm['lift_10']:.2f} | {cm['brier']:.4f} | {cm['ece']:.4f} |")
    report.append("")
    report.append(f"**Calibration retenue**: {best_cal_name}")
    report.append("")
    report.append("### Definition risque_attendu\n")
    report.append("```")
    report.append("risque_attendu = proba_calibree × longueur_km")
    report.append("```")
    report.append("Ou proba_calibree est la sortie du modele apres calibration isotonique/Platt.")
    report.append("")

    # Section 4: Comparaison finale
    report.append("## 4. Iteration 3 — Comparaison finale\n")
    report.append(dataframe_to_markdown(df_summary))
    report.append("")

    # Section 5: Stabilite segments
    report.append("## 5. Stabilite par segments\n")
    report.append("### Par materiau (M*)\n")
    if not best_model_res["seg_materiau"].empty:
        report.append(
            dataframe_to_markdown(
                best_model_res["seg_materiau"][
                    ["segment", "n", "events", "capture", "fn_rate", "avg_score"]
                ]
            )
        )
    report.append("")
    report.append("### Par decennie (M*)\n")
    if not best_model_res["seg_decade"].empty:
        report.append(
            dataframe_to_markdown(
                best_model_res["seg_decade"][
                    ["segment", "n", "events", "capture", "fn_rate", "avg_score"]
                ]
            )
        )
    report.append("")

    # Section 6: Recommandation
    report.append("## 6. Recommandation finale\n")
    report.append(f"### Modele retenu: **{best_model_res['label']}**\n")
    report.append(f"- **Lift@10%**: {best_model_res['lift_10']:.2f}")
    report.append(f"- **Capture@10%**: {best_model_res['cap_10']:.2%}")
    report.append(f"- **Brier Score**: {best_model_res['brier']:.4f}")
    report.append(f"- **ECE**: {best_model_res['ece']:.4f}")
    report.append(f"- **FN@10%**: {best_model_res['conf']['FN']}")
    report.append(f"- **FP@10%**: {best_model_res['conf']['FP']}")
    report.append("")
    report.append("### Hypotheses metier restantes\n")
    report.append("1. L'heuristique d'abandon preventif est une approximation — "
                  "idealement confronter avec les motifs DHS du SIG.")
    report.append("2. Les segments PEHD/PVC/FTVI restent sous-represents — "
                  "enrichir les donnees si possible.")
    report.append("3. Le modele ne capture pas les defaillances dues a des "
                  "facteurs externes (travaux tiers, mouvements terrain).")
    report.append("")
    report.append("### Pret pour moteur d'optimisation?\n")
    report.append("**OUI** sous reserve de:")
    report.append("- Utiliser `risque_attendu = proba_calibree * longueur_km` comme valeur")
    report.append("- Monitorer les segments faibles trimestriellement")
    report.append("- Recalibrer annuellement avec nouvelles donnees")
    report.append("")

    report.append("## 7. Artefacts generes\n")
    report.append("| Fichier | Description |")
    report.append("|---------|-------------|")
    report.append(f"| `model_M0{suffix}.joblib` | Baseline LightGBM |")
    report.append(f"| `model_M1_best{suffix}.joblib` | Meilleur M1 (corrections biais) |")
    report.append(f"| `model_M2_best{suffix}.joblib` | M2 avec calibration |")
    report.append(f"| `model_M_star{suffix}.joblib` | Modele final retenu |")
    report.append(f"| `metadata_phase2{suffix}.json` | Metadata complete |")
    report.append("| `comparison_M0_M1_M2.csv` | Tableau comparatif |")
    report.append("| `plots/` | Graphiques AVANT/APRES |")
    report.append("")

    report_text = "\n".join(report)
    (phase2_dir / "rapport_phase2.md").write_text(report_text, encoding="utf-8")
    print(f"  Rapport sauvegarde: {phase2_dir / 'rapport_phase2.md'}")

    # Summary JSON
    summary_json = dict(
        M0=dict(lift_10=res_m0["lift_10"], cap_10=res_m0["cap_10"],
                roc=res_m0["roc_auc"], brier=res_m0["brier"], ece=res_m0["ece"],
                fn=res_m0["conf"]["FN"]),
        M1=dict(variant=m1_best_name,
                lift_10=res_m1["lift_10"], cap_10=res_m1["cap_10"],
                roc=res_m1["roc_auc"], brier=res_m1["brier"], ece=res_m1["ece"],
                fn=res_m1["conf"]["FN"]),
        M2=dict(variant=best_cal_name,
                lift_10=res_m2["lift_10"], cap_10=res_m2["cap_10"],
                roc=res_m2["roc_auc"], brier=res_m2["brier"], ece=res_m2["ece"],
                fn=res_m2["conf"]["FN"]),
        M_star=best_model_res["label"],
    )
    with open(phase2_dir / "summary_phase2.json", "w") as f:
        json.dump(summary_json, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("PHASE 2 TERMINEE")
    print(f"Modele final: {best_model_res['label']}")
    print(f"Lift@10%={best_model_res['lift_10']:.2f} "
          f"Cap@10%={best_model_res['cap_10']:.2%} "
          f"Brier={best_model_res['brier']:.4f}")
    print("=" * 70)

    return summary_json


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2 model improvement")
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="Horizon en annees (ex: 1, 3, 5).",
    )
    args = parser.parse_args()
    main(args.horizon)
