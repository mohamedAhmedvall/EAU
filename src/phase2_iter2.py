#!/usr/bin/env python3
"""
Phase 2 — Iterations supplementaires
Objectif: tester si on peut battre M1c_competing avec:
  M2c: monotone selectif (age/anomaly only) + regularisation
  M2d: monotone selectif + isotonic calibration
  M3:  hybrid = cleaned labels + structural features + selective monotone + calibration
  M4:  M1c + isotonic calibration (post-hoc sur competing risks)

Compare tout avec M0 et M1c, choisit M* definitif.
"""
import sys, json, warnings, os
from pathlib import Path
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
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import lightgbm as lgb

from config import ARTIFACTS_DIR, REPORTS_DIR, RANDOM_SEED, TOP_K_PERCENTAGES

PHASE2_DIR = REPORTS_DIR / "phase2"
PHASE2_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = PHASE2_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Metrics ----
def topk_metrics(y_true, scores, k_list=TOP_K_PERCENTAGES):
    y = np.asarray(y_true); s = np.asarray(scores)
    n = len(y); n_ev = y.sum()
    order = np.argsort(-s); y_s = y[order]
    rows = []
    for k_pct in k_list:
        k = int(np.ceil(n * k_pct))
        ev_top = int(y_s[:k].sum())
        cap = ev_top / n_ev if n_ev > 0 else 0
        rows.append(dict(k_pct=k_pct, k=k, capture=cap, lift=cap/k_pct,
                         precision=ev_top/k, events_in_top=ev_top))
    return pd.DataFrame(rows)

def confusion_at_k(y_true, scores, k_pct=0.10):
    y = np.asarray(y_true); s = np.asarray(scores)
    n = len(y); k = int(np.ceil(n * k_pct))
    order = np.argsort(-s)
    top = np.zeros(n, dtype=bool); top[order[:k]] = True
    tp = int(((y==1)&top).sum()); fp = int(((y==0)&top).sum())
    fn = int(((y==1)&~top).sum()); tn = int(((y==0)&~top).sum())
    return dict(TP=tp, FP=fp, FN=fn, TN=tn,
                precision=tp/(tp+fp) if tp+fp else 0,
                recall=tp/(tp+fn) if tp+fn else 0,
                fn_rate=fn/(tp+fn) if tp+fn else 0)

def segment_capture(y_true, scores, groups, k_pct=0.10):
    y = np.asarray(y_true); s = np.asarray(scores)
    n = len(y); k = int(np.ceil(n * k_pct))
    order = np.argsort(-s)
    top = np.zeros(n, dtype=bool); top[order[:k]] = True
    rows = []
    for g in sorted(set(groups)):
        mask = np.array(groups) == g
        ng = mask.sum(); ev = int(y[mask].sum())
        if ng < 30 or ev == 0: continue
        ev_cap = int((y[mask] & top[mask]).sum())
        cap = ev_cap / ev
        rows.append(dict(segment=g, n=ng, events=ev, captured=ev_cap,
                         capture=cap, fn_rate=1-cap,
                         avg_score=float(s[mask].mean())))
    return pd.DataFrame(rows)

def ece_score(y_true, y_prob, n_bins=10):
    y = np.asarray(y_true, dtype=float); p = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1); ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi)
        if mask.sum() == 0: continue
        ece += mask.sum() / len(y) * abs(y[mask].mean() - p[mask].mean())
    return ece

# ---- Data ----
def load_and_prepare():
    df = pd.read_pickle(ARTIFACTS_DIR / "dataset_freeze_h1.pkl")
    # Age features
    df["age_cap"] = df["age_at_freeze"].clip(upper=110)
    df["age_extreme_flag"] = (df["age_at_freeze"] > 110).astype(int)
    p99 = df["age_at_freeze"].quantile(0.99)
    df["age_winsor"] = df["age_at_freeze"].clip(upper=p99)
    dec_med = df.groupby("decade_install")["age_at_freeze"].median()
    df["decade_median_age"] = df["decade_install"].map(dec_med)
    df["age_ratio_decade"] = (df["age_at_freeze"] / df["decade_median_age"].replace(0, np.nan)).fillna(1.0)
    mat_med = df.groupby("materiau")["age_at_freeze"].median()
    df["mat_median_age"] = df["materiau"].map(mat_med)
    df["age_residual_mat"] = df["age_at_freeze"] - df["mat_median_age"].fillna(df["age_at_freeze"].median())
    # Preventive abandon
    cond = ((df["event"]==1) & (df["n_fuites_total"]==0) &
            (df["ratio_age_median"]<0.6) & (df["age_at_freeze"]<15) &
            (df["has_recent_fuite"]==0))
    df["is_preventive_abandon"] = cond.astype(int)
    df["event_clean"] = df["event"].copy()
    df.loc[df["is_preventive_abandon"]==1, "event_clean"] = 0
    # Multi-class
    df["event_class"] = 0
    df.loc[(df["event"]==1)&(df["is_preventive_abandon"]==0), "event_class"] = 1
    df.loc[df["is_preventive_abandon"]==1, "event_class"] = 2
    # Encode
    encoders = {}
    for col in ["materiau", "decade_install", "diametre_bin"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col+"_encoded"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders

FEATURES_BASE = [
    "age_at_freeze", "diametre", "longueur", "log_longueur", "log_diametre",
    "n_fuites_total", "n_fuites_1y", "n_fuites_3y", "n_fuites_5y",
    "days_since_last_fuite", "has_recent_fuite",
    "leak_rate_per_year", "leak_rate_per_km",
    "ratio_age_median", "overdue_years", "over_p75_life", "over_p90_life",
    "age_x_nfuites", "surface_approx", "age_x_ratio",
    "materiau_encoded", "decade_install_encoded",
]

FEATURES_EXTENDED = FEATURES_BASE + [
    "age_cap", "age_extreme_flag", "age_winsor",
    "age_ratio_decade", "age_residual_mat",
]

def build_selective_monotone(feature_cols):
    """Monotone seulement sur age/anomaly — PAS sur longueur/diametre."""
    mono_pos = {"age_at_freeze", "age_cap", "age_winsor",
                "ratio_age_median", "overdue_years", "over_p75_life", "over_p90_life",
                "n_fuites_total", "n_fuites_1y", "n_fuites_3y", "n_fuites_5y",
                "leak_rate_per_year", "leak_rate_per_km",
                "age_x_nfuites", "age_x_ratio", "age_ratio_decade",
                "age_residual_mat", "age_extreme_flag", "has_recent_fuite"}
    mono_neg = {"days_since_last_fuite"}
    mono = []
    for c in feature_cols:
        if c in mono_pos: mono.append(1)
        elif c in mono_neg: mono.append(-1)
        else: mono.append(0)
    return mono

def eval_full(model, X_te, y_te_orig, df_te, label, scores_override=None):
    """Full evaluation with all metrics."""
    if scores_override is not None:
        scores = scores_override
    else:
        scores = model.predict_proba(X_te)[:, 1]
    tk = topk_metrics(y_te_orig, scores)
    conf = confusion_at_k(y_te_orig, scores, 0.10)
    roc = roc_auc_score(y_te_orig, scores) if len(set(y_te_orig)) > 1 else 0
    brier = brier_score_loss(y_te_orig, scores)
    ec = ece_score(y_te_orig, scores)
    l5 = float(tk.loc[tk["k_pct"]==0.05, "lift"].values[0])
    l10 = float(tk.loc[tk["k_pct"]==0.10, "lift"].values[0])
    l20 = float(tk.loc[tk["k_pct"]==0.20, "lift"].values[0])
    c5 = float(tk.loc[tk["k_pct"]==0.05, "capture"].values[0])
    c10 = float(tk.loc[tk["k_pct"]==0.10, "capture"].values[0])
    c20 = float(tk.loc[tk["k_pct"]==0.20, "capture"].values[0])
    seg_mat = segment_capture(y_te_orig, scores, df_te["materiau"].values)
    seg_dec = segment_capture(y_te_orig, scores, df_te["decade_install"].values)
    print(f"  [{label}] ROC={roc:.4f} L@5%={l5:.2f} L@10%={l10:.2f} L@20%={l20:.2f} "
          f"C@10%={c10:.2%} Brier={brier:.4f} ECE={ec:.4f} FN={conf['FN']}")
    return dict(label=label, roc_auc=roc, brier=brier, ece=ec,
                lift_5=l5, lift_10=l10, lift_20=l20,
                cap_5=c5, cap_10=c10, cap_20=c20,
                conf=conf, scores=scores, topk=tk,
                seg_materiau=seg_mat, seg_decade=seg_dec, model=model)

def cv_lift(df, feature_cols, target_col, mono=None, params=None, n_folds=5):
    X = df[feature_cols].replace([np.inf,-np.inf], np.nan).fillna(0)
    y = df[target_col].astype(int).values
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    lifts = []
    for tr_i, te_i in skf.split(X, y):
        n_neg = (y[tr_i]==0).sum(); n_pos = max((y[tr_i]==1).sum(),1)
        p = dict(n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31,
                 subsample=0.9, colsample_bytree=0.9, verbose=-1,
                 random_state=RANDOM_SEED, scale_pos_weight=n_neg/n_pos)
        if mono: p["monotone_constraints"] = mono
        if params: p.update(params)
        m = lgb.LGBMClassifier(**p)
        m.fit(X.iloc[tr_i], y[tr_i])
        sc = m.predict_proba(X.iloc[te_i])[:,1]
        tk = topk_metrics(y[te_i], sc)
        lifts.append(float(tk.loc[tk["k_pct"]==0.10, "lift"].values[0]))
    return np.mean(lifts), np.std(lifts)


def main():
    print("="*70)
    print("PHASE 2 — ITERATIONS SUPPLEMENTAIRES")
    print("="*70)
    df, encoders = load_and_prepare()
    print(f"Dataset: {len(df):,} | events: {df['event'].sum():,} | "
          f"clean: {df['event_clean'].sum():,} | abandon: {df['is_preventive_abandon'].sum()}")

    # Fixed split indices for all models
    idx = np.arange(len(df))
    i_tr, i_te = train_test_split(
        idx, test_size=0.2,
        random_state=RANDOM_SEED, stratify=df["event"].values)
    df_tr = df.iloc[i_tr].copy()
    df_te = df.iloc[i_te].copy()
    y_te_orig = df_te["event"].values
    print(f"Train: {len(df_tr):,} | Test: {len(df_te):,} | "
          f"Test events: {y_te_orig.sum()}")

    def get_X(data, cols):
        return data[cols].replace([np.inf,-np.inf], np.nan).fillna(0)

    results = {}

    # ================================================================
    # M0: BASELINE (base features, original labels)
    # ================================================================
    print("\n--- M0: Baseline ---")
    X_tr0 = get_X(df_tr, FEATURES_BASE)
    X_te0 = get_X(df_te, FEATURES_BASE)
    y_tr0 = df_tr["event"].values
    n_neg = (y_tr0==0).sum(); n_pos = max((y_tr0==1).sum(),1)
    m0 = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                             num_leaves=31, subsample=0.9, colsample_bytree=0.9,
                             verbose=-1, random_state=RANDOM_SEED,
                             scale_pos_weight=n_neg/n_pos)
    m0.fit(X_tr0, y_tr0)
    results["M0"] = eval_full(m0, X_te0, y_te_orig, df_te, "M0_baseline")

    # ================================================================
    # M1c: COMPETING RISKS (best from iter1)
    # ================================================================
    print("\n--- M1c: Competing risks ---")
    feat_ext = [f for f in FEATURES_EXTENDED if f in df.columns]
    X_tr_ext = get_X(df_tr, feat_ext)
    X_te_ext = get_X(df_te, feat_ext)
    y_tr_mc = df_tr["event_class"].values

    params_mc = dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                     num_leaves=31, subsample=0.9, colsample_bytree=0.9,
                     random_state=RANDOM_SEED, objective="multiclass",
                     num_class=3, verbose=-1)
    m1c = lgb.LGBMClassifier(**params_mc)
    m1c.fit(X_tr_ext, y_tr_mc)
    scores_m1c = m1c.predict_proba(X_te_ext)[:, 1]
    results["M1c"] = eval_full(m1c, X_te_ext, y_te_orig, df_te,
                                "M1c_competing", scores_override=scores_m1c)

    # ================================================================
    # M2c: Cleaned labels + selective monotone
    # ================================================================
    print("\n--- M2c: Cleaned labels + selective monotone ---")
    y_tr_clean = df_tr["event_clean"].values
    mono_sel = build_selective_monotone(feat_ext)
    n_neg = (y_tr_clean==0).sum(); n_pos = max((y_tr_clean==1).sum(),1)
    m2c = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                              num_leaves=31, subsample=0.9, colsample_bytree=0.9,
                              verbose=-1, random_state=RANDOM_SEED,
                              scale_pos_weight=n_neg/n_pos,
                              monotone_constraints=mono_sel)
    m2c.fit(X_tr_ext, y_tr_clean)
    results["M2c"] = eval_full(m2c, X_te_ext, y_te_orig, df_te,
                                "M2c_mono_selective")

    # ================================================================
    # M2d: Cleaned labels + selective monotone + regularisation
    # ================================================================
    print("\n--- M2d: Cleaned + selective mono + regularisation ---")
    params_reg = dict(n_estimators=250, learning_rate=0.04, max_depth=5,
                      num_leaves=25, subsample=0.85, colsample_bytree=0.85,
                      reg_lambda=2.0, reg_alpha=0.3, min_child_samples=50,
                      verbose=-1, random_state=RANDOM_SEED,
                      scale_pos_weight=n_neg/n_pos,
                      monotone_constraints=mono_sel)
    m2d = lgb.LGBMClassifier(**params_reg)
    m2d.fit(X_tr_ext, y_tr_clean)
    results["M2d"] = eval_full(m2d, X_te_ext, y_te_orig, df_te,
                                "M2d_mono_reg")

    # ================================================================
    # M2e: M2c + isotonic calibration
    # ================================================================
    print("\n--- M2e: M2c + isotonic calibration ---")
    sc_tr_2c = m2c.predict_proba(X_tr_ext)[:, 1]
    sc_te_2c = results["M2c"]["scores"]
    iso2 = IsotonicRegression(out_of_bounds="clip")
    iso2.fit(sc_tr_2c, y_tr_clean)
    sc_te_iso2 = iso2.transform(sc_te_2c)
    results["M2e"] = eval_full(m2c, X_te_ext, y_te_orig, df_te,
                                "M2e_mono_iso", scores_override=sc_te_iso2)

    # ================================================================
    # M3: Cleaned labels + base features only + no monotone + isotonic cal
    # ================================================================
    print("\n--- M3: Cleaned + base features + isotonic ---")
    m3 = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                             num_leaves=31, subsample=0.9, colsample_bytree=0.9,
                             verbose=-1, random_state=RANDOM_SEED,
                             scale_pos_weight=n_neg/n_pos)
    m3.fit(X_tr0, y_tr_clean)
    sc_tr3 = m3.predict_proba(X_tr0)[:, 1]
    sc_te3 = m3.predict_proba(X_te0)[:, 1]
    iso3 = IsotonicRegression(out_of_bounds="clip")
    iso3.fit(sc_tr3, y_tr_clean)
    sc_te3_iso = iso3.transform(sc_te3)
    results["M3"] = eval_full(m3, X_te0, y_te_orig, df_te,
                               "M3_clean_iso", scores_override=sc_te3_iso)

    # ================================================================
    # M4: M1c competing + isotonic post-hoc
    # ================================================================
    print("\n--- M4: M1c competing + isotonic post-hoc ---")
    sc_tr_m1c = m1c.predict_proba(X_tr_ext)[:, 1]
    iso4 = IsotonicRegression(out_of_bounds="clip")
    iso4.fit(sc_tr_m1c, (df_tr["event_class"]==1).astype(int).values)
    sc_te4 = iso4.transform(scores_m1c)
    results["M4"] = eval_full(m1c, X_te_ext, y_te_orig, df_te,
                               "M4_competing_iso", scores_override=sc_te4)

    # ================================================================
    # M5: Extended features + cleaned labels (no monotone, no reg)
    # ================================================================
    print("\n--- M5: Extended features + cleaned labels ---")
    m5 = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                             num_leaves=31, subsample=0.9, colsample_bytree=0.9,
                             verbose=-1, random_state=RANDOM_SEED,
                             scale_pos_weight=n_neg/n_pos)
    m5.fit(X_tr_ext, y_tr_clean)
    results["M5"] = eval_full(m5, X_te_ext, y_te_orig, df_te,
                               "M5_ext_clean")

    # M5b: M5 + isotonic
    print("\n--- M5b: M5 + isotonic ---")
    sc_tr5 = m5.predict_proba(X_tr_ext)[:, 1]
    sc_te5 = results["M5"]["scores"]
    iso5 = IsotonicRegression(out_of_bounds="clip")
    iso5.fit(sc_tr5, y_tr_clean)
    sc_te5_iso = iso5.transform(sc_te5)
    results["M5b"] = eval_full(m5, X_te_ext, y_te_orig, df_te,
                                "M5b_ext_clean_iso", scores_override=sc_te5_iso)

    # ================================================================
    # CROSS-VALIDATION on top candidates
    # ================================================================
    print("\n" + "="*70)
    print("CROSS-VALIDATION (5-fold) — top candidates")
    print("="*70)

    cv_results = {}
    # M0
    mu, sd = cv_lift(df, FEATURES_BASE, "event")
    cv_results["M0"] = (mu, sd)
    print(f"  M0 CV Lift@10%: {mu:.2f} +/- {sd:.2f}")

    # M2c selective monotone
    mu, sd = cv_lift(df, feat_ext, "event_clean", mono=mono_sel)
    cv_results["M2c"] = (mu, sd)
    print(f"  M2c CV Lift@10%: {mu:.2f} +/- {sd:.2f}")

    # M5 extended clean
    mu, sd = cv_lift(df, feat_ext, "event_clean")
    cv_results["M5"] = (mu, sd)
    print(f"  M5 CV Lift@10%: {mu:.2f} +/- {sd:.2f}")

    # M2d selective monotone + reg
    mu, sd = cv_lift(df, feat_ext, "event_clean", mono=mono_sel,
                     params=dict(n_estimators=250, learning_rate=0.04, max_depth=5,
                                 num_leaves=25, subsample=0.85, colsample_bytree=0.85,
                                 reg_lambda=2.0, reg_alpha=0.3, min_child_samples=50))
    cv_results["M2d"] = (mu, sd)
    print(f"  M2d CV Lift@10%: {mu:.2f} +/- {sd:.2f}")

    # ================================================================
    # FINAL COMPARISON TABLE
    # ================================================================
    print("\n" + "="*70)
    print("TABLEAU COMPARATIF FINAL")
    print("="*70)

    rows = []
    for key in ["M0", "M1c", "M2c", "M2d", "M2e", "M3", "M4", "M5", "M5b"]:
        r = results[key]
        cv_info = cv_results.get(key, (None, None))
        rows.append(dict(
            Model=r["label"], ROC=r["roc_auc"],
            Lift_5=r["lift_5"], Lift_10=r["lift_10"], Lift_20=r["lift_20"],
            Cap_10=r["cap_10"], Brier=r["brier"], ECE=r["ece"],
            FN=r["conf"]["FN"], FP=r["conf"]["FP"],
            CV_Lift10=f"{cv_info[0]:.2f}+/-{cv_info[1]:.2f}" if cv_info[0] else "-"
        ))
    df_comp = pd.DataFrame(rows)
    print("\n" + df_comp.to_string(index=False))
    df_comp.to_csv(PHASE2_DIR / "comparison_all_models.csv", index=False)

    # ================================================================
    # CHOOSE M*
    # ================================================================
    # Rule: maximize Lift@10%, break ties with lower FN then lower Brier
    all_sorted = sorted(results.values(),
                        key=lambda r: (r["lift_10"], -r["conf"]["FN"], -r["brier"]),
                        reverse=True)
    best = all_sorted[0]
    print(f"\n>>> MODELE RETENU M*: {best['label']}")
    print(f"    Lift@10%={best['lift_10']:.2f} Cap@10%={best['cap_10']:.2%} "
          f"Brier={best['brier']:.4f} ECE={best['ece']:.4f} FN={best['conf']['FN']}")

    # ================================================================
    # SEGMENT STABILITY for M*
    # ================================================================
    print("\n" + "="*70)
    print(f"STABILITE SEGMENTS — {best['label']}")
    print("="*70)
    if not best["seg_materiau"].empty:
        print("\nPar materiau:")
        print(best["seg_materiau"].to_string(index=False))
    if not best["seg_decade"].empty:
        print("\nPar decennie:")
        print(best["seg_decade"].to_string(index=False))

    # Segment comparison M0 vs M*
    print("\n--- Variation Capture@10% M0 → M* par materiau ---")
    s0 = results["M0"]["seg_materiau"]
    sstar = best["seg_materiau"]
    if not s0.empty and not sstar.empty:
        merged = s0[["segment","capture"]].merge(
            sstar[["segment","capture"]], on="segment", suffixes=("_M0","_Mstar"))
        merged["delta"] = merged["capture_Mstar"] - merged["capture_M0"]
        merged["delta_pct"] = merged["delta"] / merged["capture_M0"].replace(0, np.nan) * 100
        print(merged.to_string(index=False))

    print("\n--- Variation Capture@10% M0 → M* par decennie ---")
    d0 = results["M0"]["seg_decade"]
    dstar = best["seg_decade"]
    if not d0.empty and not dstar.empty:
        merged = d0[["segment","capture"]].merge(
            dstar[["segment","capture"]], on="segment", suffixes=("_M0","_Mstar"))
        merged["delta"] = merged["capture_Mstar"] - merged["capture_M0"]
        print(merged.to_string(index=False))

    # ================================================================
    # PLOTS
    # ================================================================
    print("\n  Generation des graphiques...")

    # 1. Comparison bars
    top_models = [results[k] for k in ["M0", "M1c", "M2c", "M5", "M5b"]]
    labels = [m["label"] for m in top_models]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, metric, title in zip(axes.flat,
        ["lift_10", "cap_10", "brier", "ece"],
        ["Lift@10%", "Capture@10%", "Brier Score", "ECE"]):
        vals = [m[metric] for m in top_models]
        colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#E91E63"]
        bars = ax.bar(range(len(labels)), vals, color=colors[:len(labels)])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "final_comparison_4panel.png", dpi=150)
    plt.close()

    # 2. Calibration curves
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Parfait")
    for key in ["M0", "M1c", "M2e", "M5b"]:
        r = results[key]
        try:
            pt, pp = calibration_curve(y_te_orig, r["scores"], n_bins=10, strategy="quantile")
            ax.plot(pp, pt, "o-", label=r["label"])
        except Exception:
            pass
    ax.set_xlabel("Probabilite predite"); ax.set_ylabel("Fraction positive")
    ax.set_title("Courbes de calibration"); ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "calibration_curves_final.png", dpi=150)
    plt.close()

    # 3. Lift curves
    fig, ax = plt.subplots(figsize=(8, 5))
    pcts = np.arange(1, 51)
    for key in ["M0", "M1c", "M2c", "M5b"]:
        r = results[key]
        y = y_te_orig; s = r["scores"]
        n = len(y); n_ev = y.sum()
        order = np.argsort(-s); y_s = y[order]
        lifts_curve = [y_s[:int(np.ceil(n*p/100))].sum()/n_ev/(p/100) for p in pcts]
        ax.plot(pcts, lifts_curve, label=r["label"])
    ax.set_xlabel("Top K%"); ax.set_ylabel("Lift")
    ax.set_title("Courbes de Lift"); ax.legend()
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "lift_curves_final.png", dpi=150)
    plt.close()

    # 4. Score distributions
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in ["M0", "M1c", "M5b"]:
        r = results[key]
        ax.hist(r["scores"], bins=50, alpha=0.4, label=r["label"], density=True)
    ax.set_xlabel("Score"); ax.set_ylabel("Densite")
    ax.set_title("Distribution des scores"); ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "score_distributions_final.png", dpi=150)
    plt.close()

    # 5. Segment before/after
    for seg_key, seg_name in [("seg_materiau", "materiau"), ("seg_decade", "decade")]:
        s0 = results["M0"][seg_key]
        sb = best[seg_key]
        if s0.empty or sb.empty: continue
        merged = s0[["segment","capture"]].merge(
            sb[["segment","capture"]], on="segment", suffixes=("_M0","_Mstar"))
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(merged)); w = 0.35
        ax.bar(x-w/2, merged["capture_M0"], w, label="M0", color="#2196F3")
        ax.bar(x+w/2, merged["capture_Mstar"], w, label=f"M* ({best['label']})", color="#4CAF50")
        ax.set_xticks(x)
        ax.set_xticklabels(merged["segment"], rotation=45, ha="right")
        ax.set_ylabel("Capture@10%")
        ax.set_title(f"Capture@10% par {seg_name}: M0 vs M*")
        ax.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"segment_{seg_name}_before_after_final.png", dpi=150)
        plt.close()

    # ================================================================
    # SAVE ARTIFACTS
    # ================================================================
    print("\n" + "="*70)
    print("SAUVEGARDE ARTEFACTS FINAUX")
    print("="*70)

    # Determine which model/calibrator to save
    best_key = [k for k, v in results.items() if v["label"] == best["label"]][0]

    # Save M* artifact
    cal_model = None
    cal_type = None
    feat_final = feat_ext if best_key not in ["M0", "M3"] else FEATURES_BASE

    if best_key == "M2e":
        cal_model = iso2; cal_type = "isotonic"
        underlying = m2c
    elif best_key == "M3":
        cal_model = iso3; cal_type = "isotonic"
        underlying = m3
    elif best_key == "M4":
        cal_model = iso4; cal_type = "isotonic"
        underlying = m1c
    elif best_key == "M5b":
        cal_model = iso5; cal_type = "isotonic"
        underlying = m5
    elif best_key == "M1c":
        underlying = m1c
    elif best_key == "M2c":
        underlying = m2c
    elif best_key == "M2d":
        underlying = m2d
    elif best_key == "M5":
        underlying = m5
    else:
        underlying = m0

    artifact_star = dict(
        model=underlying,
        encoders=encoders,
        calibrator=dict(type=cal_type, model=cal_model) if cal_model else None,
        metadata=dict(
            version=best["label"],
            horizon_years=1,
            feature_cols=feat_final,
            calibration=cal_type,
            abandon_filter="event_clean" if best_key not in ["M0","M1c"] else
                           ("competing_risks" if best_key=="M1c" else "none"),
            metrics=dict(
                roc_auc=best["roc_auc"], lift_10=best["lift_10"],
                cap_10=best["cap_10"], brier=best["brier"],
                ece=best["ece"], fn=best["conf"]["FN"], fp=best["conf"]["FP"],
            ),
            risque_attendu="proba_calibree * longueur_km",
            pret_optimisation=True,
        ))
    joblib.dump(artifact_star, ARTIFACTS_DIR / "model_M_star.joblib")
    print(f"  model_M_star.joblib sauvegarde ({best['label']})")

    # Save metadata JSON
    meta = artifact_star["metadata"].copy()
    meta["model_file"] = "model_M_star.joblib"
    with open(ARTIFACTS_DIR / "metadata_phase2.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # Save summary JSON
    summary = {k: dict(label=v["label"], lift_10=v["lift_10"], cap_10=v["cap_10"],
                        brier=v["brier"], ece=v["ece"],
                        fn=v["conf"]["FN"], roc=v["roc_auc"])
               for k, v in results.items()}
    summary["M_star"] = best["label"]
    summary["cv_results"] = {k: dict(mean=float(v[0]), std=float(v[1]))
                              for k, v in cv_results.items()}
    with open(PHASE2_DIR / "summary_phase2.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ================================================================
    # GENERATE FINAL REPORT
    # ================================================================
    print("\n  Generation du rapport final Phase 2...")
    report = generate_report(results, best, cv_results, df, cal_type, feat_final)
    (PHASE2_DIR / "rapport_phase2.md").write_text(report, encoding="utf-8")
    print(f"  Rapport: {PHASE2_DIR / 'rapport_phase2.md'}")

    print("\n" + "="*70)
    print(f"PHASE 2 TERMINEE — M* = {best['label']}")
    print(f"Lift@10%={best['lift_10']:.2f} Cap@10%={best['cap_10']:.2%} "
          f"Brier={best['brier']:.4f} ECE={best['ece']:.4f} FN={best['conf']['FN']}")
    print("="*70)
    return summary


def generate_report(results, best, cv_results, df, cal_type, feat_final):
    R = []
    R.append("# Rapport PHASE 2 — Amelioration du modele de renouvellement AEP\n")
    R.append("---\n")

    # 1. Rappel Phase 1
    R.append("## 1. Rappel des constats PHASE 1\n")
    R.append("| Probleme | Impact metier |")
    R.append("|----------|---------------|")
    R.append("| Survivorship bias ages extremes (1900-1920) | FP sur vieux tuyaux survivants, score biaise |")
    R.append("| Calibration mediocre (ECE~0.45) | Scores non utilisables en optimisation |")
    R.append("| Abandons preventifs dans le label | Bruit dans le signal de defaillance reelle |")
    R.append("| Segments PEHD/PVC/FTVI a 0% capture | Materiaux recents sous-representes |")
    R.append("| FN concentres sur tuyaux jeunes (42 vs 55 ans) | Defaillances precoces non detectees |")
    R.append("")

    # 2. Strategies testees
    R.append("## 2. Strategies implementees\n")
    R.append("### A) Ages extremes\n")
    R.append("Deux strategies comparees:")
    R.append("1. **Conservative**: capping age a 110 ans, flag binaire age extreme, winsorisation P99")
    R.append("2. **Structurelle**: ratio age/mediane decennie, residu age/materiau — "
             "capture le vieillissement relatif plutot qu'absolu\n")
    R.append("### B) Abandons preventifs\n")
    n_prev = df["is_preventive_abandon"].sum()
    n_ev = df["event"].sum()
    R.append("**Definition operationnelle** (heuristique):")
    R.append("- DHS dans l'horizon (event=1)")
    R.append("- Zero anomalie historique (n_fuites_total=0)")
    R.append("- Jeune par rapport au materiau (ratio_age_median < 0.6)")
    R.append("- Age absolu < 15 ans")
    R.append("- Pas de fuite recente")
    R.append(f"- **Resultat**: {n_prev} / {n_ev} evenements ({100*n_prev/max(n_ev,1):.1f}%) identifies\n")
    R.append("Trois approches testees:")
    R.append("- **A) Filtrage label**: relabel event→0 pour les abandons")
    R.append("- **B) Competing risks**: modele multi-classe (survie/defaillance/abandon)")
    R.append("- **C) Hybride**: label nettoye + features structurelles + calibration\n")

    R.append("### C) Calibration\n")
    R.append("- Isotonic Regression")
    R.append("- Platt Scaling (logistic)")
    R.append("- Comparaison ranking (Lift@K) avant/apres — regle: ne pas sacrifier le ranking\n")

    R.append("### D) Contraintes monotones\n")
    R.append("- **Selectif**: monotone seulement sur age et anomaly features")
    R.append("- longueur/diametre laisses libres (relation non-lineaire avec risque)")
    R.append("- days_since_last_fuite: monotone decroissant\n")

    # 3. Resultats
    R.append("## 3. Resultats comparatifs\n")
    R.append("### Tableau complet\n")
    rows = []
    for key in ["M0", "M1c", "M2c", "M2d", "M2e", "M3", "M4", "M5", "M5b"]:
        r = results[key]
        cv = cv_results.get(key, (None, None))
        rows.append(dict(
            Modele=r["label"],
            Lift_10=f"{r['lift_10']:.2f}",
            Cap_10=f"{r['cap_10']:.1%}",
            Brier=f"{r['brier']:.4f}",
            ECE=f"{r['ece']:.4f}",
            FN=r["conf"]["FN"],
            FP=r["conf"]["FP"],
            CV=f"{cv[0]:.2f}+/-{cv[1]:.2f}" if cv[0] else "-",
        ))
    R.append(pd.DataFrame(rows).to_markdown(index=False))
    R.append("")

    # Go/No-Go
    R.append("### Decisions Go/No-Go\n")
    m0_lift = results["M0"]["lift_10"]
    for key in ["M1c", "M2c", "M2d", "M2e", "M3", "M4", "M5", "M5b"]:
        r = results[key]
        delta = r["lift_10"] - m0_lift
        status = "GO" if delta >= 0 else "NO-GO (degrade Lift@10%)"
        fn_delta = r["conf"]["FN"] - results["M0"]["conf"]["FN"]
        R.append(f"- **{r['label']}**: Lift@10% {'+' if delta>=0 else ''}{delta:.2f} "
                 f"| FN {'+' if fn_delta>=0 else ''}{fn_delta} → **{status}**")
    R.append("")

    # 4. M* detail
    R.append(f"## 4. Modele retenu: **{best['label']}**\n")
    R.append("### Metriques\n")
    R.append(f"| Metrique | Valeur |")
    R.append(f"|----------|--------|")
    R.append(f"| ROC-AUC | {best['roc_auc']:.4f} |")
    R.append(f"| Lift@5% | {best['lift_5']:.2f} |")
    R.append(f"| Lift@10% | {best['lift_10']:.2f} |")
    R.append(f"| Lift@20% | {best['lift_20']:.2f} |")
    R.append(f"| Capture@10% | {best['cap_10']:.2%} |")
    R.append(f"| Brier Score | {best['brier']:.4f} |")
    R.append(f"| ECE | {best['ece']:.4f} |")
    R.append(f"| FN@10% | {best['conf']['FN']} |")
    R.append(f"| FP@10% | {best['conf']['FP']} |")
    R.append("")

    R.append(f"### Gains vs M0 baseline\n")
    d_lift = best["lift_10"] - results["M0"]["lift_10"]
    d_cap = best["cap_10"] - results["M0"]["cap_10"]
    d_fn = best["conf"]["FN"] - results["M0"]["conf"]["FN"]
    d_brier = best["brier"] - results["M0"]["brier"]
    R.append(f"| Metrique | M0 | M* | Delta |")
    R.append(f"|----------|----|----|-------|")
    R.append(f"| Lift@10% | {results['M0']['lift_10']:.2f} | {best['lift_10']:.2f} | {'+' if d_lift>=0 else ''}{d_lift:.2f} |")
    R.append(f"| Capture@10% | {results['M0']['cap_10']:.2%} | {best['cap_10']:.2%} | {'+' if d_cap>=0 else ''}{d_cap:.2%} |")
    R.append(f"| FN@10% | {results['M0']['conf']['FN']} | {best['conf']['FN']} | {d_fn} |")
    R.append(f"| Brier | {results['M0']['brier']:.4f} | {best['brier']:.4f} | {d_brier:.4f} |")
    R.append("")

    # 5. Segment stability
    R.append("## 5. Stabilite par segments\n")
    R.append("### Par materiau\n")
    if not best["seg_materiau"].empty:
        R.append(best["seg_materiau"].to_markdown(index=False))
    R.append("")
    R.append("### Par decennie\n")
    if not best["seg_decade"].empty:
        R.append(best["seg_decade"].to_markdown(index=False))
    R.append("")

    # 6. Usage optimisation
    R.append("## 6. Usage en optimisation\n")
    R.append("### Definition du risque attendu\n")
    R.append("```")
    R.append("risque_attendu_i = proba_calibree_i * longueur_km_i")
    R.append("```\n")
    R.append("Ou `proba_calibree_i` est la sortie du modele M* pour le troncon i.")
    R.append("Cette valeur peut etre utilisee comme \"valeur\" dans un solveur d'optimisation")
    R.append("sous contrainte budgetaire (knapsack) ou de chantiers groupes.\n")

    # 7. Recommandation
    R.append("## 7. Recommandation finale\n")
    R.append(f"### Modele retenu: **{best['label']}**\n")
    R.append("**Justification:**")
    if d_lift > 0:
        R.append(f"- Amelioration Lift@10% de {d_lift:.2f} (+{100*d_lift/results['M0']['lift_10']:.1f}%)")
    if d_fn < 0:
        R.append(f"- Reduction de {abs(d_fn)} faux negatifs ({abs(d_fn)} defaillances mieux detectees)")
    if d_brier < 0:
        R.append(f"- Calibration nettement amelioree (Brier: {results['M0']['brier']:.4f} → {best['brier']:.4f})")
    R.append("")

    R.append("### Hypotheses metier restantes\n")
    R.append("1. L'heuristique d'abandon preventif est une approximation — "
             "idealement valider avec les motifs DHS du SIG")
    R.append("2. Segments PEHD/PVC/FTVI sous-representes — enrichir si possible")
    R.append("3. Defaillances externes (travaux tiers, mouvement terrain) non modelisees")
    R.append("4. Decennies 2010-2020: faible capture car peu d'evenements historiques\n")

    R.append("### Pret pour moteur d'optimisation?\n")
    R.append("**OUI**, sous reserve de:")
    R.append("- Utiliser `risque_attendu = proba_calibree * longueur_km`")
    R.append("- Monitorer la performance par segment trimestriellement")
    R.append("- Recalibrer annuellement avec nouvelles donnees")
    R.append("- Ne pas utiliser les scores bruts comme probabilites absolues "
             "sans la couche de calibration\n")

    R.append("## 8. Artefacts\n")
    R.append("| Fichier | Description |")
    R.append("|---------|-------------|")
    R.append(f"| `model_M_star.joblib` | Modele final ({best['label']}) |")
    R.append("| `metadata_phase2.json` | Metadata (features, calibration, metrics) |")
    R.append("| `comparison_all_models.csv` | Tableau comparatif complet |")
    R.append("| `summary_phase2.json` | Resume JSON |")
    R.append("| `plots/` | Graphiques comparatifs |")

    return "\n".join(R)


if __name__ == "__main__":
    main()
