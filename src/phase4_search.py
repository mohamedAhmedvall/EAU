#!/usr/bin/env python3
"""
PHASE 4 — Recherche Exhaustive du Meilleur Modele
==================================================
Tests systematiques de multiples approches pour battre M4_competing_iso (Lift@10%=6.21)
"""

import sys, os, json, warnings, time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

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

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, QuantileTransformer
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, BaggingClassifier, StackingClassifier,
    VotingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

import lightgbm as lgb

try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False

try:
    import catboost as cb
    HAS_CB = True
except:
    HAS_CB = False

from config import ARTIFACTS_DIR, REPORTS_DIR, RANDOM_SEED, TOP_K_PERCENTAGES

# Output directory
PHASE4_DIR = REPORTS_DIR / "phase4_experiments"
PHASE4_DIR.mkdir(parents=True, exist_ok=True)

# Current best to beat
BEST_LIFT10 = 6.21
BEST_BRIER = 0.0104

# ============================================================================
# METRICS
# ============================================================================

def topk_metrics(y_true, scores, k_list=[0.05, 0.10, 0.20]):
    y = np.asarray(y_true); s = np.asarray(scores)
    n = len(y); n_ev = y.sum()
    order = np.argsort(-s); y_s = y[order]
    rows = []
    for k_pct in k_list:
        k = int(np.ceil(n * k_pct))
        ev_top = int(y_s[:k].sum())
        cap = ev_top / n_ev if n_ev > 0 else 0
        rows.append(dict(k_pct=k_pct, k=k, capture=cap, lift=cap/k_pct, events_in_top=ev_top))
    return pd.DataFrame(rows)

def confusion_at_k(y_true, scores, k_pct=0.10):
    y = np.asarray(y_true); s = np.asarray(scores)
    n = len(y); k = int(np.ceil(n * k_pct))
    order = np.argsort(-s)
    top = np.zeros(n, dtype=bool); top[order[:k]] = True
    tp = int(((y==1)&top).sum()); fp = int(((y==0)&top).sum())
    fn = int(((y==1)&~top).sum()); tn = int(((y==0)&~top).sum())
    return dict(TP=tp, FP=fp, FN=fn, TN=tn)

def ece_score(y_true, y_prob, n_bins=10):
    y = np.asarray(y_true, dtype=float); p = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1); ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi)
        if mask.sum() == 0: continue
        ece += mask.sum() / len(y) * abs(y[mask].mean() - p[mask].mean())
    return ece

def eval_model(scores, y_test, label):
    tk = topk_metrics(y_test, scores)
    conf = confusion_at_k(y_test, scores, 0.10)
    try:
        roc = roc_auc_score(y_test, scores)
    except:
        roc = 0.5
    brier = brier_score_loss(y_test, scores)
    ec = ece_score(y_test, scores)
    l5 = float(tk.loc[tk["k_pct"]==0.05, "lift"].values[0])
    l10 = float(tk.loc[tk["k_pct"]==0.10, "lift"].values[0])
    l20 = float(tk.loc[tk["k_pct"]==0.20, "lift"].values[0])
    c10 = float(tk.loc[tk["k_pct"]==0.10, "capture"].values[0])

    improved = "+++" if l10 > BEST_LIFT10 else ("+" if l10 > BEST_LIFT10 * 0.99 else "")
    print(f"  [{label}] L@10%={l10:.3f}{improved} C@10%={c10:.1%} Brier={brier:.4f} ROC={roc:.4f} FN={conf['FN']}")

    return dict(
        label=label, roc_auc=roc, brier=brier, ece=ec,
        lift_5=l5, lift_10=l10, lift_20=l20, cap_10=c10,
        fn=conf['FN'], fp=conf['FP'], scores=scores
    )

# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_prepare():
    df = pd.read_pickle(ARTIFACTS_DIR / "dataset_freeze_h1.pkl")

    # Encodings
    for col in ["materiau", "decade_install", "diametre_bin"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col+"_enc"] = le.fit_transform(df[col].astype(str))

    # Age corrections
    df["age_cap"] = df["age_at_freeze"].clip(upper=110)
    df["age_extreme"] = (df["age_at_freeze"] > 110).astype(int)
    p99 = df["age_at_freeze"].quantile(0.99)
    df["age_winsor"] = df["age_at_freeze"].clip(upper=p99)

    # Decade median
    dec_med = df.groupby("decade_install")["age_at_freeze"].median()
    df["decade_med_age"] = df["decade_install"].map(dec_med)
    df["age_ratio_dec"] = df["age_at_freeze"] / df["decade_med_age"].replace(0, np.nan)
    df["age_ratio_dec"] = df["age_ratio_dec"].fillna(1.0)

    # Material median
    mat_med = df.groupby("materiau")["age_at_freeze"].median()
    df["mat_med_age"] = df["materiau"].map(mat_med)
    df["age_resid_mat"] = df["age_at_freeze"] - df["mat_med_age"].fillna(df["age_at_freeze"].median())

    # Preventive abandons
    cond = ((df["event"]==1) & (df["n_fuites_total"]==0) &
            (df["ratio_age_median"]<0.6) & (df["age_at_freeze"]<15))
    df["is_prev_abandon"] = cond.astype(int)
    df["event_clean"] = df["event"].copy()
    df.loc[df["is_prev_abandon"]==1, "event_clean"] = 0
    df["event_class"] = 0
    df.loc[(df["event"]==1)&(df["is_prev_abandon"]==0), "event_class"] = 1
    df.loc[df["is_prev_abandon"]==1, "event_class"] = 2

    # Additional features
    df["age_sq"] = df["age_at_freeze"] ** 2
    df["age_log"] = np.log1p(df["age_at_freeze"])
    df["n_fuites_sq"] = df["n_fuites_total"] ** 2
    df["ratio_sq"] = df["ratio_age_median"] ** 2
    df["age_x_recent"] = df["age_at_freeze"] * df["has_recent_fuite"]
    df["age_x_leakrate"] = df["age_at_freeze"] * df["leak_rate_per_year"]
    df["overdue_x_fuites"] = df["overdue_years"] * df["n_fuites_total"]

    # Quantile features
    for col in ["age_at_freeze", "n_fuites_total", "ratio_age_median"]:
        df[f"{col}_q"] = pd.qcut(df[col], q=10, labels=False, duplicates="drop")

    return df

FEAT_BASE = [
    "age_at_freeze", "diametre", "longueur", "log_longueur", "log_diametre",
    "n_fuites_total", "n_fuites_1y", "n_fuites_3y", "n_fuites_5y",
    "days_since_last_fuite", "has_recent_fuite",
    "leak_rate_per_year", "leak_rate_per_km",
    "ratio_age_median", "overdue_years", "over_p75_life", "over_p90_life",
    "age_x_nfuites", "surface_approx", "age_x_ratio",
    "materiau_enc", "decade_install_enc",
]

FEAT_EXT = FEAT_BASE + [
    "age_cap", "age_extreme", "age_winsor",
    "age_ratio_dec", "age_resid_mat",
]

FEAT_FULL = FEAT_EXT + [
    "age_sq", "age_log", "n_fuites_sq", "ratio_sq",
    "age_x_recent", "age_x_leakrate", "overdue_x_fuites",
]

def get_X(df, cols):
    cols_exist = [c for c in cols if c in df.columns]
    return df[cols_exist].replace([np.inf, -np.inf], np.nan).fillna(0)

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

def get_lgbm_params(variant="base"):
    base = dict(n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31,
                subsample=0.9, colsample_bytree=0.9, verbose=-1, random_state=RANDOM_SEED)

    variants = {
        "base": base,
        "deep": {**base, "max_depth": 10, "num_leaves": 63, "n_estimators": 400},
        "shallow": {**base, "max_depth": 4, "num_leaves": 15, "n_estimators": 500},
        "reg": {**base, "reg_lambda": 2.0, "reg_alpha": 0.5, "min_child_samples": 50},
        "dart": {**base, "boosting_type": "dart", "n_estimators": 200},
        "goss": {**base, "boosting_type": "goss", "n_estimators": 400},
    }
    return variants.get(variant, base)

def get_xgb_params(variant="base"):
    base = dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.9, colsample_bytree=0.9, verbosity=0,
                random_state=RANDOM_SEED, use_label_encoder=False, eval_metric='logloss')

    variants = {
        "base": base,
        "deep": {**base, "max_depth": 10, "n_estimators": 400},
        "reg": {**base, "reg_lambda": 2.0, "reg_alpha": 0.5},
    }
    return variants.get(variant, base)

def get_cb_params(variant="base"):
    base = dict(iterations=300, learning_rate=0.05, depth=6,
                verbose=False, random_state=RANDOM_SEED)

    variants = {
        "base": base,
        "deep": {**base, "depth": 10, "iterations": 400},
        "ordered": {**base, "boosting_type": "Ordered"},
    }
    return variants.get(variant, base)

# ============================================================================
# EXPERIMENTS
# ============================================================================

def run_experiments(df, i_tr, i_te):
    results = []
    df_tr = df.iloc[i_tr].copy()
    df_te = df.iloc[i_te].copy()
    y_te = df_te["event"].values
    y_tr = df_tr["event"].values
    y_tr_clean = df_tr["event_clean"].values
    y_tr_mc = df_tr["event_class"].values

    n_neg = (y_tr==0).sum()
    n_pos = max((y_tr==1).sum(), 1)
    sw = n_neg / n_pos

    # Feature sets
    X_tr_base = get_X(df_tr, FEAT_BASE)
    X_te_base = get_X(df_te, FEAT_BASE)
    X_tr_ext = get_X(df_tr, FEAT_EXT)
    X_te_ext = get_X(df_te, FEAT_EXT)
    X_tr_full = get_X(df_tr, FEAT_FULL)
    X_te_full = get_X(df_te, FEAT_FULL)

    print("\n" + "="*70)
    print("EXPERIMENTS - LIGHTGBM VARIANTS")
    print("="*70)

    # LightGBM variants
    for variant in ["base", "deep", "shallow", "reg", "dart", "goss"]:
        for feat_name, X_tr, X_te in [("base", X_tr_base, X_te_base),
                                       ("ext", X_tr_ext, X_te_ext),
                                       ("full", X_tr_full, X_te_full)]:
            params = get_lgbm_params(variant)
            params["scale_pos_weight"] = sw
            m = lgb.LGBMClassifier(**params)
            m.fit(X_tr, y_tr)
            sc = m.predict_proba(X_te)[:, 1]
            r = eval_model(sc, y_te, f"lgbm_{variant}_{feat_name}")
            r["model"] = m
            r["X_tr"], r["X_te"] = X_tr, X_te
            results.append(r)

    # LightGBM with cleaned labels
    print("\n--- LightGBM + Cleaned Labels ---")
    for feat_name, X_tr, X_te in [("ext", X_tr_ext, X_te_ext), ("full", X_tr_full, X_te_full)]:
        params = get_lgbm_params("base")
        n_neg_c = (y_tr_clean==0).sum()
        n_pos_c = max((y_tr_clean==1).sum(), 1)
        params["scale_pos_weight"] = n_neg_c / n_pos_c
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr_clean)
        sc = m.predict_proba(X_te)[:, 1]
        r = eval_model(sc, y_te, f"lgbm_clean_{feat_name}")
        r["model"] = m
        results.append(r)

    # LightGBM multiclass (competing risks)
    print("\n--- LightGBM Multiclass (Competing Risks) ---")
    for feat_name, X_tr, X_te in [("ext", X_tr_ext, X_te_ext), ("full", X_tr_full, X_te_full)]:
        params = dict(n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31,
                     subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_SEED,
                     objective="multiclass", num_class=3, verbose=-1)
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr_mc)
        sc = m.predict_proba(X_te)[:, 1]
        r = eval_model(sc, y_te, f"lgbm_mc_{feat_name}")
        r["model"], r["is_mc"] = m, True

        # + isotonic
        sc_tr = m.predict_proba(X_tr)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(sc_tr, (y_tr_mc==1).astype(int))
        sc_iso = iso.transform(sc)
        r2 = eval_model(sc_iso, y_te, f"lgbm_mc_iso_{feat_name}")
        r2["model"], r2["calibrator"] = m, iso
        results.append(r)
        results.append(r2)

    if HAS_XGB:
        print("\n" + "="*70)
        print("EXPERIMENTS - XGBOOST VARIANTS")
        print("="*70)

        for variant in ["base", "deep", "reg"]:
            for feat_name, X_tr, X_te in [("ext", X_tr_ext, X_te_ext), ("full", X_tr_full, X_te_full)]:
                params = get_xgb_params(variant)
                params["scale_pos_weight"] = sw
                m = xgb.XGBClassifier(**params)
                m.fit(X_tr, y_tr)
                sc = m.predict_proba(X_te)[:, 1]
                r = eval_model(sc, y_te, f"xgb_{variant}_{feat_name}")
                r["model"] = m
                results.append(r)

        # XGBoost multiclass
        print("\n--- XGBoost Multiclass ---")
        for feat_name, X_tr, X_te in [("ext", X_tr_ext, X_te_ext)]:
            params = dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                         verbosity=0, random_state=RANDOM_SEED,
                         objective="multi:softprob", num_class=3,
                         use_label_encoder=False, eval_metric='mlogloss')
            m = xgb.XGBClassifier(**params)
            m.fit(X_tr, y_tr_mc)
            sc = m.predict_proba(X_te)[:, 1]
            r = eval_model(sc, y_te, f"xgb_mc_{feat_name}")
            r["model"] = m

            # + isotonic
            sc_tr = m.predict_proba(X_tr)[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(sc_tr, (y_tr_mc==1).astype(int))
            sc_iso = iso.transform(sc)
            r2 = eval_model(sc_iso, y_te, f"xgb_mc_iso_{feat_name}")
            r2["model"], r2["calibrator"] = m, iso
            results.append(r)
            results.append(r2)

    if HAS_CB:
        print("\n" + "="*70)
        print("EXPERIMENTS - CATBOOST VARIANTS")
        print("="*70)

        for variant in ["base", "deep", "ordered"]:
            for feat_name, X_tr, X_te in [("ext", X_tr_ext, X_te_ext)]:
                params = get_cb_params(variant)
                params["scale_pos_weight"] = sw
                m = cb.CatBoostClassifier(**params)
                m.fit(X_tr, y_tr)
                sc = m.predict_proba(X_te)[:, 1]
                r = eval_model(sc, y_te, f"cb_{variant}_{feat_name}")
                r["model"] = m
                results.append(r)

        # CatBoost multiclass
        print("\n--- CatBoost Multiclass ---")
        params = dict(iterations=300, learning_rate=0.05, depth=6,
                     verbose=False, random_state=RANDOM_SEED,
                     loss_function="MultiClass")
        m = cb.CatBoostClassifier(**params)
        m.fit(X_tr_ext, y_tr_mc)
        sc = m.predict_proba(X_te_ext)[:, 1]
        r = eval_model(sc, y_te, "cb_mc_ext")
        r["model"] = m

        # + isotonic
        sc_tr = m.predict_proba(X_tr_ext)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(sc_tr, (y_tr_mc==1).astype(int))
        sc_iso = iso.transform(sc)
        r2 = eval_model(sc_iso, y_te, "cb_mc_iso_ext")
        r2["model"], r2["calibrator"] = m, iso
        results.append(r)
        results.append(r2)

    print("\n" + "="*70)
    print("EXPERIMENTS - SKLEARN MODELS")
    print("="*70)

    # HistGradientBoosting (native sklearn, very fast)
    print("\n--- HistGradientBoosting ---")
    for max_iter in [200, 400]:
        for max_depth in [6, 10, None]:
            m = HistGradientBoostingClassifier(
                max_iter=max_iter, max_depth=max_depth,
                learning_rate=0.05, random_state=RANDOM_SEED
            )
            m.fit(X_tr_ext, y_tr)
            sc = m.predict_proba(X_te_ext)[:, 1]
            r = eval_model(sc, y_te, f"histgb_d{max_depth}_i{max_iter}")
            r["model"] = m
            results.append(r)

    # Random Forest
    print("\n--- Random Forest ---")
    for n_est in [200, 500]:
        for max_depth in [10, 20, None]:
            m = RandomForestClassifier(
                n_estimators=n_est, max_depth=max_depth,
                class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
            )
            m.fit(X_tr_ext, y_tr)
            sc = m.predict_proba(X_te_ext)[:, 1]
            r = eval_model(sc, y_te, f"rf_d{max_depth}_n{n_est}")
            r["model"] = m
            results.append(r)

    # Extra Trees
    print("\n--- Extra Trees ---")
    m = ExtraTreesClassifier(n_estimators=500, max_depth=20,
                             class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
    m.fit(X_tr_ext, y_tr)
    sc = m.predict_proba(X_te_ext)[:, 1]
    r = eval_model(sc, y_te, "et_d20_n500")
    r["model"] = m
    results.append(r)

    print("\n" + "="*70)
    print("EXPERIMENTS - ENSEMBLES")
    print("="*70)

    # Voting ensemble of best models
    print("\n--- Voting Ensemble ---")
    best_results = sorted(results, key=lambda x: x["lift_10"], reverse=True)[:5]
    best_models = [(r["label"], r["model"]) for r in best_results if "model" in r]

    if len(best_models) >= 3:
        # Soft voting
        scores_ensemble = np.zeros(len(y_te))
        for name, m in best_models[:3]:
            try:
                sc = m.predict_proba(X_te_ext)
                if sc.shape[1] == 3:  # multiclass
                    scores_ensemble += sc[:, 1]
                else:
                    scores_ensemble += sc[:, 1]
            except:
                pass
        scores_ensemble /= 3
        r = eval_model(scores_ensemble, y_te, "voting_top3")
        results.append(r)

        # Weighted average based on lift
        weights = np.array([best_results[i]["lift_10"] for i in range(3)])
        weights = weights / weights.sum()
        scores_weighted = np.zeros(len(y_te))
        for i, (name, m) in enumerate(best_models[:3]):
            try:
                sc = m.predict_proba(X_te_ext)
                if sc.shape[1] == 3:
                    scores_weighted += weights[i] * sc[:, 1]
                else:
                    scores_weighted += weights[i] * sc[:, 1]
            except:
                pass
        r = eval_model(scores_weighted, y_te, "voting_weighted_top3")
        results.append(r)

    # Stacking
    print("\n--- Stacking ---")
    base_estimators = [
        ("lgbm", lgb.LGBMClassifier(**get_lgbm_params("base"), scale_pos_weight=sw)),
        ("histgb", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=RANDOM_SEED)),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=15, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)),
    ]

    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3, n_jobs=-1
    )
    stack.fit(X_tr_ext, y_tr)
    sc = stack.predict_proba(X_te_ext)[:, 1]
    r = eval_model(sc, y_te, "stacking_3models")
    r["model"] = stack
    results.append(r)

    # Calibrated stacking + isotonic
    iso = IsotonicRegression(out_of_bounds="clip")
    sc_tr = stack.predict_proba(X_tr_ext)[:, 1]
    iso.fit(sc_tr, y_tr)
    sc_iso = iso.transform(sc)
    r2 = eval_model(sc_iso, y_te, "stacking_3models_iso")
    r2["model"], r2["calibrator"] = stack, iso
    results.append(r2)

    return results

def run_calibration_experiments(results, df, i_tr, i_te):
    """Apply various calibration methods to top models."""
    print("\n" + "="*70)
    print("CALIBRATION EXPERIMENTS")
    print("="*70)

    df_tr = df.iloc[i_tr].copy()
    df_te = df.iloc[i_te].copy()
    y_te = df_te["event"].values
    y_tr = df_tr["event"].values

    # Get top 5 non-calibrated models
    top_models = [r for r in results if "calibrator" not in r and "model" in r]
    top_models = sorted(top_models, key=lambda x: x["lift_10"], reverse=True)[:5]

    calibrated_results = []

    for r in top_models:
        label = r["label"]
        m = r["model"]
        X_tr = r.get("X_tr", get_X(df_tr, FEAT_EXT))
        X_te = r.get("X_te", get_X(df_te, FEAT_EXT))

        # Get raw scores
        try:
            sc_tr = m.predict_proba(X_tr)
            sc_te = m.predict_proba(X_te)
            if sc_tr.shape[1] == 3:
                sc_tr = sc_tr[:, 1]
                sc_te = sc_te[:, 1]
            else:
                sc_tr = sc_tr[:, 1]
                sc_te = sc_te[:, 1]
        except:
            continue

        # Isotonic
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(sc_tr, y_tr)
        sc_iso = iso.transform(sc_te)
        r_iso = eval_model(sc_iso, y_te, f"{label}_iso")
        r_iso["model"], r_iso["calibrator"] = m, iso
        calibrated_results.append(r_iso)

        # Platt scaling
        platt = LogisticRegression(max_iter=1000)
        platt.fit(sc_tr.reshape(-1, 1), y_tr)
        sc_platt = platt.predict_proba(sc_te.reshape(-1, 1))[:, 1]
        r_platt = eval_model(sc_platt, y_te, f"{label}_platt")
        r_platt["model"], r_platt["calibrator"] = m, platt
        calibrated_results.append(r_platt)

        # Temperature scaling (simple version)
        temps = [0.5, 0.7, 1.0, 1.5, 2.0]
        for t in temps:
            sc_temp = 1 / (1 + np.exp(-np.log(sc_te / (1 - sc_te + 1e-10)) / t))
            r_temp = eval_model(sc_temp, y_te, f"{label}_temp{t}")
            calibrated_results.append(r_temp)

    return calibrated_results

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("PHASE 4 - RECHERCHE EXHAUSTIVE MEILLEUR MODELE")
    print(f"Objectif: battre Lift@10% = {BEST_LIFT10}")
    print("="*70)

    # Load data
    df = load_and_prepare()
    print(f"\nDataset: {len(df):,} | events: {df['event'].sum():,}")

    # Fixed split
    idx = np.arange(len(df))
    i_tr, i_te = train_test_split(idx, test_size=0.2, random_state=RANDOM_SEED,
                                   stratify=df["event"].values)
    print(f"Train: {len(i_tr):,} | Test: {len(i_te):,}")

    # Run experiments
    results = run_experiments(df, i_tr, i_te)

    # Run calibration experiments
    calibrated = run_calibration_experiments(results, df, i_tr, i_te)
    results.extend(calibrated)

    # Final leaderboard
    print("\n" + "="*70)
    print("LEADERBOARD FINAL")
    print("="*70)

    # Sort by lift
    results_sorted = sorted(results, key=lambda x: x["lift_10"], reverse=True)

    rows = []
    for r in results_sorted[:30]:
        rows.append({
            "Model": r["label"],
            "Lift@10%": r["lift_10"],
            "Cap@10%": r["cap_10"],
            "Brier": r["brier"],
            "ROC": r["roc_auc"],
            "FN": r["fn"],
            "vs_best": f"{100*(r['lift_10']/BEST_LIFT10-1):+.1f}%"
        })

    df_lb = pd.DataFrame(rows)
    print("\n" + df_lb.to_string(index=False))

    # Save leaderboard
    df_lb.to_csv(PHASE4_DIR / "leaderboard_phase4.csv", index=False)

    # Best model
    best = results_sorted[0]
    print(f"\n>>> MEILLEUR MODELE: {best['label']}")
    print(f"    Lift@10%: {best['lift_10']:.3f} ({100*(best['lift_10']/BEST_LIFT10-1):+.1f}% vs M4)")
    print(f"    Cap@10%: {best['cap_10']:.1%}")
    print(f"    Brier: {best['brier']:.4f}")
    print(f"    FN: {best['fn']}")

    # Save best model if improved
    if best["lift_10"] > BEST_LIFT10:
        print("\n>>> NOUVEAU RECORD! Sauvegarde du modele...")
        artifact = {
            "model": best.get("model"),
            "calibrator": best.get("calibrator"),
            "label": best["label"],
            "metrics": {
                "lift_10": best["lift_10"],
                "cap_10": best["cap_10"],
                "brier": best["brier"],
                "roc_auc": best["roc_auc"],
                "fn": best["fn"],
            },
            "created_at": datetime.now().isoformat(),
        }
        joblib.dump(artifact, ARTIFACTS_DIR / "best_model_phase4.joblib")

        with open(ARTIFACTS_DIR / "metadata_phase4.json", "w") as f:
            json.dump({k: v for k, v in artifact.items() if k != "model" and k != "calibrator"},
                     f, indent=2, default=str)

    return results_sorted

if __name__ == "__main__":
    results = main()
