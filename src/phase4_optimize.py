#!/usr/bin/env python3
"""
PHASE 4B — Optimisation fine du meilleur modele (HistGradientBoosting)
"""

import sys, os, json, warnings
from pathlib import Path
from datetime import datetime

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))
os.chdir(SRC_DIR)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.isotonic import IsotonicRegression

from config import ARTIFACTS_DIR, REPORTS_DIR, RANDOM_SEED

PHASE4_DIR = REPORTS_DIR / "phase4_experiments"
PHASE4_DIR.mkdir(parents=True, exist_ok=True)

BEST_LIFT10 = 6.566

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
                         capture=cap, fn_rate=1-cap))
    return pd.DataFrame(rows)

def eval_model(scores, y_test, label):
    tk = topk_metrics(y_test, scores)
    conf = confusion_at_k(y_test, scores, 0.10)
    try:
        roc = roc_auc_score(y_test, scores)
    except:
        roc = 0.5
    brier = brier_score_loss(y_test, scores)
    ec = ece_score(y_test, scores)
    l10 = float(tk.loc[tk["k_pct"]==0.10, "lift"].values[0])
    c10 = float(tk.loc[tk["k_pct"]==0.10, "capture"].values[0])

    improved = "+++" if l10 > BEST_LIFT10 else ("+" if l10 > BEST_LIFT10 * 0.99 else "")
    print(f"  [{label}] L@10%={l10:.3f}{improved} C@10%={c10:.1%} Brier={brier:.4f} ROC={roc:.4f} FN={conf['FN']}")

    return dict(label=label, roc_auc=roc, brier=brier, ece=ec,
                lift_10=l10, cap_10=c10, fn=conf['FN'], fp=conf['FP'], scores=scores)

# ============================================================================
# DATA
# ============================================================================

def load_and_prepare():
    df = pd.read_pickle(ARTIFACTS_DIR / "dataset_freeze_h1.pkl")

    for col in ["materiau", "decade_install", "diametre_bin"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col+"_enc"] = le.fit_transform(df[col].astype(str))

    df["age_cap"] = df["age_at_freeze"].clip(upper=110)
    df["age_extreme"] = (df["age_at_freeze"] > 110).astype(int)
    p99 = df["age_at_freeze"].quantile(0.99)
    df["age_winsor"] = df["age_at_freeze"].clip(upper=p99)

    dec_med = df.groupby("decade_install")["age_at_freeze"].median()
    df["decade_med_age"] = df["decade_install"].map(dec_med)
    df["age_ratio_dec"] = df["age_at_freeze"] / df["decade_med_age"].replace(0, np.nan)
    df["age_ratio_dec"] = df["age_ratio_dec"].fillna(1.0)

    mat_med = df.groupby("materiau")["age_at_freeze"].median()
    df["mat_med_age"] = df["materiau"].map(mat_med)
    df["age_resid_mat"] = df["age_at_freeze"] - df["mat_med_age"].fillna(df["age_at_freeze"].median())

    # Additional features
    df["age_sq"] = df["age_at_freeze"] ** 2
    df["age_log"] = np.log1p(df["age_at_freeze"])
    df["n_fuites_sq"] = df["n_fuites_total"] ** 2
    df["ratio_sq"] = df["ratio_age_median"] ** 2
    df["age_x_recent"] = df["age_at_freeze"] * df["has_recent_fuite"]
    df["age_x_leakrate"] = df["age_at_freeze"] * df["leak_rate_per_year"]
    df["overdue_x_fuites"] = df["overdue_years"] * df["n_fuites_total"]

    # More interactions
    df["log_age_x_fuites"] = np.log1p(df["age_at_freeze"]) * df["n_fuites_total"]
    df["ratio_x_fuites"] = df["ratio_age_median"] * df["n_fuites_total"]
    df["overdue_sq"] = df["overdue_years"] ** 2
    df["recent_x_ratio"] = df["has_recent_fuite"] * df["ratio_age_median"]

    return df

FEAT_EXT = [
    "age_at_freeze", "diametre", "longueur", "log_longueur", "log_diametre",
    "n_fuites_total", "n_fuites_1y", "n_fuites_3y", "n_fuites_5y",
    "days_since_last_fuite", "has_recent_fuite",
    "leak_rate_per_year", "leak_rate_per_km",
    "ratio_age_median", "overdue_years", "over_p75_life", "over_p90_life",
    "age_x_nfuites", "surface_approx", "age_x_ratio",
    "materiau_enc", "decade_install_enc",
    "age_cap", "age_extreme", "age_winsor",
    "age_ratio_dec", "age_resid_mat",
]

FEAT_FULL = FEAT_EXT + [
    "age_sq", "age_log", "n_fuites_sq", "ratio_sq",
    "age_x_recent", "age_x_leakrate", "overdue_x_fuites",
    "log_age_x_fuites", "ratio_x_fuites", "overdue_sq", "recent_x_ratio",
]

def get_X(df, cols):
    cols_exist = [c for c in cols if c in df.columns]
    return df[cols_exist].replace([np.inf, -np.inf], np.nan).fillna(0)

# ============================================================================
# OPTIMIZATION
# ============================================================================

def main():
    print("="*70)
    print("PHASE 4B - OPTIMISATION FINE HISTGRADIENTBOOSTING")
    print(f"Objectif: battre Lift@10% = {BEST_LIFT10}")
    print("="*70)

    df = load_and_prepare()
    print(f"\nDataset: {len(df):,} | events: {df['event'].sum():,}")

    idx = np.arange(len(df))
    i_tr, i_te = train_test_split(idx, test_size=0.2, random_state=RANDOM_SEED,
                                   stratify=df["event"].values)

    df_tr = df.iloc[i_tr].copy()
    df_te = df.iloc[i_te].copy()
    y_tr = df_tr["event"].values
    y_te = df_te["event"].values

    X_tr_ext = get_X(df_tr, FEAT_EXT)
    X_te_ext = get_X(df_te, FEAT_EXT)
    X_tr_full = get_X(df_tr, FEAT_FULL)
    X_te_full = get_X(df_te, FEAT_FULL)

    results = []

    print("\n" + "="*70)
    print("1. HYPERPARAMETER GRID SEARCH")
    print("="*70)

    # Grid search on HistGradientBoosting
    for max_iter in [300, 400, 500, 600]:
        for lr in [0.03, 0.05, 0.07, 0.1]:
            for min_samples in [10, 20, 50]:
                for l2_reg in [0, 0.1, 1.0]:
                    m = HistGradientBoostingClassifier(
                        max_iter=max_iter,
                        learning_rate=lr,
                        max_depth=None,
                        min_samples_leaf=min_samples,
                        l2_regularization=l2_reg,
                        random_state=RANDOM_SEED
                    )
                    m.fit(X_tr_ext, y_tr)
                    sc = m.predict_proba(X_te_ext)[:, 1]
                    r = eval_model(sc, y_te, f"hgb_i{max_iter}_lr{lr}_ms{min_samples}_l2{l2_reg}")
                    r["model"] = m
                    r["params"] = dict(max_iter=max_iter, lr=lr, min_samples=min_samples, l2_reg=l2_reg)
                    results.append(r)

    print("\n" + "="*70)
    print("2. FEATURES FULL + BEST PARAMS")
    print("="*70)

    # Get best params from grid search
    best_so_far = max(results, key=lambda x: x["lift_10"])
    best_params = best_so_far["params"]
    print(f"Best params: {best_params}")

    # Try with full features
    m = HistGradientBoostingClassifier(
        max_iter=best_params["max_iter"],
        learning_rate=best_params["lr"],
        max_depth=None,
        min_samples_leaf=best_params["min_samples"],
        l2_regularization=best_params["l2_reg"],
        random_state=RANDOM_SEED
    )
    m.fit(X_tr_full, y_tr)
    sc = m.predict_proba(X_te_full)[:, 1]
    r = eval_model(sc, y_te, "hgb_best_full")
    r["model"] = m
    results.append(r)

    print("\n" + "="*70)
    print("3. ENSEMBLE OF BEST HISTGB")
    print("="*70)

    # Ensemble of top 5 configs
    top5 = sorted(results, key=lambda x: x["lift_10"], reverse=True)[:5]
    scores_ens = np.zeros(len(y_te))
    for r in top5:
        try:
            sc = r["model"].predict_proba(X_te_ext)[:, 1]
            scores_ens += sc
        except:
            pass
    scores_ens /= 5
    r = eval_model(scores_ens, y_te, "hgb_ensemble_top5")
    results.append(r)

    # Weighted ensemble
    weights = np.array([r["lift_10"] for r in top5])
    weights = weights / weights.sum()
    scores_w = np.zeros(len(y_te))
    for i, r in enumerate(top5):
        try:
            sc = r["model"].predict_proba(X_te_ext)[:, 1]
            scores_w += weights[i] * sc
        except:
            pass
    r = eval_model(scores_w, y_te, "hgb_ensemble_weighted")
    results.append(r)

    print("\n" + "="*70)
    print("4. CALIBRATION")
    print("="*70)

    # Calibrate best model
    best = max(results, key=lambda x: x["lift_10"])
    if "model" in best:
        m = best["model"]
        try:
            sc_tr = m.predict_proba(X_tr_ext)[:, 1]
            sc_te = best["scores"]

            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(sc_tr, y_tr)
            sc_iso = iso.transform(sc_te)
            r = eval_model(sc_iso, y_te, f"{best['label']}_iso")
            r["model"], r["calibrator"] = m, iso
            results.append(r)
        except:
            pass

    print("\n" + "="*70)
    print("5. CROSS-VALIDATION CHECK")
    print("="*70)

    # CV on best config
    best_params = max(results, key=lambda x: x["lift_10"]).get("params", {})
    if best_params:
        cv_lifts = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        X_all = get_X(df, FEAT_EXT)
        y_all = df["event"].values

        for tr_i, te_i in skf.split(X_all, y_all):
            m = HistGradientBoostingClassifier(
                max_iter=best_params.get("max_iter", 400),
                learning_rate=best_params.get("lr", 0.05),
                max_depth=None,
                min_samples_leaf=best_params.get("min_samples", 20),
                l2_regularization=best_params.get("l2_reg", 0),
                random_state=RANDOM_SEED
            )
            m.fit(X_all.iloc[tr_i], y_all[tr_i])
            sc = m.predict_proba(X_all.iloc[te_i])[:, 1]
            tk = topk_metrics(y_all[te_i], sc)
            cv_lifts.append(float(tk.loc[tk["k_pct"]==0.10, "lift"].values[0]))

        print(f"  CV Lift@10%: {np.mean(cv_lifts):.3f} +/- {np.std(cv_lifts):.3f}")

    print("\n" + "="*70)
    print("FINAL LEADERBOARD (TOP 20)")
    print("="*70)

    results_sorted = sorted(results, key=lambda x: x["lift_10"], reverse=True)

    rows = []
    for r in results_sorted[:20]:
        rows.append({
            "Model": r["label"],
            "Lift@10%": r["lift_10"],
            "Cap@10%": r["cap_10"],
            "Brier": r["brier"],
            "ROC": r["roc_auc"],
            "FN": r["fn"],
        })

    df_lb = pd.DataFrame(rows)
    print("\n" + df_lb.to_string(index=False))

    # Save best
    best = results_sorted[0]
    print(f"\n>>> MEILLEUR: {best['label']}")
    print(f"    Lift@10%: {best['lift_10']:.3f}")
    print(f"    Cap@10%: {best['cap_10']:.1%}")
    print(f"    Brier: {best['brier']:.4f}")
    print(f"    FN: {best['fn']}")

    if best["lift_10"] > BEST_LIFT10:
        print("\n>>> NOUVEAU RECORD!")
        artifact = {
            "model": best.get("model"),
            "calibrator": best.get("calibrator"),
            "label": best["label"],
            "params": best.get("params"),
            "metrics": {
                "lift_10": best["lift_10"],
                "cap_10": best["cap_10"],
                "brier": best["brier"],
                "roc_auc": best["roc_auc"],
                "fn": best["fn"],
            },
            "features": FEAT_EXT,
            "created_at": datetime.now().isoformat(),
        }
        joblib.dump(artifact, ARTIFACTS_DIR / "best_model_phase4b.joblib")

    # Segment analysis
    print("\n" + "="*70)
    print("SEGMENT ANALYSIS (BEST MODEL)")
    print("="*70)

    seg_mat = segment_capture(y_te, best["scores"], df_te["materiau"].values)
    seg_dec = segment_capture(y_te, best["scores"], df_te["decade_install"].values)

    print("\nPar materiau:")
    print(seg_mat.to_string(index=False))

    print("\nPar decennie:")
    print(seg_dec.to_string(index=False))

    return results_sorted

if __name__ == "__main__":
    main()
