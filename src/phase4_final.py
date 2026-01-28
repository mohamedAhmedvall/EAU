#!/usr/bin/env python3
"""
PHASE 4 FINAL — Modele definitif avec ensemble avance
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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb

try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False

from config import ARTIFACTS_DIR, REPORTS_DIR, RANDOM_SEED

PHASE4_DIR = REPORTS_DIR / "phase4_final"
PHASE4_DIR.mkdir(parents=True, exist_ok=True)

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
        rows.append(dict(k_pct=k_pct, capture=cap, lift=cap/k_pct, events_in_top=ev_top))
    return pd.DataFrame(rows)

def confusion_at_k(y_true, scores, k_pct=0.10):
    y = np.asarray(y_true); s = np.asarray(scores)
    n = len(y); k = int(np.ceil(n * k_pct))
    order = np.argsort(-s)
    top = np.zeros(n, dtype=bool); top[order[:k]] = True
    tp = int(((y==1)&top).sum()); fn = int(((y==1)&~top).sum())
    return dict(TP=tp, FN=fn)

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
        ev = int(y[mask].sum())
        if ev == 0: continue
        ev_cap = int((y[mask] & top[mask]).sum())
        rows.append(dict(segment=g, events=ev, capture=ev_cap/ev))
    return pd.DataFrame(rows)

def eval_full(scores, y_test, label):
    tk = topk_metrics(y_test, scores)
    conf = confusion_at_k(y_test, scores, 0.10)
    roc = roc_auc_score(y_test, scores)
    brier = brier_score_loss(y_test, scores)
    ec = ece_score(y_test, scores)
    l10 = float(tk.loc[tk["k_pct"]==0.10, "lift"].values[0])
    c10 = float(tk.loc[tk["k_pct"]==0.10, "capture"].values[0])
    print(f"  [{label}] L@10%={l10:.3f} C@10%={c10:.1%} Brier={brier:.4f} ROC={roc:.4f} FN={conf['FN']}")
    return dict(label=label, lift_10=l10, cap_10=c10, brier=brier, roc_auc=roc, ece=ec, fn=conf['FN'], scores=scores)

# ============================================================================
# DATA
# ============================================================================

def load_data():
    df = pd.read_pickle(ARTIFACTS_DIR / "dataset_freeze_h1.pkl")

    for col in ["materiau", "decade_install"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col+"_enc"] = le.fit_transform(df[col].astype(str))

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

    # Competing risks labels
    cond = ((df["event"]==1) & (df["n_fuites_total"]==0) &
            (df["ratio_age_median"]<0.6) & (df["age_at_freeze"]<15))
    df["event_class"] = 0
    df.loc[(df["event"]==1)&(~cond), "event_class"] = 1
    df.loc[cond, "event_class"] = 2

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
    return df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

# ============================================================================
# ENSEMBLE MODEL
# ============================================================================

class EnsembleModel:
    """Ensemble of best models with weighted averaging."""

    def __init__(self):
        self.models = []
        self.weights = []
        self.calibrator = None

    def fit(self, X, y, y_mc=None):
        n_neg = (y==0).sum(); n_pos = max((y==1).sum(), 1)
        sw = n_neg / n_pos

        # Model 1: HistGradientBoosting (best single model)
        m1 = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.03, max_depth=None,
            min_samples_leaf=50, random_state=RANDOM_SEED
        )
        m1.fit(X, y)
        self.models.append(("hgb1", m1, "binary"))

        # Model 2: HistGradientBoosting variant
        m2 = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=None,
            min_samples_leaf=20, random_state=RANDOM_SEED
        )
        m2.fit(X, y)
        self.models.append(("hgb2", m2, "binary"))

        # Model 3: LightGBM
        m3 = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.9, colsample_bytree=0.9,
            verbose=-1, random_state=RANDOM_SEED, scale_pos_weight=sw
        )
        m3.fit(X, y)
        self.models.append(("lgbm", m3, "binary"))

        # Model 4: XGBoost
        if HAS_XGB:
            m4 = xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.9, colsample_bytree=0.9, verbosity=0,
                random_state=RANDOM_SEED, scale_pos_weight=sw,
                use_label_encoder=False, eval_metric='logloss'
            )
            m4.fit(X, y)
            self.models.append(("xgb", m4, "binary"))

        # Model 5: LightGBM multiclass
        if y_mc is not None:
            m5 = lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                num_leaves=31, subsample=0.9, colsample_bytree=0.9,
                random_state=RANDOM_SEED, objective="multiclass",
                num_class=3, verbose=-1
            )
            m5.fit(X, y_mc)
            self.models.append(("lgbm_mc", m5, "multiclass"))

        # Compute optimal weights via CV
        self._compute_weights(X, y, y_mc)

        # Fit calibrator
        scores_train = self._raw_predict(X)
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(scores_train, y)

        return self

    def _compute_weights(self, X, y, y_mc):
        """Compute weights based on CV lift."""
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        model_lifts = {name: [] for name, _, _ in self.models}

        for tr_i, te_i in skf.split(X, y):
            X_tr, X_te = X.iloc[tr_i], X.iloc[te_i]
            y_tr, y_te = y[tr_i], y[te_i]
            y_mc_tr = y_mc[tr_i] if y_mc is not None else None

            for name, m, mtype in self.models:
                # Clone and fit
                if mtype == "binary":
                    m_clone = m.__class__(**m.get_params())
                    m_clone.fit(X_tr, y_tr)
                    sc = m_clone.predict_proba(X_te)[:, 1]
                else:
                    m_clone = m.__class__(**m.get_params())
                    m_clone.fit(X_tr, y_mc_tr)
                    sc = m_clone.predict_proba(X_te)[:, 1]

                tk = topk_metrics(y_te, sc)
                lift = float(tk.loc[tk["k_pct"]==0.10, "lift"].values[0])
                model_lifts[name].append(lift)

        # Weights proportional to mean lift
        mean_lifts = {name: np.mean(lifts) for name, lifts in model_lifts.items()}
        total = sum(mean_lifts.values())
        self.weights = {name: lift/total for name, lift in mean_lifts.items()}
        print(f"  Ensemble weights: {self.weights}")

    def _raw_predict(self, X):
        scores = np.zeros(len(X))
        for name, m, mtype in self.models:
            w = self.weights[name]
            if mtype == "binary":
                scores += w * m.predict_proba(X)[:, 1]
            else:
                scores += w * m.predict_proba(X)[:, 1]
        return scores

    def predict_proba(self, X):
        scores_raw = self._raw_predict(X)
        scores_cal = self.calibrator.transform(scores_raw)
        return np.column_stack([1-scores_cal, scores_cal])

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("PHASE 4 FINAL - MODELE DEFINITIF")
    print("="*70)

    df = load_data()
    print(f"Dataset: {len(df):,} | events: {df['event'].sum():,}")

    idx = np.arange(len(df))
    i_tr, i_te = train_test_split(idx, test_size=0.2, random_state=RANDOM_SEED,
                                   stratify=df["event"].values)

    df_tr = df.iloc[i_tr].copy()
    df_te = df.iloc[i_te].copy()
    y_tr = df_tr["event"].values
    y_te = df_te["event"].values
    y_mc_tr = df_tr["event_class"].values

    X_tr = get_X(df_tr)
    X_te = get_X(df_te)

    results = []

    # ======================================================================
    # 1. Best single models
    # ======================================================================
    print("\n" + "="*70)
    print("1. BEST SINGLE MODELS")
    print("="*70)

    # HistGB best config
    m1 = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.03, max_depth=None,
        min_samples_leaf=50, random_state=RANDOM_SEED
    )
    m1.fit(X_tr, y_tr)
    sc1 = m1.predict_proba(X_te)[:, 1]
    r1 = eval_full(sc1, y_te, "hgb_best")
    results.append(r1)

    # XGBoost
    if HAS_XGB:
        n_neg = (y_tr==0).sum(); n_pos = max((y_tr==1).sum(), 1)
        m2 = xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, verbosity=0,
            random_state=RANDOM_SEED, scale_pos_weight=n_neg/n_pos,
            use_label_encoder=False, eval_metric='logloss'
        )
        m2.fit(X_tr, y_tr)
        sc2 = m2.predict_proba(X_te)[:, 1]
        r2 = eval_full(sc2, y_te, "xgb_best")
        results.append(r2)

    # ======================================================================
    # 2. Ensemble model
    # ======================================================================
    print("\n" + "="*70)
    print("2. ENSEMBLE MODEL")
    print("="*70)

    ensemble = EnsembleModel()
    ensemble.fit(X_tr, y_tr, y_mc_tr)
    sc_ens = ensemble.predict_proba(X_te)[:, 1]
    r_ens = eval_full(sc_ens, y_te, "ensemble_weighted_calibrated")
    r_ens["model"] = ensemble
    results.append(r_ens)

    # ======================================================================
    # 3. Simple average of top 2
    # ======================================================================
    print("\n" + "="*70)
    print("3. SIMPLE COMBINATIONS")
    print("="*70)

    # HGB + XGB average
    if HAS_XGB:
        sc_avg = (sc1 + sc2) / 2
        r_avg = eval_full(sc_avg, y_te, "hgb_xgb_avg")
        results.append(r_avg)

        # Calibrate
        iso = IsotonicRegression(out_of_bounds="clip")
        sc1_tr = m1.predict_proba(X_tr)[:, 1]
        sc2_tr = m2.predict_proba(X_tr)[:, 1]
        sc_avg_tr = (sc1_tr + sc2_tr) / 2
        iso.fit(sc_avg_tr, y_tr)
        sc_avg_cal = iso.transform(sc_avg)
        r_avg_cal = eval_full(sc_avg_cal, y_te, "hgb_xgb_avg_iso")
        results.append(r_avg_cal)

    # ======================================================================
    # 4. Final leaderboard
    # ======================================================================
    print("\n" + "="*70)
    print("FINAL LEADERBOARD")
    print("="*70)

    results_sorted = sorted(results, key=lambda x: x["lift_10"], reverse=True)
    for r in results_sorted:
        print(f"  {r['label']}: L@10%={r['lift_10']:.3f} C@10%={r['cap_10']:.1%} Brier={r['brier']:.4f} FN={r['fn']}")

    # ======================================================================
    # 5. Save best model
    # ======================================================================
    best = results_sorted[0]
    print(f"\n>>> MEILLEUR MODELE: {best['label']}")
    print(f"    Lift@10%: {best['lift_10']:.3f}")
    print(f"    Capture@10%: {best['cap_10']:.1%}")
    print(f"    Brier: {best['brier']:.4f}")
    print(f"    FN: {best['fn']}")

    # Segment analysis
    print("\n" + "="*70)
    print("SEGMENT ANALYSIS")
    print("="*70)

    seg_mat = segment_capture(y_te, best["scores"], df_te["materiau"].values)
    seg_dec = segment_capture(y_te, best["scores"], df_te["decade_install"].values)

    print("\nPar materiau:")
    print(seg_mat.to_string(index=False))
    print("\nPar decennie:")
    print(seg_dec.to_string(index=False))

    # Save artifact
    if "model" in best:
        artifact = {
            "model": best["model"],
            "label": best["label"],
            "features": FEATURES,
            "metrics": {
                "lift_10": best["lift_10"],
                "cap_10": best["cap_10"],
                "brier": best["brier"],
                "roc_auc": best["roc_auc"],
                "ece": best["ece"],
                "fn": best["fn"],
            },
            "created_at": datetime.now().isoformat(),
        }
        joblib.dump(artifact, ARTIFACTS_DIR / "best_model_final.joblib")
        print(f"\nModele sauvegarde: artifacts/best_model_final.joblib")

        # Metadata
        meta = {k: v for k, v in artifact.items() if k not in ["model"]}
        with open(ARTIFACTS_DIR / "metadata_final.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

    # ======================================================================
    # 6. Cross-validation check
    # ======================================================================
    print("\n" + "="*70)
    print("CROSS-VALIDATION")
    print("="*70)

    X_all = get_X(df)
    y_all = df["event"].values
    y_mc_all = df["event_class"].values

    cv_lifts = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for tr_i, te_i in skf.split(X_all, y_all):
        ens_cv = EnsembleModel()
        ens_cv.fit(X_all.iloc[tr_i], y_all[tr_i], y_mc_all[tr_i])
        sc_cv = ens_cv.predict_proba(X_all.iloc[te_i])[:, 1]
        tk = topk_metrics(y_all[te_i], sc_cv)
        cv_lifts.append(float(tk.loc[tk["k_pct"]==0.10, "lift"].values[0]))

    print(f"  CV Lift@10%: {np.mean(cv_lifts):.3f} +/- {np.std(cv_lifts):.3f}")

    return results_sorted

if __name__ == "__main__":
    main()
