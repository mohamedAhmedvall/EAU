"""
Phase 2 iterative improvements for pipe renewal prioritisation.
M0: existing LightGBM (best_model.joblib).
M1: bias corrections (age winsorisation + preventive-abandon relabel).
M2: calibration + monotone constraints / regularisation.

All steps preserve anti-leakage by using only features available at freeze_date.
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb

from config import ARTIFACTS_DIR, REPORTS_DIR, RANDOM_SEED, TOP_K_PERCENTAGES


BASE_DATASET = ARTIFACTS_DIR / "dataset_freeze_h1.pkl"
OUTPUT_DIR = REPORTS_DIR / "phase2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Utility: metrics & top-k
# ---------------------------------------------------------------------------

def topk_metrics(y_true, scores, k_list=TOP_K_PERCENTAGES):
    n = len(y_true)
    n_events = y_true.sum()
    order = np.argsort(-scores)
    y_sorted = y_true.values[order]
    rows = []
    for k_pct in k_list:
        k = int(np.ceil(n * k_pct))
        events_in_top = y_sorted[:k].sum()
        capture = events_in_top / n_events if n_events > 0 else 0
        precision = events_in_top / k if k > 0 else 0
        lift = capture / k_pct if k_pct > 0 else 0
        rows.append({
            "k_pct": k_pct,
            "k": k,
            "capture": capture,
            "precision": precision,
            "lift": lift,
            "events_in_top": int(events_in_top)
        })
    return pd.DataFrame(rows)


def confusion_topk(y_true, scores, k_pct=0.10):
    n = len(y_true)
    k = int(np.ceil(n * k_pct))
    order = np.argsort(-scores)
    top_mask = np.zeros(n, dtype=bool)
    top_mask[order[:k]] = True
    y = y_true.values
    tp = int(((y == 1) & top_mask).sum())
    fp = int(((y == 0) & top_mask).sum())
    fn = int(((y == 1) & (~top_mask)).sum())
    tn = int(((y == 0) & (~top_mask)).sum())
    return {
        "k_pct": k_pct,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "fn_rate": fn / (tp + fn) if (tp + fn) > 0 else 0,
        "fp_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def load_base_dataset():
    df = pd.read_pickle(BASE_DATASET)
    return df.copy()


def add_age_robust_features(df, cap_years=110):
    df = df.copy()
    df["age_cap"] = df["age_at_freeze"].clip(upper=cap_years)
    df["age_extreme_flag"] = (df["age_at_freeze"] > cap_years).astype(int)
    return df


def add_structural_age_features(df):
    df = df.copy()
    decade_median = (
        df.groupby("decade_install", dropna=False)["age_at_freeze"]
        .median()
        .rename("decade_median_age")
    )
    df = df.join(decade_median, on="decade_install")
    df["age_ratio_decade"] = df["age_at_freeze"] / df["decade_median_age"].replace(0, np.nan)
    df["age_ratio_decade"] = df["age_ratio_decade"].fillna(1.0)
    return df


def flag_preventive_abandon(df, age_ratio_threshold=0.6, min_age_years=15):
    """Heuristic: event with no past anomalies, young vs material life, no recent leak."""
    df = df.copy()
    no_anom = df["n_fuites_total"] == 0
    young_vs_material = df["ratio_age_median"] < age_ratio_threshold
    young_absolute = df["age_at_freeze"] < min_age_years
    no_recent = df.get("has_recent_fuite", 0) == 0
    preventive = no_anom & young_vs_material & young_absolute & no_recent & (df["event"] == 1)
    df["is_preventive_abandon"] = preventive.astype(int)
    return df


def prepare_features(df, feature_cols, cat_cols=("materiau", "decade_install", "diametre_bin")):
    df_prep = df.copy()
    encoders = {}
    for col in cat_cols:
        if col in df_prep.columns:
            le = LabelEncoder()
            df_prep[col + "_encoded"] = le.fit_transform(df_prep[col].astype(str))
            encoders[col] = le
    X = df_prep[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_prep["event_mod"].astype(int)
    return X, y, encoders


def compute_segment_capture(df_eval: pd.DataFrame, segment_col: str, top_mask: np.ndarray):
    rows = []
    y = df_eval["event"].values.astype(int)
    scores = df_eval["score"].values

    for value, idx in df_eval.groupby(segment_col).groups.items():
        idx = np.array(idx)
        n_total = len(idx)
        events = int(y[idx].sum())
        if n_total < 50 or events == 0:
            continue

        events_captured = int(((y[idx] == 1) & top_mask[idx]).sum())
        capture = events_captured / events
        fn_rate = 1 - capture
        rows.append(
            {
                segment_col: value,
                "n_total": n_total,
                "events": events,
                "event_rate": events / n_total,
                "events_captured": events_captured,
                "capture_10pct": capture,
                "fn_rate": fn_rate,
                "avg_score": float(scores[idx].mean()),
            }
        )

    df_res = pd.DataFrame(rows)
    if not df_res.empty:
        df_res = df_res.sort_values("events", ascending=False)
    return df_res


def compute_age_extreme_impact(df_eval: pd.DataFrame, age_cap: int, top_mask: np.ndarray):
    df_extreme = df_eval["age_at_freeze"] > age_cap
    fn = int(((df_eval["event"] == 1) & (~top_mask)).sum())
    fp = int(((df_eval["event"] == 0) & top_mask).sum())

    extreme_fn = int(((df_eval["event"] == 1) & (~top_mask) & df_extreme).sum())
    extreme_fp = int(((df_eval["event"] == 0) & top_mask & df_extreme).sum())

    return {
        "age_cap": age_cap,
        "extreme_population": int(df_extreme.sum()),
        "extreme_fn_share": extreme_fn / fn if fn > 0 else 0,
        "extreme_fp_share": extreme_fp / fp if fp > 0 else 0,
        "extreme_event_rate": float(df_eval.loc[df_extreme, "event"].mean() if df_extreme.any() else 0),
        "overall_event_rate": float(df_eval["event"].mean()),
    }


def reliability_bins(y_true: pd.Series, scores: np.ndarray, n_bins: int = 10):
    df_tmp = pd.DataFrame({"y": y_true.values, "score": scores})
    df_tmp["bin"] = pd.qcut(df_tmp["score"], q=n_bins, duplicates="drop")
    return (
        df_tmp.groupby("bin")
        .agg(avg_score=("score", "mean"), event_rate=("y", "mean"), n=("y", "size"))
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Training / calibration
# ---------------------------------------------------------------------------

def train_lgbm(X_train, y_train, monotone_constraints=None, params=None):
    if params is None:
        params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": RANDOM_SEED,
            "scale_pos_weight": (len(y_train) - y_train.sum()) / max(y_train.sum(), 1),
        }
    if monotone_constraints is not None:
        params["monotone_constraints"] = monotone_constraints
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model


def calibrate_scores(train_scores, train_labels, test_scores):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(train_scores, train_labels)
    iso_test = iso.transform(test_scores)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(train_scores.reshape(-1, 1), train_labels)
    lr_test = lr.predict_proba(test_scores.reshape(-1, 1))[:, 1]

    return iso_test, lr_test, iso, lr


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_baseline_M0(df):
    artifact = joblib.load(ARTIFACTS_DIR / "best_model.joblib")
    model = artifact["model"]
    encoders = artifact["encoders"]
    # rebuild split for comparability
    df_prep = df.copy()
    for col, le in encoders.items():
        df_prep[col + "_encoded"] = df_prep[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else 0
        )
    X = df_prep[artifact["metadata"]["feature_cols"]].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_prep["event"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    scores = model.predict_proba(X_test)[:, 1]
    metrics = topk_metrics(y_test, scores)
    conf = confusion_topk(y_test, scores, 0.10)
    roc = roc_auc_score(y_test, scores)
    return {
        "name": "M0_baseline",
        "roc_auc": roc,
        "topk": metrics,
        "conf10": conf,
        "scores": scores,
        "y_test": y_test,
        "test_index": X_test.index,
        "feature_cols": artifact["metadata"]["feature_cols"],
    }


def train_binary_model(df, feature_cols, event_col="event_mod"):
    df_prep = df.copy()
    df_prep["event_mod"] = df_prep[event_col]
    X, y, enc = prepare_features(df_prep, feature_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    model = train_lgbm(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    metrics = topk_metrics(y_test, scores)
    conf = confusion_topk(y_test, scores, 0.10)
    roc = roc_auc_score(y_test, scores)
    return {
        "model": model,
        "encoders": enc,
        "feature_cols": list(X.columns),
        "scores": scores,
        "y_test": y_test,
        "test_index": X_test.index,
        "metrics": metrics,
        "conf10": conf,
        "roc_auc": roc,
    }


def train_competing_risk_model(df, feature_cols):
    df_multi = df.copy()
    df_multi["event_class"] = 0
    df_multi.loc[(df_multi["event"] == 1) & (df_multi["is_preventive_abandon"] == 0), "event_class"] = 1
    df_multi.loc[df_multi["is_preventive_abandon"] == 1, "event_class"] = 2
    df_multi["event_mod"] = df_multi["event_class"]

    X, y, enc = prepare_features(df_multi, feature_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    params = {
        "n_estimators": 260,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": RANDOM_SEED,
        "objective": "multiclass",
        "num_class": 3,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    scores = proba[:, 1]
    y_true = (y_test == 1).astype(int)
    metrics = topk_metrics(y_true, scores)
    conf = confusion_topk(y_true, scores, 0.10)
    roc = roc_auc_score(y_true, scores)
    return {
        "model": model,
        "encoders": enc,
        "feature_cols": list(X.columns),
        "scores": scores,
        "y_test": y_true,
        "test_index": X_test.index,
        "metrics": metrics,
        "conf10": conf,
        "roc_auc": roc,
    }


def run_M1(df):
    df1 = add_age_robust_features(df)
    df1 = add_structural_age_features(df1)
    df1 = flag_preventive_abandon(df1)
    df1["event_mod"] = df1["event"].copy()
    df1.loc[df1["is_preventive_abandon"] == 1, "event_mod"] = 0

    conservative_features = [
        "age_at_freeze",
        "age_cap",
        "age_extreme_flag",
        "diametre",
        "longueur",
        "log_longueur",
        "log_diametre",
        "n_fuites_total",
        "n_fuites_1y",
        "n_fuites_3y",
        "n_fuites_5y",
        "days_since_last_fuite",
        "has_recent_fuite",
        "leak_rate_per_year",
        "leak_rate_per_km",
        "ratio_age_median",
        "overdue_years",
        "over_p75_life",
        "over_p90_life",
        "age_x_nfuites",
        "surface_approx",
        "age_x_ratio",
        "materiau_encoded",
        "decade_install_encoded",
    ]

    structural_features = conservative_features + ["age_ratio_decade"]

    conservative = train_binary_model(df1, conservative_features, event_col="event_mod")
    structural = train_binary_model(df1, structural_features, event_col="event_mod")
    competing = train_competing_risk_model(df1, structural_features)

    return {
        "name": "M1_bias_corrections",
        "df": df1,
        "features": structural_features,
        "conservative": conservative,
        "structural": structural,
        "competing": competing,
    }


def run_M2(df, feature_cols, event_col="event_mod"):
    df2 = add_age_robust_features(df)
    df2 = add_structural_age_features(df2)
    df2 = flag_preventive_abandon(df2)
    df2["event_mod"] = df2["event"].copy()
    df2.loc[df2["is_preventive_abandon"] == 1, "event_mod"] = 0
    df2["event_mod"] = df2[event_col]

    base_features = feature_cols
    X, y, enc = prepare_features(df2, base_features)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f\"[M2] train {len(X_train):,} / test {len(X_test):,} (pos {y_train.sum():,})\")

    # Monotone constraints aligned to feature order
    mono = []
    for col in X.columns:
        if col in ["age_at_freeze", "age_cap", "ratio_age_median", "overdue_years", "over_p75_life", "over_p90_life",
                   "n_fuites_total", "n_fuites_1y", "n_fuites_3y", "n_fuites_5y",
                   "leak_rate_per_year", "leak_rate_per_km", "age_x_nfuites", "age_x_ratio", "surface_approx", "longueur", "diametre", "log_longueur", "log_diametre"]:
            mono.append(1)
        elif col == "days_since_last_fuite":
            mono.append(-1)
        else:
            mono.append(0)
    params = {
        "n_estimators": 220,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 1.0,
        "reg_alpha": 0.1,
        "random_state": RANDOM_SEED,
        "scale_pos_weight": (len(y_train) - y_train.sum()) / max(y_train.sum(), 1),
    }
    model = train_lgbm(X_train, y_train, monotone_constraints=mono, params=params)
    scores_test = model.predict_proba(X_test)[:, 1]

    # Calibration on train split
    scores_train = model.predict_proba(X_train)[:, 1]
    iso_test, platt_test, iso_model, platt_model = calibrate_scores(
        scores_train, y_train.values, scores_test
    )

    # Metrics raw
    metrics_raw = topk_metrics(y_test, scores_test)
    conf_raw = confusion_topk(y_test, scores_test, 0.10)
    roc_raw = roc_auc_score(y_test, scores_test)
    brier_raw = brier_score_loss(y_test, scores_test)

    # Metrics iso
    metrics_iso = topk_metrics(y_test, iso_test)
    conf_iso = confusion_topk(y_test, iso_test, 0.10)
    roc_iso = roc_auc_score(y_test, iso_test)
    brier_iso = brier_score_loss(y_test, iso_test)

    # Metrics platt
    metrics_platt = topk_metrics(y_test, platt_test)
    conf_platt = confusion_topk(y_test, platt_test, 0.10)
    roc_platt = roc_auc_score(y_test, platt_test)
    brier_platt = brier_score_loss(y_test, platt_test)

    artifact_raw = {
        "model": model,
        "encoders": enc,
        "metadata": {
            "horizon_years": 1,
            "version": "M2_raw",
            "feature_cols": list(X.columns),
            "monotone": mono,
            "reg": params,
        },
    }
    joblib.dump(artifact_raw, ARTIFACTS_DIR / "model_M2_raw.joblib")

    artifact_iso = {
        "model": model,
        "encoders": enc,
        "metadata": {
            "horizon_years": 1,
            "version": "M2_iso",
            "feature_cols": list(X.columns),
            "calibration": "isotonic",
            "monotone": mono,
            "reg": params,
        },
        "calibrator": {
            "type": "isotonic",
            "model": iso_model,
            "train_stats": {
                "brier_train": brier_score_loss(y_train, scores_train)
            },
        },
    }
    joblib.dump(artifact_iso, ARTIFACTS_DIR / "model_M2_iso.joblib")

    artifact_platt = {
        "model": model,
        "encoders": enc,
        "metadata": {
            "horizon_years": 1,
            "version": "M2_platt",
            "feature_cols": list(X.columns),
            "calibration": "platt",
            "monotone": mono,
            "reg": params,
        },
        "calibrator": {
            "type": "platt",
            "model": platt_model,
        },
    }
    joblib.dump(artifact_platt, ARTIFACTS_DIR / "model_M2_platt.joblib")

    return {
        "name": "M2_monotone_calibrated",
        "model": model,
        "encoders": enc,
        "iso_model": iso_model,
        "platt_model": platt_model,
        "feature_cols": list(X.columns),
        "roc_auc_raw": roc_raw,
        "topk_raw": metrics_raw,
        "conf_raw": conf_raw,
        "brier_raw": brier_raw,
        "roc_auc_iso": roc_iso,
        "topk_iso": metrics_iso,
        "conf_iso": conf_iso,
        "brier_iso": brier_iso,
        "roc_auc_platt": roc_platt,
        "topk_platt": metrics_platt,
        "conf_platt": conf_platt,
        "brier_platt": brier_platt,
        "scores_raw": scores_test,
        "scores_iso": iso_test,
        "scores_platt": platt_test,
        "y_test": y_test,
        "test_index": X_test.index,
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def write_phase2_report(summary: Dict, details: Dict):
    report_path = OUTPUT_DIR / "phase2_report.md"
    report = []
    report.append("# Rapport Phase 2 - Ameliorations du score de renouvellement\n")
    report.append("## 1. Rappel des constats PHASE 1\n")
    report.append("- Biais de survivants sur canalisations tres anciennes (1940-1950).")
    report.append("- Segments PEHD/FTVI/PVC sous-couverts.")
    report.append("- Probabilites non calibrees, usage du rang recommande.\n")

    report.append("## 2. Iteration 1 - Biais ages extremes & abandons preventifs\n")
    report.append("### Strategies ages extremes")
    report.append("- Conservative: capage de l'age et indicateur d'age extreme.")
    report.append("- Structurelle: ratio age/mediane par decennie pour reduire le survivorship bias.")

    report.append("\n### Abandons preventifs")
    report.append("- Heuristique: absence d'anomalies, age relatif faible, age absolu < 15 ans, pas de fuite recente.")
    report.append("- Approche A: relabel en non-evenement (event=0).")
    report.append("- Approche B: modele multi-classe (vraie defaillance vs abandon preventif).")

    report.append("\n### Resultats Iteration 1 (Lift@10%)")
    for key, vals in summary["M1"].items():
        report.append(f"- **{key}**: Lift@10%={vals['lift10']:.2f}, Capture@10%={vals['capture10']:.2%}")

    report.append("\n## 3. Iteration 2 - Calibration & stabilite\n")
    report.append("- Contraintes monotones + regularisation.")
    report.append("- Calibration isotonic / Platt comparee via Brier.")
    report.append(f"- Lift@10% raw={summary['M2_raw']['lift10']:.2f}, iso={summary['M2_iso']['lift10']:.2f}, platt={summary['M2_platt']['lift10']:.2f}")

    report.append("\n## 4. Iteration 3 - Comparaison finale & choix M*\n")
    report.append(f"- M0 Lift@10%: {summary['M0']['lift10']:.2f}")
    report.append(f"- M1 retenu: {summary['M1_best']['name']} (Lift@10% {summary['M1_best']['lift10']:.2f})")
    report.append(f"- M2 retenu: {summary['M2_best']['name']} (Lift@10% {summary['M2_best']['lift10']:.2f})")
    report.append(f"- Modele final: **{summary['M_star']['name']}**\n")

    report.append("## 5. Recommandation finale\n")
    report.append("- Le modele M* est retenu pour optimisation sous contrainte.")
    report.append("- Risque attendu = proba calibree * exposition (longueur ou cout).")
    report.append("- Segments sous-couverts a monitorer en production.\n")

    report_path.write_text("\n".join(report), encoding="utf-8")
    with open(OUTPUT_DIR / "phase2_details.json", "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)


def main():
    df = load_base_dataset()
    results = {}
    print("[START] Phase 2 iterations on dataset:", len(df))
    # Baseline
    print("[M0] evaluating baseline best_model.joblib ...")
    baseline = run_baseline_M0(df)
    results["M0"] = baseline

    # Iteration 1
    print("[M1] training with bias corrections ...")
    m1 = run_M1(df)
    results["M1"] = m1

    # Iteration 2
    print("[M2] training monotone + calibration ...")
    m1_candidates = {
        "conservative": m1["conservative"],
        "structural": m1["structural"],
        "competing": m1["competing"],
    }
    m1_best_name = max(
        m1_candidates,
        key=lambda k: m1_candidates[k]["metrics"].loc[
            m1_candidates[k]["metrics"]["k_pct"] == 0.10, "lift"
        ].values[0],
    )
    m1_best = m1_candidates[m1_best_name]
    m2 = run_M2(df, m1["features"])
    results["M2"] = m2

    # Save summary
    summary = {
        "M0": {
            "roc_auc": baseline["roc_auc"],
            "lift10": float(baseline["topk"].loc[baseline["topk"]["k_pct"]==0.10, "lift"].values[0]),
            "capture10": float(baseline["topk"].loc[baseline["topk"]["k_pct"]==0.10, "capture"].values[0]),
        },
        "M1": {
            name: {
                "roc_auc": m1_candidates[name]["roc_auc"],
                "lift10": float(
                    m1_candidates[name]["metrics"].loc[
                        m1_candidates[name]["metrics"]["k_pct"] == 0.10, "lift"
                    ].values[0]
                ),
                "capture10": float(
                    m1_candidates[name]["metrics"].loc[
                        m1_candidates[name]["metrics"]["k_pct"] == 0.10, "capture"
                    ].values[0]
                ),
            }
            for name in m1_candidates
        },
        "M1_best": {
            "name": m1_best_name,
            "lift10": float(
                m1_best["metrics"].loc[m1_best["metrics"]["k_pct"] == 0.10, "lift"].values[0]
            ),
            "capture10": float(
                m1_best["metrics"].loc[m1_best["metrics"]["k_pct"] == 0.10, "capture"].values[0]
            ),
        },
        "M2_raw": {
            "roc_auc": m2["roc_auc_raw"],
            "lift10": float(m2["topk_raw"].loc[m2["topk_raw"]["k_pct"]==0.10, "lift"].values[0]),
            "capture10": float(m2["topk_raw"].loc[m2["topk_raw"]["k_pct"]==0.10, "capture"].values[0]),
        },
        "M2_iso": {
            "roc_auc": m2["roc_auc_iso"],
            "lift10": float(m2["topk_iso"].loc[m2["topk_iso"]["k_pct"]==0.10, "lift"].values[0]),
            "capture10": float(m2["topk_iso"].loc[m2["topk_iso"]["k_pct"]==0.10, "capture"].values[0]),
            "brier": m2["brier_iso"],
        },
        "M2_platt": {
            "roc_auc": m2["roc_auc_platt"],
            "lift10": float(m2["topk_platt"].loc[m2["topk_platt"]["k_pct"]==0.10, "lift"].values[0]),
            "capture10": float(m2["topk_platt"].loc[m2["topk_platt"]["k_pct"]==0.10, "capture"].values[0]),
            "brier": m2["brier_platt"],
        },
    }

    m2_candidates = {
        "M2_raw": summary["M2_raw"]["lift10"],
        "M2_iso": summary["M2_iso"]["lift10"],
        "M2_platt": summary["M2_platt"]["lift10"],
    }
    m2_best_name = max(m2_candidates, key=m2_candidates.get)
    summary["M2_best"] = {
        "name": m2_best_name,
        "lift10": summary[m2_best_name]["lift10"],
        "capture10": summary[m2_best_name]["capture10"],
    }
    summary["M_star"] = (
        summary["M2_best"]
        if summary["M2_best"]["lift10"] >= summary["M1_best"]["lift10"]
        else summary["M1_best"]
    )

    m1_best_artifact = {
        "model": m1_best["model"],
        "encoders": m1_best["encoders"],
        "metadata": {
            "horizon_years": 1,
            "version": f"M1_{m1_best_name}",
            "feature_cols": m1_best["feature_cols"],
            "event_definition": "event relabeled when preventive_abandon",
        },
    }
    joblib.dump(m1_best_artifact, ARTIFACTS_DIR / "model_M1_best.joblib")

    if summary["M_star"]["name"] == m1_best_name:
        joblib.dump(m1_best_artifact, ARTIFACTS_DIR / "model_M_star.joblib")
    else:
        calibrator = None
        calibration_type = None
        if summary["M_star"]["name"] == "M2_iso":
            calibrator = m2["iso_model"]
            calibration_type = "isotonic"
        elif summary["M_star"]["name"] == "M2_platt":
            calibrator = m2["platt_model"]
            calibration_type = "platt"
        artifact_star = {
            "model": m2["model"],
            "encoders": m2["encoders"],
            "metadata": {
                "horizon_years": 1,
                "version": summary["M_star"]["name"],
                "feature_cols": m2["feature_cols"],
                "calibration": calibration_type,
            },
            "calibrator": calibrator,
        }
        joblib.dump(artifact_star, ARTIFACTS_DIR / "model_M_star.joblib")

    order = np.argsort(-baseline["scores"])
    top_mask = np.zeros(len(order), dtype=bool)
    top_mask[order[: int(np.ceil(len(order) * 0.10))]] = True
    df_eval = pd.DataFrame(
        {
            "event": baseline["y_test"].values,
            "score": baseline["scores"],
            "age_at_freeze": df.loc[baseline["test_index"], "age_at_freeze"].values,
        }
    )

    def segment_tables(model_result, score_key="scores"):
        scores = model_result[score_key]
        order_local = np.argsort(-scores)
        top_mask_local = np.zeros(len(scores), dtype=bool)
        top_mask_local[order_local[: int(np.ceil(len(scores) * 0.10))]] = True
        df_seg = df.loc[model_result["test_index"]].copy().reset_index(drop=True)
        df_seg["event"] = model_result["y_test"].values
        df_seg["score"] = scores
        segments = {}
        for col in ["materiau", "decade_install", "diametre_bin", "longueur_bin"]:
            if col in df_seg.columns:
                segments[col] = compute_segment_capture(df_seg, col, top_mask_local).to_dict(orient="records")
        return segments

    m2_best_scores = {
        "M2_raw": "scores_raw",
        "M2_iso": "scores_iso",
        "M2_platt": "scores_platt",
    }
    details = {
        "age_extreme_impact": compute_age_extreme_impact(df_eval, age_cap=110, top_mask=top_mask),
        "calibration_bins_raw": reliability_bins(m2["y_test"], m2["scores_raw"]).to_dict(orient="records"),
        "calibration_bins_iso": reliability_bins(m2["y_test"], m2["scores_iso"]).to_dict(orient="records"),
        "calibration_bins_platt": reliability_bins(m2["y_test"], m2["scores_platt"]).to_dict(orient="records"),
        "segment_m1_best": segment_tables(m1_best),
        "segment_m2_best": segment_tables(m2, score_key=m2_best_scores[m2_best_name]),
    }
    with open(OUTPUT_DIR / "summary_phase2.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_phase2_report(summary, details)

    print("=== Phase2 iterations done ===")
    for k, v in summary.items():
        print(k, v)


if __name__ == "__main__":
    main()
