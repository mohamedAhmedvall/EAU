"""
Phase 2 iterative improvements for pipe renewal prioritisation.
M0: existing LightGBM (best_model.joblib).
M1: bias corrections (age winsorisation + preventive-abandon relabel).
M2: calibration + monotone constraints / regularisation.

All steps preserve anti-leakage by using only features available at freeze_date.
"""

import json
from pathlib import Path
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
from modeling import compute_business_metrics


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

    return iso_test, lr_test


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
    }


def run_M1(df):
    df1 = add_age_robust_features(df)
    df1 = flag_preventive_abandon(df1)
    df1["event_mod"] = df1["event"].copy()
    # relabel preventive abandons to 0
    df1.loc[df1["is_preventive_abandon"] == 1, "event_mod"] = 0

    base_features = [
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
    X, y, enc = prepare_features(df1, base_features)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f\"[M1] train {len(X_train):,} / test {len(X_test):,} (pos {y_train.sum():,})\")
    model = train_lgbm(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    metrics = topk_metrics(y_test, scores)
    conf = confusion_topk(y_test, scores, 0.10)
    roc = roc_auc_score(y_test, scores)
    # save artifact
    artifact = {
        "model": model,
        "encoders": enc,
        "metadata": {
            "horizon_years": 1,
            "version": "M1",
            "feature_cols": list(X.columns),
            "event_definition": "event relabeled when preventive_abandon",
        },
    }
    joblib.dump(artifact, ARTIFACTS_DIR / "model_M1.joblib")
    return {
        "name": "M1_bias_corrections",
        "roc_auc": roc,
        "topk": metrics,
        "conf10": conf,
        "scores": scores,
        "y_test": y_test,
        "model": model,
        "enc": enc,
        "feature_cols": list(X.columns),
    }


def run_M2(df):
    df2 = add_age_robust_features(df)
    df2 = flag_preventive_abandon(df2)
    df2["event_mod"] = df2["event"].copy()
    df2.loc[df2["is_preventive_abandon"] == 1, "event_mod"] = 0

    base_features = [
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
    iso_test, platt_test = calibrate_scores(scores_train, y_train.values, scores_test)

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
            "train_stats": {
                "brier_train": brier_score_loss(y_train, scores_train)
            }
        }
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
        }
    }
    joblib.dump(artifact_platt, ARTIFACTS_DIR / "model_M2_platt.joblib")

    return {
        "name": "M2_monotone_calibrated",
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
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

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
    m2 = run_M2(df)
    results["M2"] = m2

    # Save summary
    summary = {
        "M0": {
            "roc_auc": baseline["roc_auc"],
            "lift10": float(baseline["topk"].loc[baseline["topk"]["k_pct"]==0.10, "lift"].values[0]),
            "capture10": float(baseline["topk"].loc[baseline["topk"]["k_pct"]==0.10, "capture"].values[0]),
        },
        "M1": {
            "roc_auc": m1["roc_auc"],
            "lift10": float(m1["topk"].loc[m1["topk"]["k_pct"]==0.10, "lift"].values[0]),
            "capture10": float(m1["topk"].loc[m1["topk"]["k_pct"]==0.10, "capture"].values[0]),
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
    with open(OUTPUT_DIR / "summary_phase2.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Phase2 iterations done ===")
    for k, v in summary.items():
        print(k, v)


if __name__ == "__main__":
    main()
