#!/usr/bin/env python3
"""
PHASE 3 — Pipeline Consolide pour Maintenance Predictive AEP
=============================================================

Ce script consolide toutes les iterations en un pipeline reproductible:
1. Reproduction baseline avec metriques completes
2. Tests anti-leakage automatiques (bloquants)
3. Iterations controlees (M0 -> M5+)
4. Selection du modele final
5. Generation des artefacts prod

Usage:
  python phase3_consolidated.py --horizon 1 --run-all
  python phase3_consolidated.py --horizon 1 --baseline-only
  python phase3_consolidated.py --horizon 1 --iterations-only

Auteur: ML Engineer Senior
Date: 2026-01-28
"""

import sys
import os
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Setup paths
SRC_DIR = Path(__file__).parent
BASE_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
os.chdir(SRC_DIR)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss,
    precision_recall_curve, average_precision_score
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import lightgbm as lgb

from config import ARTIFACTS_DIR, REPORTS_DIR, RANDOM_SEED, TOP_K_PERCENTAGES

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_HORIZON = 1
LEAKAGE_TESTS_BLOCKING = True  # Si True, stop si leakage detecte

# Features de base
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

# Features etendues (corrections biais)
EXTENDED_FEATURES = BASE_FEATURES + [
    "age_cap", "age_extreme_flag", "age_winsor",
    "age_ratio_decade", "age_residual_mat",
]


# =============================================================================
# TESTS ANTI-LEAKAGE (BLOQUANTS)
# =============================================================================

class LeakageError(Exception):
    """Exception raised when temporal leakage is detected."""
    pass


def test_no_future_anomalies(df: pd.DataFrame, freeze_date: pd.Timestamp) -> bool:
    """TEST 1: Aucune anomalie post-freeze dans les features."""
    if 'date_last_fuite' in df.columns:
        valid = df['date_last_fuite'].notna()
        if valid.any():
            max_date = df.loc[valid, 'date_last_fuite'].max()
            if max_date > freeze_date:
                raise LeakageError(
                    f"LEAKAGE: date_last_fuite max ({max_date}) > freeze_date ({freeze_date})"
                )
    return True


def test_no_hs_before_freeze(df: pd.DataFrame, freeze_date: pd.Timestamp) -> bool:
    """TEST 2: Aucun troncon deja HS avant freeze dans le dataset."""
    if 'DHS_parsed' in df.columns:
        has_dhs_before = df['DHS_parsed'].notna() & (df['DHS_parsed'] <= freeze_date)
        if has_dhs_before.any():
            raise LeakageError(
                f"LEAKAGE: {has_dhs_before.sum()} troncons avec DHS <= freeze_date"
            )
    return True


def test_features_temporal_consistency(df: pd.DataFrame, freeze_date: pd.Timestamp) -> bool:
    """TEST 3: Coherence temporelle des features."""
    # Age ne peut pas etre negatif
    if (df['age_at_freeze'] < 0).any():
        raise LeakageError("LEAKAGE: age_at_freeze negatif detecte")

    # n_fuites sliding windows coherentes
    if 'n_fuites_1y' in df.columns:
        if (df['n_fuites_1y'] > df['n_fuites_total']).any():
            raise LeakageError("LEAKAGE: n_fuites_1y > n_fuites_total")

    return True


def test_target_in_future(df: pd.DataFrame, freeze_date: pd.Timestamp, horizon_years: int) -> bool:
    """TEST 4: Le target (event) ne peut etre calcule qu'avec info future."""
    # Si event=1, DHS doit etre dans (freeze, freeze+H]
    horizon_end = freeze_date + pd.DateOffset(years=horizon_years)

    if 'DHS_parsed' in df.columns:
        events = df['event'] == 1
        if events.any():
            dhs_events = df.loc[events, 'DHS_parsed']
            invalid = (dhs_events <= freeze_date) | (dhs_events > horizon_end)
            if invalid.any():
                raise LeakageError(
                    f"LEAKAGE: {invalid.sum()} events avec DHS hors horizon"
                )
    return True


def run_all_leakage_tests(df: pd.DataFrame, freeze_date: pd.Timestamp,
                          horizon_years: int, blocking: bool = True) -> Dict[str, bool]:
    """Execute tous les tests anti-leakage."""
    print("\n" + "="*60)
    print("TESTS ANTI-LEAKAGE AUTOMATIQUES")
    print("="*60)

    results = {}
    tests = [
        ("future_anomalies", lambda: test_no_future_anomalies(df, freeze_date)),
        ("hs_before_freeze", lambda: test_no_hs_before_freeze(df, freeze_date)),
        ("features_consistency", lambda: test_features_temporal_consistency(df, freeze_date)),
        ("target_in_future", lambda: test_target_in_future(df, freeze_date, horizon_years)),
    ]

    all_passed = True
    for name, test_fn in tests:
        try:
            test_fn()
            results[name] = True
            print(f"  [PASS] {name}")
        except LeakageError as e:
            results[name] = False
            all_passed = False
            print(f"  [FAIL] {name}: {e}")
            if blocking:
                raise

    if all_passed:
        print("\n  >>> TOUS LES TESTS ANTI-LEAKAGE PASSES <<<")

    return results


# =============================================================================
# METRIQUES
# =============================================================================

def topk_metrics(y_true: np.ndarray, scores: np.ndarray,
                 k_list: List[float] = TOP_K_PERCENTAGES) -> pd.DataFrame:
    """Calcule Lift@K, Capture@K, Precision@K."""
    y = np.asarray(y_true)
    s = np.asarray(scores)
    n = len(y)
    n_events = y.sum()
    order = np.argsort(-s)
    y_sorted = y[order]

    rows = []
    for k_pct in k_list:
        k = int(np.ceil(n * k_pct))
        events_in_top = int(y_sorted[:k].sum())
        capture = events_in_top / n_events if n_events > 0 else 0
        precision = events_in_top / k if k > 0 else 0
        lift = capture / k_pct if k_pct > 0 else 0
        rows.append({
            "k_pct": k_pct,
            "k": k,
            "capture": capture,
            "precision": precision,
            "lift": lift,
            "events_in_top": events_in_top
        })
    return pd.DataFrame(rows)


def confusion_at_k(y_true: np.ndarray, scores: np.ndarray, k_pct: float = 0.10) -> Dict:
    """Matrice de confusion pour Top-K."""
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

    return {
        "k_pct": k_pct,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "fn_rate": fn / (tp + fn) if (tp + fn) > 0 else 0,
        "fp_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
    }


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
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


def segment_capture(y_true: np.ndarray, scores: np.ndarray,
                    groups: np.ndarray, k_pct: float = 0.10) -> pd.DataFrame:
    """Capture@K par segment."""
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
        events = int(y[mask].sum())
        if ng < 30 or events == 0:
            continue
        events_cap = int((y[mask] & top[mask]).sum())
        capture = events_cap / events
        rows.append({
            "segment": g,
            "n": ng,
            "events": events,
            "event_rate": events / ng,
            "captured": events_cap,
            "capture": capture,
            "fn_rate": 1 - capture,
            "avg_score": float(s[mask].mean()),
        })
    return pd.DataFrame(rows)


def reliability_diagram_data(y_true: np.ndarray, y_prob: np.ndarray,
                              n_bins: int = 10) -> pd.DataFrame:
    """Donnees pour le diagramme de fiabilite."""
    df = pd.DataFrame({"y": y_true, "prob": y_prob})
    df["bin"] = pd.qcut(df["prob"], q=n_bins, duplicates="drop")
    return df.groupby("bin").agg(
        avg_prob=("prob", "mean"),
        event_rate=("y", "mean"),
        n=("y", "size")
    ).reset_index()


# =============================================================================
# DATA LOADING & PREPARATION
# =============================================================================

def load_dataset(horizon: int) -> pd.DataFrame:
    """Charge le dataset freeze+horizon."""
    pkl_path = ARTIFACTS_DIR / f"dataset_freeze_h{horizon}.pkl"
    csv_path = ARTIFACTS_DIR / f"dataset_freeze_h{horizon}.csv"

    try:
        df = pd.read_pickle(pkl_path)
    except Exception as e:
        print(f"[WARN] Pickle failed ({e}), loading CSV...")
        df = pd.read_csv(csv_path)
        # Restore datetime columns
        for col in ['DDP_parsed', 'DHS_parsed', 'freeze_date', 'date_last_fuite']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

    print(f"Dataset h{horizon}: {len(df):,} rows, {df['event'].sum():,} events "
          f"({100*df['event'].mean():.2f}%)")
    return df


def encode_categoricals(df: pd.DataFrame,
                        cat_cols: Tuple[str, ...] = ("materiau", "decade_install", "diametre_bin")
                        ) -> Tuple[pd.DataFrame, Dict]:
    """Encode les variables categorielles."""
    df = df.copy()
    encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def add_bias_correction_features(df: pd.DataFrame, age_cap: int = 110) -> pd.DataFrame:
    """Ajoute les features de correction de biais (ages extremes)."""
    df = df.copy()

    # Capping
    df["age_cap"] = df["age_at_freeze"].clip(upper=age_cap)
    df["age_extreme_flag"] = (df["age_at_freeze"] > age_cap).astype(int)

    # Winsorisation P99
    p99 = df["age_at_freeze"].quantile(0.99)
    df["age_winsor"] = df["age_at_freeze"].clip(upper=p99)

    # Ratio par decennie
    dec_med = df.groupby("decade_install")["age_at_freeze"].median()
    df["decade_median_age"] = df["decade_install"].map(dec_med)
    df["age_ratio_decade"] = df["age_at_freeze"] / df["decade_median_age"].replace(0, np.nan)
    df["age_ratio_decade"] = df["age_ratio_decade"].fillna(1.0)

    # Residuel par materiau
    mat_med = df.groupby("materiau")["age_at_freeze"].median()
    df["mat_median_age"] = df["materiau"].map(mat_med)
    df["age_residual_mat"] = df["age_at_freeze"] - df["mat_median_age"].fillna(
        df["age_at_freeze"].median())

    return df


def flag_preventive_abandons(df: pd.DataFrame,
                              ratio_thresh: float = 0.6,
                              min_age: int = 15) -> pd.DataFrame:
    """
    Detecte les abandons preventifs (heuristique metier).

    Definition: event=1 MAIS
    - Zero anomalie historique
    - Jeune par rapport au materiau (ratio < seuil)
    - Age absolu < min_age
    - Pas de fuite recente
    """
    df = df.copy()
    cond = (
        (df["event"] == 1) &
        (df["n_fuites_total"] == 0) &
        (df["ratio_age_median"] < ratio_thresh) &
        (df["age_at_freeze"] < min_age) &
        (df.get("has_recent_fuite", 0) == 0)
    )
    df["is_preventive_abandon"] = cond.astype(int)

    # Labels modifies
    df["event_clean"] = df["event"].copy()
    df.loc[df["is_preventive_abandon"] == 1, "event_clean"] = 0

    # Multi-class pour competing risks
    df["event_class"] = 0
    df.loc[(df["event"] == 1) & (df["is_preventive_abandon"] == 0), "event_class"] = 1
    df.loc[df["is_preventive_abandon"] == 1, "event_class"] = 2

    n_prev = cond.sum()
    n_events = df["event"].sum()
    print(f"  Abandons preventifs: {n_prev} / {n_events} ({100*n_prev/max(n_events,1):.1f}%)")

    return df


def get_Xy(df: pd.DataFrame, feature_cols: List[str],
           target_col: str = "event") -> Tuple[pd.DataFrame, np.ndarray]:
    """Extrait X et y."""
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).copy()
    y = df[target_col].astype(int).values
    return X, y


# =============================================================================
# TRAINING
# =============================================================================

def train_lgbm(X_train: pd.DataFrame, y_train: np.ndarray,
               monotone: Optional[List[int]] = None,
               params_override: Optional[Dict] = None) -> lgb.LGBMClassifier:
    """Entraine un LightGBM."""
    n_neg = (y_train == 0).sum()
    n_pos = max((y_train == 1).sum(), 1)

    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "verbose": -1,
        "random_state": RANDOM_SEED,
        "scale_pos_weight": n_neg / n_pos,
    }

    if monotone is not None:
        params["monotone_constraints"] = monotone
    if params_override:
        params.update(params_override)

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model


def train_competing_risks(X_train: pd.DataFrame, y_train: np.ndarray,
                          params_override: Optional[Dict] = None) -> lgb.LGBMClassifier:
    """Entraine un modele multi-class (competing risks)."""
    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": RANDOM_SEED,
        "objective": "multiclass",
        "num_class": 3,
        "verbose": -1,
    }
    if params_override:
        params.update(params_override)

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model


def build_selective_monotone(feature_cols: List[str]) -> List[int]:
    """
    Contraintes monotones selectives.

    +1: features qui doivent augmenter le risque quand elles augmentent
    -1: features qui doivent diminuer le risque quand elles augmentent
    0: pas de contrainte
    """
    mono_pos = {
        "age_at_freeze", "age_cap", "age_winsor",
        "ratio_age_median", "overdue_years", "over_p75_life", "over_p90_life",
        "n_fuites_total", "n_fuites_1y", "n_fuites_3y", "n_fuites_5y",
        "leak_rate_per_year", "leak_rate_per_km",
        "age_x_nfuites", "age_x_ratio", "age_ratio_decade",
        "age_residual_mat", "age_extreme_flag", "has_recent_fuite",
    }
    mono_neg = {"days_since_last_fuite"}

    mono = []
    for col in feature_cols:
        if col in mono_pos:
            mono.append(1)
        elif col in mono_neg:
            mono.append(-1)
        else:
            mono.append(0)
    return mono


# =============================================================================
# CALIBRATION
# =============================================================================

def calibrate_isotonic(train_scores: np.ndarray, train_labels: np.ndarray,
                       test_scores: np.ndarray) -> Tuple[np.ndarray, IsotonicRegression]:
    """Calibration isotonique."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(train_scores, train_labels)
    return iso.transform(test_scores), iso


def calibrate_platt(train_scores: np.ndarray, train_labels: np.ndarray,
                    test_scores: np.ndarray) -> Tuple[np.ndarray, LogisticRegression]:
    """Calibration Platt (logistic scaling)."""
    platt = LogisticRegression(max_iter=1000)
    platt.fit(train_scores.reshape(-1, 1), train_labels)
    return platt.predict_proba(test_scores.reshape(-1, 1))[:, 1], platt


# =============================================================================
# EVALUATION
# =============================================================================

def full_evaluation(model: Any, X_test: pd.DataFrame, y_test: np.ndarray,
                    df_test: pd.DataFrame, label: str,
                    scores_override: Optional[np.ndarray] = None) -> Dict:
    """Evaluation complete d'un modele."""
    if scores_override is not None:
        scores = scores_override
    else:
        proba = model.predict_proba(X_test)
        scores = proba[:, 1] if proba.shape[1] == 2 else proba[:, 1]

    # Metriques top-K
    tk = topk_metrics(y_test, scores)
    conf = confusion_at_k(y_test, scores, 0.10)

    # Metriques globales
    try:
        roc = roc_auc_score(y_test, scores)
    except:
        roc = 0.5
    brier = brier_score_loss(y_test, scores)
    ec = ece_score(y_test, scores)

    # Extraire les valeurs
    lift_5 = float(tk.loc[tk["k_pct"] == 0.05, "lift"].values[0])
    lift_10 = float(tk.loc[tk["k_pct"] == 0.10, "lift"].values[0])
    lift_20 = float(tk.loc[tk["k_pct"] == 0.20, "lift"].values[0])
    cap_5 = float(tk.loc[tk["k_pct"] == 0.05, "capture"].values[0])
    cap_10 = float(tk.loc[tk["k_pct"] == 0.10, "capture"].values[0])
    cap_20 = float(tk.loc[tk["k_pct"] == 0.20, "capture"].values[0])

    # Segments
    seg_mat = segment_capture(y_test, scores, df_test["materiau"].values)
    seg_dec = segment_capture(y_test, scores, df_test["decade_install"].values)

    print(f"  [{label}] ROC={roc:.4f} L@5%={lift_5:.2f} L@10%={lift_10:.2f} "
          f"L@20%={lift_20:.2f} C@10%={cap_10:.2%} Brier={brier:.4f} "
          f"ECE={ec:.4f} FN={conf['FN']}")

    return {
        "label": label,
        "roc_auc": roc,
        "brier": brier,
        "ece": ec,
        "lift_5": lift_5,
        "lift_10": lift_10,
        "lift_20": lift_20,
        "cap_5": cap_5,
        "cap_10": cap_10,
        "cap_20": cap_20,
        "conf": conf,
        "scores": scores,
        "topk": tk,
        "seg_materiau": seg_mat,
        "seg_decade": seg_dec,
        "model": model,
    }


def cv_evaluation(df: pd.DataFrame, feature_cols: List[str], target_col: str,
                  label: str, monotone: Optional[List[int]] = None,
                  params_override: Optional[Dict] = None,
                  n_folds: int = 5) -> Dict:
    """Cross-validation stratifiee."""
    X, y = get_Xy(df, feature_cols, target_col)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    lifts, caps, rocs, briers = [], [], [], []
    for tr_i, te_i in skf.split(X, y):
        m = train_lgbm(X.iloc[tr_i], y[tr_i], monotone=monotone,
                       params_override=params_override)
        sc = m.predict_proba(X.iloc[te_i])[:, 1]
        tk = topk_metrics(y[te_i], sc)
        lifts.append(float(tk.loc[tk["k_pct"] == 0.10, "lift"].values[0]))
        caps.append(float(tk.loc[tk["k_pct"] == 0.10, "capture"].values[0]))
        rocs.append(roc_auc_score(y[te_i], sc))
        briers.append(brier_score_loss(y[te_i], sc))

    return {
        "label": label,
        "lift10_mean": np.mean(lifts),
        "lift10_std": np.std(lifts),
        "cap10_mean": np.mean(caps),
        "cap10_std": np.std(caps),
        "roc_mean": np.mean(rocs),
        "roc_std": np.std(rocs),
        "brier_mean": np.mean(briers),
        "brier_std": np.std(briers),
    }


# =============================================================================
# PLOTS
# =============================================================================

def plot_comparison_bar(results: List[Dict], metric: str, title: str,
                        fname: str, output_dir: Path) -> None:
    """Bar chart comparatif."""
    labels = [r["label"] for r in results]
    values = [r[metric] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    bars = ax.bar(labels, values, color=colors)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_title(title)
    ax.set_ylabel(metric)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / fname, dpi=150)
    plt.close()


def plot_calibration_curves(y_true: np.ndarray, scores_dict: Dict[str, np.ndarray],
                            fname: str, output_dir: Path) -> None:
    """Courbes de calibration."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", label="Parfait")

    for name, scores in scores_dict.items():
        prob_true, prob_pred = calibration_curve(y_true, scores, n_bins=10, strategy="quantile")
        ax.plot(prob_pred, prob_true, "o-", label=name)

    ax.set_xlabel("Probabilite predite")
    ax.set_ylabel("Fraction positive observee")
    ax.set_title("Courbes de calibration")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / fname, dpi=150)
    plt.close()


def plot_lift_curves(y_true: np.ndarray, scores_dict: Dict[str, np.ndarray],
                     fname: str, output_dir: Path) -> None:
    """Courbes de lift."""
    fig, ax = plt.subplots(figsize=(9, 6))
    pcts = np.arange(1, 51)

    for name, scores in scores_dict.items():
        y = np.asarray(y_true)
        s = np.asarray(scores)
        n = len(y)
        n_ev = y.sum()
        order = np.argsort(-s)
        y_s = y[order]
        lifts = []
        for p in pcts:
            k = int(np.ceil(n * p / 100))
            cap = y_s[:k].sum() / n_ev if n_ev else 0
            lifts.append(cap / (p / 100))
        ax.plot(pcts, lifts, label=name, linewidth=2)

    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.set_xlabel("Top K%")
    ax.set_ylabel("Lift")
    ax.set_title("Courbes de Lift")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / fname, dpi=150)
    plt.close()


def plot_error_distributions(y_true: np.ndarray, scores: np.ndarray,
                              df_test: pd.DataFrame, fname: str,
                              output_dir: Path) -> None:
    """Distribution des erreurs par segment."""
    n = len(y_true)
    k = int(np.ceil(n * 0.10))
    order = np.argsort(-scores)
    top = np.zeros(n, dtype=bool)
    top[order[:k]] = True

    tp = (y_true == 1) & top
    fp = (y_true == 0) & top
    fn = (y_true == 1) & ~top
    tn = (y_true == 0) & ~top

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Age distribution by error type
    ax = axes[0, 0]
    for label, mask, color in [("TP", tp, "green"), ("FP", fp, "orange"),
                                ("FN", fn, "red"), ("TN", tn, "blue")]:
        if mask.any():
            ax.hist(df_test.loc[mask, "age_at_freeze"], bins=30, alpha=0.5,
                    label=f"{label} (n={mask.sum()})", color=color, density=True)
    ax.set_xlabel("Age at freeze")
    ax.set_title("Distribution par age")
    ax.legend()

    # Material distribution
    ax = axes[0, 1]
    mat_data = []
    for mat in df_test["materiau"].unique():
        mask_mat = df_test["materiau"] == mat
        if mask_mat.sum() < 10:
            continue
        mat_data.append({
            "mat": mat,
            "TP": (tp & mask_mat).sum(),
            "FP": (fp & mask_mat).sum(),
            "FN": (fn & mask_mat).sum(),
        })
    if mat_data:
        mat_df = pd.DataFrame(mat_data).set_index("mat")
        mat_df[["TP", "FP", "FN"]].plot(kind="bar", ax=ax, color=["green", "orange", "red"])
        ax.set_title("Erreurs par materiau")
        ax.tick_params(axis='x', rotation=45)

    # Score distribution by error type
    ax = axes[1, 0]
    for label, mask, color in [("TP", tp, "green"), ("FP", fp, "orange"),
                                ("FN", fn, "red")]:
        if mask.any():
            ax.hist(scores[mask], bins=30, alpha=0.5, label=label,
                    color=color, density=True)
    ax.set_xlabel("Score")
    ax.set_title("Distribution des scores par type d'erreur")
    ax.legend()

    # Decade distribution
    ax = axes[1, 1]
    dec_data = []
    for dec in sorted(df_test["decade_install"].unique()):
        mask_dec = df_test["decade_install"] == dec
        if mask_dec.sum() < 10:
            continue
        dec_data.append({
            "decade": str(int(dec)),
            "FN_rate": (fn & mask_dec).sum() / max((y_true[mask_dec] == 1).sum(), 1),
            "FP_rate": (fp & mask_dec).sum() / max((y_true[mask_dec] == 0).sum(), 1),
        })
    if dec_data:
        dec_df = pd.DataFrame(dec_data).set_index("decade")
        dec_df.plot(kind="bar", ax=ax)
        ax.set_title("Taux FN/FP par decennie")
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / fname, dpi=150)
    plt.close()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_baseline(horizon: int, output_dir: Path) -> Dict:
    """PHASE B: Reproduction baseline avec metriques completes."""
    print("\n" + "="*70)
    print(f"PHASE B: REPRODUCTION BASELINE (H={horizon})")
    print("="*70)

    # Load data
    df = load_dataset(horizon)
    df, encoders = encode_categoricals(df)

    # Get freeze date from data
    freeze_date = pd.Timestamp(df['freeze_date'].iloc[0])
    print(f"Freeze date: {freeze_date}")

    # Run leakage tests
    run_all_leakage_tests(df, freeze_date, horizon, blocking=LEAKAGE_TESTS_BLOCKING)

    # Prepare features
    feature_cols = [f for f in BASE_FEATURES if f in df.columns]
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")

    # Split
    X, y = get_Xy(df, feature_cols, "event")
    idx = np.arange(len(df))
    X_tr, X_te, y_tr, y_te, i_tr, i_te = train_test_split(
        X, y, idx, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    df_te = df.iloc[i_te].copy()

    print(f"\nTrain: {len(X_tr):,} | Test: {len(X_te):,}")
    print(f"Events train: {y_tr.sum()} | Events test: {y_te.sum()}")

    # Train baseline
    print("\n--- Training M0 Baseline ---")
    model = train_lgbm(X_tr, y_tr)
    result = full_evaluation(model, X_te, y_te, df_te, "M0_baseline")

    # CV for robustness
    print("\n--- Cross-validation ---")
    cv = cv_evaluation(df, feature_cols, "event", "M0_cv")
    print(f"  [CV] Lift@10%={cv['lift10_mean']:.2f}+/-{cv['lift10_std']:.2f}")

    # Save baseline result
    result["cv"] = cv
    result["feature_cols"] = feature_cols
    result["encoders"] = encoders
    result["df_test"] = df_te
    result["y_test"] = y_te
    result["X_test"] = X_te
    result["train_indices"] = i_tr
    result["test_indices"] = i_te
    result["df_full"] = df

    return result


def run_iterations(baseline: Dict, horizon: int, output_dir: Path) -> List[Dict]:
    """PHASE D: Iterations controlees."""
    print("\n" + "="*70)
    print("PHASE D: ITERATIONS CONTROLEES")
    print("="*70)

    df = baseline["df_full"]
    encoders = baseline["encoders"]
    i_tr = baseline["train_indices"]
    i_te = baseline["test_indices"]
    y_te_orig = baseline["y_test"]
    df_te = baseline["df_test"]

    results = [baseline]  # M0 is first

    # Add bias correction features
    print("\n--- Preparation des features etendues ---")
    df = add_bias_correction_features(df)
    df = flag_preventive_abandons(df)

    df_tr = df.iloc[i_tr].copy()
    df_te = df.iloc[i_te].copy()

    def get_X_ext(data, cols):
        return data[cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Extended features
    feat_ext = [f for f in EXTENDED_FEATURES if f in df.columns]
    X_tr_ext = get_X_ext(df_tr, feat_ext)
    X_te_ext = get_X_ext(df_te, feat_ext)

    # =========================================================================
    # ITERATION 1: M1 - Competing Risks
    # =========================================================================
    print("\n" + "-"*50)
    print("ITERATION 1: M1 - Competing Risks")
    print("-"*50)

    y_tr_mc = df_tr["event_class"].values
    model_m1 = train_competing_risks(X_tr_ext, y_tr_mc)
    scores_m1 = model_m1.predict_proba(X_te_ext)[:, 1]  # P(real failure)
    result_m1 = full_evaluation(model_m1, X_te_ext, y_te_orig, df_te,
                                 "M1_competing", scores_override=scores_m1)
    results.append(result_m1)

    # Decision GO/NO-GO
    if result_m1["lift_10"] < baseline["lift_10"]:
        print(f"  [WARN] M1 degrade Lift@10%: {result_m1['lift_10']:.2f} < {baseline['lift_10']:.2f}")
    else:
        print(f"  [OK] M1 ameliore Lift@10%: +{100*(result_m1['lift_10']/baseline['lift_10']-1):.1f}%")

    # =========================================================================
    # ITERATION 2: M2 - Cleaned Labels + Selective Monotone
    # =========================================================================
    print("\n" + "-"*50)
    print("ITERATION 2: M2 - Cleaned Labels + Selective Monotone")
    print("-"*50)

    y_tr_clean = df_tr["event_clean"].values
    mono_sel = build_selective_monotone(feat_ext)

    model_m2 = train_lgbm(X_tr_ext, y_tr_clean, monotone=mono_sel)
    result_m2 = full_evaluation(model_m2, X_te_ext, y_te_orig, df_te, "M2_mono_clean")
    results.append(result_m2)

    # =========================================================================
    # ITERATION 3: M3 - M2 + Regularisation
    # =========================================================================
    print("\n" + "-"*50)
    print("ITERATION 3: M3 - M2 + Regularisation")
    print("-"*50)

    params_reg = {
        "n_estimators": 250,
        "learning_rate": 0.04,
        "max_depth": 5,
        "num_leaves": 25,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 2.0,
        "reg_alpha": 0.3,
        "min_child_samples": 50,
    }

    model_m3 = train_lgbm(X_tr_ext, y_tr_clean, monotone=mono_sel, params_override=params_reg)
    result_m3 = full_evaluation(model_m3, X_te_ext, y_te_orig, df_te, "M3_mono_reg")
    results.append(result_m3)

    # =========================================================================
    # ITERATION 4: M4 - M1 Competing + Isotonic Calibration
    # =========================================================================
    print("\n" + "-"*50)
    print("ITERATION 4: M4 - M1 Competing + Isotonic Calibration")
    print("-"*50)

    sc_tr_m1 = model_m1.predict_proba(X_tr_ext)[:, 1]
    y_tr_real = (df_tr["event_class"] == 1).astype(int).values
    scores_m4, iso_m4 = calibrate_isotonic(sc_tr_m1, y_tr_real, scores_m1)
    result_m4 = full_evaluation(model_m1, X_te_ext, y_te_orig, df_te,
                                 "M4_competing_iso", scores_override=scores_m4)
    result_m4["calibrator"] = iso_m4
    results.append(result_m4)

    # =========================================================================
    # ITERATION 5: M5 - M2 + Isotonic Calibration
    # =========================================================================
    print("\n" + "-"*50)
    print("ITERATION 5: M5 - M2 + Isotonic Calibration")
    print("-"*50)

    sc_tr_m2 = model_m2.predict_proba(X_tr_ext)[:, 1]
    sc_te_m2 = result_m2["scores"]
    scores_m5, iso_m5 = calibrate_isotonic(sc_tr_m2, y_tr_clean, sc_te_m2)
    result_m5 = full_evaluation(model_m2, X_te_ext, y_te_orig, df_te,
                                 "M5_mono_iso", scores_override=scores_m5)
    result_m5["calibrator"] = iso_m5
    results.append(result_m5)

    # =========================================================================
    # Store metadata
    # =========================================================================
    for r in results[1:]:  # Skip baseline
        r["feature_cols"] = feat_ext
        r["encoders"] = encoders

    return results


def select_best_model(results: List[Dict], output_dir: Path) -> Dict:
    """PHASE E: Selection du modele final."""
    print("\n" + "="*70)
    print("PHASE E: SELECTION MODELE FINAL")
    print("="*70)

    # Build comparison table
    rows = []
    for r in results:
        rows.append({
            "Model": r["label"],
            "ROC_AUC": r["roc_auc"],
            "Lift@5%": r["lift_5"],
            "Lift@10%": r["lift_10"],
            "Lift@20%": r["lift_20"],
            "Cap@10%": r["cap_10"],
            "Brier": r["brier"],
            "ECE": r["ece"],
            "FN": r["conf"]["FN"],
            "FP": r["conf"]["FP"],
        })

    df_comparison = pd.DataFrame(rows)
    print("\n" + df_comparison.to_string(index=False))

    # Save comparison
    df_comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    # Selection criteria:
    # 1. Primary: Lift@10% (must not be worse than baseline)
    # 2. Secondary: Brier (calibration)
    # 3. Tertiary: FN (safety)

    baseline_lift = results[0]["lift_10"]
    valid_models = [r for r in results if r["lift_10"] >= baseline_lift * 0.98]

    if not valid_models:
        print("\n  [WARN] Aucun modele n'ameliore la baseline, selection baseline")
        best = results[0]
    else:
        # Among valid, pick best (Lift@10%, then -Brier, then -FN)
        best = max(valid_models, key=lambda r: (r["lift_10"], -r["brier"], -r["conf"]["FN"]))

    print(f"\n  >>> MODELE RETENU: {best['label']}")
    print(f"      Lift@10%: {best['lift_10']:.2f} ({100*(best['lift_10']/baseline_lift-1):+.1f}% vs baseline)")
    print(f"      Cap@10%: {best['cap_10']:.2%}")
    print(f"      Brier: {best['brier']:.4f}")
    print(f"      ECE: {best['ece']:.4f}")
    print(f"      FN@10%: {best['conf']['FN']}")

    return best


def generate_artifacts(best: Dict, all_results: List[Dict],
                       horizon: int, output_dir: Path) -> None:
    """Genere les artefacts finaux."""
    print("\n" + "="*70)
    print("GENERATION DES ARTEFACTS")
    print("="*70)

    # 1. Save best model
    artifact = {
        "model": best["model"],
        "calibrator": best.get("calibrator"),
        "encoders": best.get("encoders", {}),
        "metadata": {
            "version": best["label"],
            "horizon_years": horizon,
            "feature_cols": best.get("feature_cols", BASE_FEATURES),
            "created_at": datetime.now().isoformat(),
            "metrics": {
                "roc_auc": best["roc_auc"],
                "lift_10": best["lift_10"],
                "cap_10": best["cap_10"],
                "brier": best["brier"],
                "ece": best["ece"],
                "fn_10": best["conf"]["FN"],
            }
        }
    }

    model_path = ARTIFACTS_DIR / f"best_model_h{horizon}_v2.joblib"
    joblib.dump(artifact, model_path)
    print(f"  Model saved: {model_path}")

    # 2. Save metadata JSON
    metadata = artifact["metadata"].copy()
    metadata["model_path"] = str(model_path)

    meta_path = ARTIFACTS_DIR / f"metadata_h{horizon}_v2.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Metadata saved: {meta_path}")

    # 3. Generate plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Comparison bars
    plot_comparison_bar(all_results, "lift_10", "Lift@10% Comparison",
                        "comparison_lift10.png", plots_dir)
    plot_comparison_bar(all_results, "brier", "Brier Score Comparison",
                        "comparison_brier.png", plots_dir)

    # Calibration curves
    scores_dict = {r["label"]: r["scores"] for r in all_results}
    plot_calibration_curves(best["y_test"] if "y_test" in best else all_results[0]["y_test"],
                            scores_dict, "calibration_curves.png", plots_dir)

    # Lift curves
    plot_lift_curves(all_results[0]["y_test"], scores_dict, "lift_curves.png", plots_dir)

    # Error distributions
    if "df_test" in best and "y_test" in best:
        plot_error_distributions(best["y_test"], best["scores"],
                                  best["df_test"], "error_distributions.png", plots_dir)

    print(f"  Plots saved: {plots_dir}")


def generate_report(best: Dict, all_results: List[Dict],
                    horizon: int, output_dir: Path) -> None:
    """Genere le rapport final."""
    print("\n  Generating report...")

    report = []
    report.append(f"# Rapport Phase 3 - Modele Final AEP (H={horizon})")
    report.append(f"\n*Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    report.append("---\n")

    # Executive summary
    report.append("## Resume Executif\n")
    report.append(f"**Modele retenu**: {best['label']}\n")
    report.append(f"- Lift@10%: {best['lift_10']:.2f}")
    report.append(f"- Capture@10%: {best['cap_10']:.2%}")
    report.append(f"- Brier Score: {best['brier']:.4f}")
    report.append(f"- ECE: {best['ece']:.4f}")
    report.append(f"- FN@10%: {best['conf']['FN']}")
    report.append("")

    baseline = all_results[0]
    improvement = 100 * (best['lift_10'] / baseline['lift_10'] - 1)
    report.append(f"**Amelioration vs baseline**: {improvement:+.1f}% Lift@10%\n")

    # Comparison table
    report.append("## Comparaison des Modeles\n")
    report.append("| Model | ROC-AUC | Lift@10% | Cap@10% | Brier | ECE | FN |")
    report.append("|-------|---------|----------|---------|-------|-----|-----|")
    for r in all_results:
        report.append(f"| {r['label']} | {r['roc_auc']:.4f} | {r['lift_10']:.2f} | "
                      f"{r['cap_10']:.2%} | {r['brier']:.4f} | {r['ece']:.4f} | {r['conf']['FN']} |")
    report.append("")

    # Segment analysis
    report.append("## Stabilite par Segments\n")
    report.append("### Par Materiau\n")
    if "seg_materiau" in best and not best["seg_materiau"].empty:
        seg = best["seg_materiau"][["segment", "n", "events", "capture", "fn_rate"]]
        report.append(seg.to_markdown(index=False))
    report.append("")

    report.append("### Par Decennie\n")
    if "seg_decade" in best and not best["seg_decade"].empty:
        seg = best["seg_decade"][["segment", "n", "events", "capture", "fn_rate"]]
        report.append(seg.to_markdown(index=False))
    report.append("")

    # Recommendations
    report.append("## Recommandations\n")
    report.append("1. **Utilisation du score**:")
    report.append("   - `risque_attendu = proba_calibree * longueur_km`")
    report.append("   - Utiliser pour le moteur d'optimisation sous contraintes budget\n")
    report.append("2. **Monitoring**:")
    report.append("   - Suivre les segments PEHD/PVC (sous-couverts)")
    report.append("   - Recalibrer trimestriellement\n")
    report.append("3. **Limitations**:")
    report.append("   - L'heuristique d'abandon preventif est approximative")
    report.append("   - Facteurs externes (travaux tiers) non captures\n")

    # Artifacts
    report.append("## Artefacts Generes\n")
    report.append(f"- `best_model_h{horizon}_v2.joblib`: Modele final")
    report.append(f"- `metadata_h{horizon}_v2.json`: Metadata")
    report.append("- `model_comparison.csv`: Tableau comparatif")
    report.append("- `plots/`: Graphiques d'analyse")

    # Write report
    report_text = "\n".join(report)
    report_path = output_dir / f"rapport_phase3_h{horizon}.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"  Report saved: {report_path}")


def main(horizon: int = DEFAULT_HORIZON,
         baseline_only: bool = False,
         iterations_only: bool = False) -> Dict:
    """Pipeline principal."""
    print("="*70)
    print("PHASE 3 - PIPELINE CONSOLIDE MAINTENANCE PREDICTIVE AEP")
    print("="*70)
    print(f"Horizon: {horizon} an(s)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Setup output directory
    output_dir = REPORTS_DIR / f"phase3_h{horizon}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase B: Baseline
    baseline = run_baseline(horizon, output_dir)

    if baseline_only:
        print("\n[INFO] Baseline only mode - stopping here")
        return {"baseline": baseline}

    # Phase D: Iterations
    all_results = run_iterations(baseline, horizon, output_dir)

    # Phase E: Selection
    best = select_best_model(all_results, output_dir)

    # Propagate test data to best
    best["y_test"] = baseline["y_test"]
    best["df_test"] = baseline["df_test"]

    # Generate artifacts
    generate_artifacts(best, all_results, horizon, output_dir)

    # Generate report
    generate_report(best, all_results, horizon, output_dir)

    print("\n" + "="*70)
    print("PHASE 3 TERMINEE")
    print(f"Modele final: {best['label']}")
    print(f"Artefacts: {output_dir}")
    print("="*70)

    return {
        "baseline": baseline,
        "all_results": all_results,
        "best": best,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3 Consolidated Pipeline")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                        help="Horizon en annees (1, 3, 5)")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Execute uniquement la baseline")
    parser.add_argument("--iterations-only", action="store_true",
                        help="Execute uniquement les iterations (necessite baseline)")
    args = parser.parse_args()

    main(args.horizon, args.baseline_only, args.iterations_only)
