"""
Phase 1 evaluation script.
Reproduces the existing LightGBM model results for horizon 1 year without
retraining, and generates all metrics, plots and exportable tables required
for the evaluation report.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib

# Non-interactive backend for figure generation
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

from config import ARTIFACTS_DIR, REPORTS_DIR, RANDOM_SEED, TOP_K_PERCENTAGES
from modeling import load_dataset, prepare_features, compute_business_metrics


# --------------------------------------------------------------------------- #
# Paths and general setup
# --------------------------------------------------------------------------- #
OUTPUT_DIR = REPORTS_DIR / "phase1"
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"

for d in (OUTPUT_DIR, PLOTS_DIR, TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")


# --------------------------------------------------------------------------- #
# Data loading and scoring
# --------------------------------------------------------------------------- #
def load_model_and_test_data():
    """Load best model, build the same test split as during training, and score."""
    artifact = joblib.load(ARTIFACTS_DIR / "best_model.joblib")
    model = artifact["model"]
    metadata = artifact["metadata"]
    horizon = metadata["horizon_years"]

    df = load_dataset(horizon)
    X, y, _, feature_cols = prepare_features(df)

    # Rebuild the original 80/20 stratified split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        df.index,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    df_test = df.loc[idx_test].copy()
    df_test.reset_index(drop=True, inplace=True)
    y_test = pd.Series(y_test, name="event").reset_index(drop=True)

    # Predictions
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.predict(X_test)

    scores = pd.Series(scores, name="score")

    return model, metadata, df_test, y_test, scores, feature_cols


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def topk_table(y_true: pd.Series, scores: pd.Series, k_list=None) -> pd.DataFrame:
    """Compute capture, lift, precision for each k in k_list."""
    if k_list is None:
        k_list = TOP_K_PERCENTAGES

    n = len(y_true)
    n_events = y_true.sum()
    order = np.argsort(-scores.values)
    y_sorted = y_true.values[order]

    rows = []
    for k_pct in k_list:
        k = int(np.ceil(n * k_pct))
        events_in_top = y_sorted[:k].sum()
        capture = events_in_top / n_events if n_events > 0 else 0.0
        precision = events_in_top / k if k > 0 else 0.0
        lift = capture / k_pct if k_pct > 0 else 0.0
        rows.append(
            {
                "k_pct": k_pct,
                "k_count": k,
                "events_in_top": int(events_in_top),
                "capture": float(capture),
                "precision": float(precision),
                "lift": float(lift),
            }
        )

    df_res = pd.DataFrame(rows)
    df_res["recall"] = df_res["capture"]  # recall equals capture in top-k logic
    return df_res


def confusion_topk(y_true: pd.Series, scores: pd.Series, k_pct: float = 0.10):
    """Top-k confusion matrix (prioritisation view)."""
    n = len(y_true)
    k = int(np.ceil(n * k_pct))
    order = np.argsort(-scores.values)
    top_mask = np.zeros(n, dtype=bool)
    top_mask[order[:k]] = True

    y_np = y_true.values.astype(int)

    tp = int(((y_np == 1) & top_mask).sum())
    fp = int(((y_np == 0) & top_mask).sum())
    fn = int(((y_np == 1) & (~top_mask)).sum())
    tn = int(((y_np == 0) & (~top_mask)).sum())

    return {
        "k_pct": k_pct,
        "k_count": k,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "fn_rate": fn / (tp + fn) if (tp + fn) > 0 else 0.0,
        "fp_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
    }


def decile_stats(y_true: pd.Series, scores: pd.Series) -> pd.DataFrame:
    """Event rate by decile (highest score = decile 1)."""
    df_tmp = pd.DataFrame({"score": scores, "event": y_true})
    df_tmp = df_tmp.sort_values("score", ascending=False).reset_index(drop=True)
    df_tmp["decile"] = (np.floor(np.arange(len(df_tmp)) / (len(df_tmp) / 10))).astype(
        int
    )
    df_tmp["decile"] = df_tmp["decile"].clip(upper=9)

    stats = (
        df_tmp.groupby("decile")
        .agg(
            n=("event", "size"),
            events=("event", "sum"),
            event_rate=("event", "mean"),
            score_min=("score", "min"),
            score_max=("score", "max"),
        )
        .reset_index()
    )
    stats["population_pct"] = stats["n"] / len(df_tmp)
    stats["capture_cum"] = stats["events"].cumsum() / df_tmp["event"].sum()
    stats["decile_label"] = 10 - stats["decile"]  # 10 = highest scores
    stats = stats.sort_values("decile_label")
    return stats[
        [
            "decile_label",
            "n",
            "population_pct",
            "events",
            "event_rate",
            "capture_cum",
            "score_min",
            "score_max",
        ]
    ]


def segment_capture(df_eval: pd.DataFrame, segment_col: str, top_mask: np.ndarray):
    """Capture@10 and FN rate by segment."""
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


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def plot_gain_curve(y_true: pd.Series, scores: pd.Series, path: Path):
    """Cumulative gain (Cap curve)."""
    n_events = y_true.sum()
    order = np.argsort(-scores.values)
    y_sorted = y_true.values[order]
    cum_events = np.cumsum(y_sorted)
    population_pct = np.arange(1, len(y_true) + 1) / len(y_true)
    gain = cum_events / n_events

    plt.figure(figsize=(8, 6))
    plt.plot(population_pct * 100, gain * 100, label="Modele", color="#e74c3c")
    plt.plot([0, 100], [0, 100], "--", color="gray", label="Aleatoire")
    plt.xlabel("Part de population triee par score (%)")
    plt.ylabel("Part des defaillances capturees (%)")
    plt.title("Courbe de gain cumule (Cap@K)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_lift_curve(y_true: pd.Series, scores: pd.Series, path: Path):
    """Lift curve by percentile."""
    n_events = y_true.sum()
    order = np.argsort(-scores.values)
    y_sorted = y_true.values[order]

    percentiles = np.linspace(0.01, 1.0, 100)
    lift_values = []
    for p in percentiles:
        k = int(np.ceil(len(y_true) * p))
        capture = y_sorted[:k].sum() / n_events
        lift_values.append(capture / p)

    plt.figure(figsize=(8, 6))
    plt.plot(percentiles * 100, np.array(lift_values), color="#3498db")
    plt.axhline(1.0, linestyle="--", color="gray", label="Aleatoire")
    plt.xlabel("Part de population (%)")
    plt.ylabel("Lift cumule")
    plt.title("Courbe de lift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_score_distribution(scores: pd.Series, path: Path):
    plt.figure(figsize=(8, 6))
    plt.hist(scores, bins=40, color="#34495e", edgecolor="black")
    plt.xlabel("Score de risque")
    plt.ylabel("Frequence")
    plt.title("Distribution globale des scores")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_error_distributions(df_eval: pd.DataFrame, top_mask: np.ndarray, path: Path):
    """Score distributions for TP/FN and FP/TN within the top-k setup."""
    df_plot = df_eval.copy()
    df_plot["topk"] = top_mask
    df_plot["prediction_type"] = "TN"
    df_plot.loc[(df_plot["event"] == 1) & df_plot["topk"], "prediction_type"] = "TP"
    df_plot.loc[(df_plot["event"] == 1) & (~df_plot["topk"]), "prediction_type"] = "FN"
    df_plot.loc[(df_plot["event"] == 0) & df_plot["topk"], "prediction_type"] = "FP"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # TP vs FN (only events)
    subset_pos = df_plot[df_plot["event"] == 1]
    axes[0].hist(
        subset_pos[subset_pos["prediction_type"] == "TP"]["score"],
        bins=30,
        alpha=0.7,
        label="TP (capture)",
        color="#2ecc71",
        edgecolor="black",
    )
    axes[0].hist(
        subset_pos[subset_pos["prediction_type"] == "FN"]["score"],
        bins=30,
        alpha=0.7,
        label="FN (rates)",
        color="#e67e22",
        edgecolor="black",
    )
    axes[0].set_title("Scores des evenements: captures (TP) vs rates (FN)")
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Frequence")
    axes[0].legend()

    # FP vs TN (non-events)
    subset_neg = df_plot[df_plot["event"] == 0]
    axes[1].hist(
        subset_neg[subset_neg["prediction_type"] == "FP"]["score"],
        bins=30,
        alpha=0.7,
        label="FP (sur-priorises)",
        color="#e74c3c",
        edgecolor="black",
    )
    axes[1].hist(
        subset_neg[subset_neg["prediction_type"] == "TN"]["score"],
        bins=30,
        alpha=0.7,
        label="TN (non selectionnes)",
        color="#3498db",
        edgecolor="black",
    )
    axes[1].set_title("Scores des non-evenements: FP vs TN")
    axes[1].set_xlabel("Score")
    axes[1].set_ylabel("Frequence")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_residuals(y_true: pd.Series, scores: pd.Series, path: Path):
    residuals = y_true - scores
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=40, color="#8e44ad", edgecolor="black")
    plt.xlabel("Residual (y - score)")
    plt.ylabel("Frequence")
    plt.title("Distribution des residus")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_decile_curve(deciles: pd.DataFrame, path: Path):
    plt.figure(figsize=(8, 6))
    plt.plot(
        deciles["decile_label"],
        deciles["event_rate"] * 100,
        marker="o",
        color="#d35400",
    )
    plt.gca().invert_xaxis()  # decile 10 = top scores on the left
    plt.xlabel("Decile (10 = scores les plus eleves)")
    plt.ylabel("Taux de defaillance (%)")
    plt.title("Taux de defaillance reel par decile de score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_bar_metric(df: pd.DataFrame, value_col: str, label_col: str, title: str, path: Path):
    df_top = df.head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_top, x=value_col, y=label_col, palette="viridis")
    plt.xlabel(value_col)
    plt.ylabel(label_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
def compute_and_plot_importance(model, feature_cols, path: Path):
    if not hasattr(model, "feature_importances_"):
        return None
    imp = model.feature_importances_
    df_imp = (
        pd.DataFrame({"feature": feature_cols, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    top = df_imp.head(20)
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top, x="importance", y="feature", palette="mako")
    plt.title("Importances relatives (LightGBM)")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return df_imp


def compute_and_plot_shap(model, X_sample: pd.DataFrame, path: Path):
    try:
        import shap
    except ImportError:
        return None

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        # For binary classification shap_values is a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap.summary_plot(
            shap_values,
            X_sample,
            show=False,
            plot_type="dot",
            max_display=20,
        )
        plt.title("SHAP summary (echantillon)")
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return shap_values
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Main execution
# --------------------------------------------------------------------------- #
def main():
    model, metadata, df_test, y_test, scores, feature_cols = load_model_and_test_data()

    df_eval = df_test.copy()
    df_eval["event"] = y_test
    df_eval["score"] = scores

    # Metrics
    roc = roc_auc_score(y_test, scores) if y_test.sum() > 0 else 0.0
    topk_metrics = topk_table(y_test, scores)
    business_metrics = compute_business_metrics(
        y_test.values, scores.values, TOP_K_PERCENTAGES
    )
    confusion_k10 = confusion_topk(y_test, scores, k_pct=0.10)

    # Top-k mask for k=10% (main focus)
    n = len(y_test)
    k10 = int(np.ceil(n * 0.10))
    order = np.argsort(-scores.values)
    top_mask_10 = np.zeros(n, dtype=bool)
    top_mask_10[order[:k10]] = True

    # Deciles
    deciles = decile_stats(y_test, scores)

    # Segments
    segment_results = {}
    segment_results["materiau"] = segment_capture(df_eval, "materiau", top_mask_10)
    if "decade_install" in df_eval.columns:
        segment_results["decade_install"] = segment_capture(
            df_eval, "decade_install", top_mask_10
        )
    if "diametre_bin" in df_eval.columns:
        segment_results["diametre_bin"] = segment_capture(
            df_eval, "diametre_bin", top_mask_10
        )
    # Length classes (create if absent)
    if "longueur" in df_eval.columns:
        bins = [0, 50, 150, 300, 600, np.inf]
        labels = ["<50m", "50-150m", "150-300m", "300-600m", "600m+"]
        df_eval["longueur_bin"] = pd.cut(
            df_eval["longueur"], bins=bins, labels=labels, include_lowest=True
        )
        segment_results["longueur_bin"] = segment_capture(
            df_eval, "longueur_bin", top_mask_10
        )

    # Explainability
    df_importance = compute_and_plot_importance(
        model, feature_cols, PLOTS_DIR / "feature_importance.png"
    )

    # Plots (fast)
    plot_gain_curve(y_test, scores, PLOTS_DIR / "gain_curve.png")
    plot_lift_curve(y_test, scores, PLOTS_DIR / "lift_curve.png")
    plot_score_distribution(scores, PLOTS_DIR / "score_distribution.png")
    plot_error_distributions(df_eval, top_mask_10, PLOTS_DIR / "score_errors.png")
    plot_residuals(y_test, scores, PLOTS_DIR / "residuals.png")
    plot_decile_curve(deciles, PLOTS_DIR / "decile_event_rate.png")

    # Segment barplots (top 10 categories)
    if not segment_results["materiau"].empty:
        plot_bar_metric(
            segment_results["materiau"],
            "capture_10pct",
            "materiau",
            "Capture@10% par materiau",
            PLOTS_DIR / "segment_capture_materiau.png",
        )
    if "decade_install" in segment_results and not segment_results["decade_install"].empty:
        plot_bar_metric(
            segment_results["decade_install"],
            "capture_10pct",
            "decade_install",
            "Capture@10% par decennie de pose",
            PLOTS_DIR / "segment_capture_decade.png",
        )
    if "diametre_bin" in segment_results and not segment_results["diametre_bin"].empty:
        plot_bar_metric(
            segment_results["diametre_bin"],
            "capture_10pct",
            "diametre_bin",
            "Capture@10% par classe de diametre",
            PLOTS_DIR / "segment_capture_diametre.png",
        )
    if "longueur_bin" in segment_results and not segment_results["longueur_bin"].empty:
        plot_bar_metric(
            segment_results["longueur_bin"],
            "capture_10pct",
            "longueur_bin",
            "Capture@10% par classe de longueur",
            PLOTS_DIR / "segment_capture_longueur.png",
        )

    # Save tables
    topk_metrics.to_csv(TABLES_DIR / "topk_metrics.csv", index=False)
    deciles.to_csv(TABLES_DIR / "decile_stats.csv", index=False)
    pd.DataFrame([confusion_k10]).to_csv(
        TABLES_DIR / "confusion_top10.csv", index=False
    )
    if df_importance is not None:
        df_importance.to_csv(TABLES_DIR / "feature_importance.csv", index=False)
    for name, df_seg in segment_results.items():
        if df_seg is not None and not df_seg.empty:
            df_seg.to_csv(TABLES_DIR / f"segment_{name}.csv", index=False)

    # Summary JSON for easy reuse
    summary = {
        "horizon_years": metadata.get("horizon_years"),
        "freeze_date": metadata.get("freeze_date"),
        "n_test": int(len(y_test)),
        "event_rate_test": float(y_test.mean()),
        "roc_auc": float(roc),
        "topk": topk_metrics.to_dict(orient="records"),
        "confusion_top10": confusion_k10,
    }
    with open(TABLES_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # SHAP on a small sample (after critical outputs are saved)
    shap_values = None
    if len(df_eval) > 0:
        sample_size = min(500, len(df_eval))
        X_sample = df_eval[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        X_sample = X_sample.sample(sample_size, random_state=RANDOM_SEED)
        shap_values = compute_and_plot_shap(
            model, X_sample, PLOTS_DIR / "shap_summary.png"
        )

    print("=== Phase 1 evaluation complete ===")
    print(f"Test set: {len(y_test):,} rows | event rate: {y_test.mean()*100:.2f}%")
    print(f"ROC-AUC: {roc:.3f}")
    print("Top-k metrics:")
    print(topk_metrics[["k_pct", "capture", "precision", "lift"]].to_string(index=False))
    print("Artifacts written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
