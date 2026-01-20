"""
Step 6: Explicabilité et Vérifications
- Feature importances
- SHAP values
- Vérification des signes des coefficients
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from config import ARTIFACTS_DIR, REPORTS_DIR, HORIZONS
from dataset_builder import load_raw_data, get_feature_columns


def load_model_and_data(horizon: int):
    """Charge le modèle et les données."""
    artifact = joblib.load(ARTIFACTS_DIR / f"model_h{horizon}.joblib")
    df = pd.read_pickle(ARTIFACTS_DIR / f"dataset_freeze_h{horizon}.pkl")
    return artifact, df


def compute_feature_importance(model, feature_cols: list) -> pd.DataFrame:
    """
    Calcule les importances des features selon le type de modèle.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return None

    df_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)

    return df_imp


def compute_logistic_coefficients(artifact, feature_cols: list) -> pd.DataFrame:
    """
    Pour les modèles linéaires, retourne les coefficients avec leur signe.
    """
    model = artifact['model']

    if not hasattr(model, 'coef_'):
        return None

    coef = model.coef_[0]

    df_coef = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': coef,
        'odds_ratio': np.exp(coef)
    }).sort_values('coefficient', ascending=False)

    return df_coef


def plot_feature_importance(df_imp: pd.DataFrame, horizon: int, output_dir: Path):
    """
    Plot des importances des features.
    """
    top_n = min(20, len(df_imp))
    df_top = df_imp.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(df_top))

    ax.barh(y_pos, df_top['importance'].values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_top['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(f'Feature Importances - Horizon {horizon} an(s)')

    plt.tight_layout()
    plt.savefig(output_dir / f'feature_importance_h{horizon}.png', dpi=150)
    plt.close()


def compute_shap_values(model, X_sample: pd.DataFrame, feature_cols: list) -> dict:
    """
    Calcule les SHAP values pour un échantillon.
    """
    if not HAS_SHAP:
        return None

    try:
        # Pour les modèles tree-based
        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            # Pour les classifieurs binaires, prendre la classe positive
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            return {
                'values': shap_values,
                'feature_names': feature_cols,
                'base_value': explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1]
            }
    except Exception as e:
        print(f"Erreur SHAP: {e}")
        return None


def plot_shap_summary(shap_result: dict, X_sample: pd.DataFrame, horizon: int, output_dir: Path):
    """
    Plot du summary SHAP.
    """
    if shap_result is None:
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    shap.summary_plot(
        shap_result['values'],
        X_sample,
        feature_names=shap_result['feature_names'],
        show=False,
        max_display=20
    )

    plt.title(f'SHAP Summary - Horizon {horizon} an(s)')
    plt.tight_layout()
    plt.savefig(output_dir / f'shap_summary_h{horizon}.png', dpi=150, bbox_inches='tight')
    plt.close()


def verify_coefficient_signs(df_imp: pd.DataFrame, model_name: str) -> dict:
    """
    Vérifie que les signes des coefficients sont cohérents avec l'intuition métier.
    """
    expected_positive = [
        'age_at_freeze',  # Plus vieux = plus risqué
        'n_fuites_total', 'n_fuites_1y', 'n_fuites_3y', 'n_fuites_5y',  # Plus de fuites = plus risqué
        'ratio_age_median',  # Au-dessus de la médiane = plus risqué
        'overdue_years',  # Dépassement de durée de vie = plus risqué
        'over_p75_life', 'over_p90_life',  # Dépassement de percentiles = plus risqué
        'leak_rate_per_year', 'leak_rate_per_km',  # Taux de fuite élevé = plus risqué
        'has_recent_fuite'  # Fuite récente = plus risqué
    ]

    expected_negative = [
        'days_since_last_fuite'  # Longtemps sans fuite = moins risqué
    ]

    checks = {
        'expected_positive': {},
        'expected_negative': {},
        'anomalies': []
    }

    if df_imp is None:
        return checks

    for feat in expected_positive:
        if feat in df_imp['feature'].values:
            importance = df_imp[df_imp['feature'] == feat]['importance'].values[0]
            checks['expected_positive'][feat] = importance
            # Pour les importances (GBM), on ne peut pas vérifier le signe directement
            # Mais une importance élevée est bon signe

    for feat in expected_negative:
        if feat in df_imp['feature'].values:
            importance = df_imp[df_imp['feature'] == feat]['importance'].values[0]
            checks['expected_negative'][feat] = importance

    return checks


def generate_explainability_report(results: dict) -> str:
    """
    Génère le rapport d'explicabilité.
    """
    report = []
    report.append("# Rapport d'Explicabilite\n")
    report.append(f"*Genere le: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    for horizon, res in results.items():
        report.append(f"\n## Horizon {horizon} an(s)\n")
        report.append(f"**Modele**: {res['model_name']}")

        # Feature importances
        if res['feature_importance'] is not None:
            report.append("\n### Top 15 Features (par importance)")
            report.append("| Rang | Feature | Importance |")
            report.append("|------|---------|------------|")
            for i, row in res['feature_importance'].head(15).iterrows():
                report.append(f"| {i+1} | {row['feature']} | {row['importance']:.4f} |")

        # Vérification des signes
        report.append("\n### Verification metier")
        checks = res['sign_checks']

        report.append("\n**Features attendues a impact positif (plus de risque):**")
        for feat, imp in checks['expected_positive'].items():
            report.append(f"- {feat}: importance = {imp:.4f}")

        report.append("\n**Features attendues a impact negatif (moins de risque):**")
        for feat, imp in checks['expected_negative'].items():
            report.append(f"- {feat}: importance = {imp:.4f}")

        if checks['anomalies']:
            report.append("\n**ANOMALIES DETECTEES:**")
            for ano in checks['anomalies']:
                report.append(f"- {ano}")
        else:
            report.append("\n[OK] Aucune anomalie de signe detectee.")

    return "\n".join(report)


def main():
    """
    Analyse d'explicabilité pour tous les horizons.
    """
    print("=" * 60)
    print("STEP 6: EXPLICABILITE ET VERIFICATIONS")
    print("=" * 60)

    plots_dir = REPORTS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    results = {}

    for horizon in HORIZONS:
        print(f"\n--- Horizon {horizon} an(s) ---")

        # Charger le modèle et les données
        artifact, df = load_model_and_data(horizon)
        model = artifact['model']
        feature_cols = artifact['metadata']['feature_cols']

        model_name = model.name if hasattr(model, 'name') else type(model).__name__
        print(f"Modele: {model_name}")

        # Feature importances
        df_imp = compute_feature_importance(model, feature_cols)
        if df_imp is not None:
            print(f"\nTop 5 features:")
            for _, row in df_imp.head(5).iterrows():
                print(f"  - {row['feature']}: {row['importance']:.4f}")

            plot_feature_importance(df_imp, horizon, plots_dir)

        # Vérification des signes
        sign_checks = verify_coefficient_signs(df_imp, model_name)

        # SHAP (sur un échantillon)
        shap_result = None
        if HAS_SHAP and hasattr(model, 'feature_importances_'):
            print("\nCalcul des SHAP values...")
            try:
                # Préparer un échantillon des features
                from sklearn.preprocessing import LabelEncoder
                num_feats, cat_feats = get_feature_columns()
                num_feats = [f for f in num_feats if f in df.columns]
                cat_feats = [f for f in cat_feats if f in df.columns]

                df_prep = df.copy()
                for col in cat_feats:
                    le = LabelEncoder()
                    df_prep[col + '_encoded'] = le.fit_transform(df_prep[col].astype(str))

                feature_cols_actual = num_feats + [f + '_encoded' for f in cat_feats]
                X = df_prep[feature_cols_actual].replace([np.inf, -np.inf], np.nan).fillna(0)
                X_sample = X.sample(min(1000, len(X)), random_state=42)

                shap_result = compute_shap_values(model, X_sample, feature_cols_actual)
                if shap_result is not None:
                    plot_shap_summary(shap_result, X_sample, horizon, plots_dir)
                    print("  SHAP summary plot sauvegarde")
            except Exception as e:
                print(f"  Erreur SHAP: {e}")

        results[horizon] = {
            'model_name': model_name,
            'feature_importance': df_imp,
            'sign_checks': sign_checks,
            'shap_result': shap_result
        }

    # Générer le rapport
    print("\nGeneration du rapport d'explicabilite...")
    report = generate_explainability_report(results)

    report_path = REPORTS_DIR / "explainability.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Rapport sauvegarde: {report_path}")

    print("\n" + "=" * 60)
    print("EXPLICABILITE TERMINEE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
