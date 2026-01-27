"""
Application Streamlit - Plan de Renouvellement Réseau AEP
=========================================================
Interface end-to-end pour générer et visualiser des scénarios
de plan de renouvellement basés sur le scoring ML anti-fuite.

Lancer : streamlit run streamlit_app/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import time
from pathlib import Path
from datetime import date, datetime

# Ajouter les chemins nécessaires
APP_DIR = Path(__file__).parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_DIR / "src"))
sys.path.insert(0, str(APP_DIR.parent / "src"))

from src.data_loader import (
    load_patrimoine,
    load_anomalies,
    load_life_stats,
    get_data_dates,
    has_geometry,
    get_geometry_column,
)
from src.scoring_service import score_pipes, get_model_info
from src.scenario import (
    ScenarioParams,
    ScenarioResult,
    compute_costs,
    run_baseline_scenario,
)
from src.optimization import greedy_solver
from src.optimization import ortools_solver
from src.viz.charts import (
    render_kpi_cards,
    plot_score_distribution,
    plot_capture_curve,
    plot_decile_analysis,
    plot_risk_by_material,
    plot_risk_by_age,
    plot_treated_vs_untreated_pie,
)
from src.viz.map_view import (
    render_map_placeholder,
    render_map_folium,
    render_table_view,
)
from src.exports import (
    export_plan_csv,
    export_all_scored_csv,
    export_geojson,
    generate_scenario_report,
)


# ─────────────────────────────────────────────
# Configuration page
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Plan de Renouvellement AEP",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────
if "scenario_result" not in st.session_state:
    st.session_state.scenario_result = None
if "optimized_result" not in st.session_state:
    st.session_state.optimized_result = None
if "df_scored" not in st.session_state:
    st.session_state.df_scored = None


# ─────────────────────────────────────────────
# Data loading (cached)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Chargement des données patrimoine...")
def _load_patrimoine():
    return load_patrimoine()


@st.cache_data(show_spinner="Chargement des anomalies...")
def _load_anomalies():
    return load_anomalies()


# ─────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────
st.sidebar.title("Plan de Renouvellement AEP")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "1 - Scenario",
        "2 - Carte & Troncons",
        "3 - Plan optimise",
        "4 - Bilan Risque",
        "5 - Export",
    ],
    index=0,
)


# ─────────────────────────────────────────────
# PAGE 1 : SCÉNARIO
# ─────────────────────────────────────────────
if page == "1 - Scenario":
    st.title("Definition du scenario")

    # Load data for date range
    df_assets = _load_patrimoine()
    df_anomalies = _load_anomalies()
    dates_info = get_data_dates(df_assets, df_anomalies)

    # Model info
    with st.expander("Informations modele"):
        try:
            model_info = get_model_info()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Type:** {model_info['model_type']}")
                st.write(f"**Features:** {model_info['n_features']}")
            with col2:
                st.write(f"**Horizon entrainement:** {model_info['horizon_years']} an(s)")
                st.write(f"**N train:** {model_info['n_train']:,}")
            with col3:
                metrics = model_info.get("test_metrics", {})
                st.write(f"**Capture@10%:** {metrics.get('capture_10pct', 'N/A')}")
                st.write(f"**Lift@10%:** {metrics.get('lift_10pct', 'N/A')}")
        except Exception as e:
            st.warning(f"Impossible de charger les infos modele: {e}")

    st.markdown("---")

    # Scenario parameters
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Parametres temporels")

        max_date = dates_info["max_obs_date"].date()
        freeze_date = st.date_input(
            "Date de gel (freeze_date)",
            value=max_date,
            min_value=date(2000, 1, 1),
            max_value=max_date,
            help="Toutes les features seront calculees avec des donnees <= cette date",
        )

        horizon_years = st.selectbox(
            "Horizon de prediction (annees)",
            options=[1, 3, 5],
            index=0,
        )

        plan_years = st.number_input(
            "Duree du plan (annees)",
            min_value=1,
            max_value=10,
            value=1,
            help="1 = plan annuel, N = multi-annuel",
        )

    with col_right:
        st.subheader("Contraintes budgetaires")

        budget_total = st.number_input(
            "Budget total (euros)",
            min_value=100_000,
            max_value=500_000_000,
            value=10_000_000,
            step=500_000,
            format="%d",
        )

        cost_per_km = st.number_input(
            "Cout par km (euros/km)",
            min_value=10_000,
            max_value=2_000_000,
            value=200_000,
            step=10_000,
            format="%d",
        )

        use_lineaire_max = st.checkbox("Contrainte lineaire max")
        lineaire_max_km = None
        if use_lineaire_max:
            lineaire_max_km = st.number_input(
                "Lineaire max (km)",
                min_value=1.0,
                max_value=10000.0,
                value=50.0,
                step=5.0,
            )

    st.markdown("---")

    # Selection mode
    col_mode1, col_mode2 = st.columns(2)

    with col_mode1:
        st.subheader("Mode de selection")
        scenario_mode = st.radio(
            "Type de plan",
            ["Baseline ML (top-K)", "Plan optimise (solveur)"],
            index=0,
            help="Baseline = simple tri par score. Optimise = solveur MILP sous contraintes.",
        )

        if scenario_mode == "Baseline ML (top-K)":
            baseline_mode = st.selectbox(
                "Critere de selection",
                ["top_k_pct", "top_n", "top_length"],
                format_func=lambda x: {
                    "top_k_pct": "Top K% des troncons",
                    "top_n": "Top N troncons",
                    "top_length": "Top lineaire (km)",
                }[x],
            )

            if baseline_mode == "top_k_pct":
                top_k_pct = st.slider("Pourcentage top-K (%)", 1, 50, 10) / 100.0
                top_n = 1000
                top_length_km = 50.0
            elif baseline_mode == "top_n":
                top_n = st.number_input("Nombre de troncons", 10, 50000, 1000, step=100)
                top_k_pct = 0.10
                top_length_km = 50.0
            else:
                top_length_km = st.number_input("Lineaire (km)", 1.0, 5000.0, 50.0, step=5.0)
                top_k_pct = 0.10
                top_n = 1000
        else:
            baseline_mode = "top_k_pct"
            top_k_pct = 0.10
            top_n = 1000
            top_length_km = 50.0

    with col_mode2:
        st.subheader("Exclusions")
        materials = sorted(df_assets["MAT"].dropna().unique().tolist())
        excluded_materials = st.multiselect(
            "Materiaux a exclure",
            options=materials,
            default=[],
        )

        weight_mode = st.radio(
            "Ponderation du risque",
            ["length", "uniform"],
            format_func=lambda x: {
                "length": "Par longueur (km)",
                "uniform": "Uniforme (1 par troncon)",
            }[x],
        )

    st.markdown("---")

    # Generate button
    if st.button("Generer le scenario", type="primary", use_container_width=True):
        with st.spinner("Scoring en cours..."):
            t0 = time.time()

            # Score all pipes
            df_scored = score_pipes(
                df_assets,
                df_anomalies,
                freeze_date=str(freeze_date),
                horizon_years=horizon_years,
            )

            scoring_time = time.time() - t0

        if len(df_scored) == 0:
            st.error("Aucun troncon actif a cette date. Verifiez la freeze_date.")
        else:
            st.success(
                f"{len(df_scored):,} troncons scores en {scoring_time:.1f}s"
            )

            # Build scenario params
            mode = "baseline" if scenario_mode == "Baseline ML (top-K)" else "optimized"
            params = ScenarioParams(
                freeze_date=str(freeze_date),
                horizon_years=horizon_years,
                plan_years=plan_years,
                mode=mode,
                baseline_mode=baseline_mode,
                top_k_pct=top_k_pct,
                top_n=top_n,
                top_length_km=top_length_km,
                budget_total=float(budget_total),
                lineaire_max_km=lineaire_max_km,
                cost_per_km=float(cost_per_km),
                excluded_materials=excluded_materials,
                weight_mode=weight_mode,
            )

            # Add costs
            df_scored["longueur_km"] = df_scored["LNG"] / 1000.0
            df_scored["cost"] = compute_costs(df_scored, params)

            # Run baseline
            with st.spinner("Generation du plan baseline..."):
                baseline_result = run_baseline_scenario(df_scored, params)

            st.session_state.scenario_result = baseline_result
            st.session_state.df_scored = df_scored

            # If optimized mode, also run solver
            if mode == "optimized":
                with st.spinner("Optimisation en cours (solveur)..."):
                    # Try OR-Tools first, fallback to greedy
                    if ortools_solver.is_available():
                        solver_result = ortools_solver.solve_milp(
                            df=df_scored,
                            budget_total=float(budget_total),
                            lineaire_max_km=lineaire_max_km,
                            weight_mode=weight_mode,
                            excluded_gids=params.excluded_gids,
                            excluded_materials=excluded_materials,
                        )
                    else:
                        solver_result = greedy_solver.solve_greedy(
                            df=df_scored,
                            budget_total=float(budget_total),
                            lineaire_max_km=lineaire_max_km,
                            weight_mode=weight_mode,
                            excluded_gids=params.excluded_gids,
                            excluded_materials=excluded_materials,
                        )

                    # Build optimized result
                    selected_gids = solver_result["selected_gids"]
                    df_selected = df_scored[df_scored["GID"].isin(selected_gids)].copy()
                    df_selected = df_selected.sort_values(
                        "risk_score", ascending=False
                    ).reset_index(drop=True)
                    df_selected["rank"] = range(1, len(df_selected) + 1)

                    opt_result = ScenarioResult(
                        params=params,
                        df_scored=df_scored,
                        df_selected=df_selected,
                        solver_status=solver_result["status"],
                        solver_time_s=solver_result["solve_time_s"],
                    )
                    opt_result.compute_kpis()
                    st.session_state.optimized_result = opt_result

                    # Show solver logs
                    with st.expander("Logs solveur"):
                        for log_line in solver_result.get("logs", []):
                            st.text(log_line)

            # Summary
            result = (
                st.session_state.optimized_result
                if st.session_state.optimized_result
                else baseline_result
            )
            st.markdown("### Resume du scenario")
            render_kpi_cards(result)


# ─────────────────────────────────────────────
# PAGE 2 : CARTE & TRONÇONS
# ─────────────────────────────────────────────
elif page == "2 - Carte & Troncons":
    st.title("Carte & Troncons")

    result = st.session_state.scenario_result or st.session_state.optimized_result
    if result is None:
        st.warning("Generez d'abord un scenario (Page 1).")
    else:
        df_scored = result.df_scored
        selected_gids = set(result.df_selected["GID"])

        # Filters
        st.sidebar.markdown("### Filtres")
        filter_selected_only = st.sidebar.checkbox("Troncons selectionnes uniquement")

        materials_available = sorted(df_scored["MAT"].dropna().unique().tolist())
        filter_materials = st.sidebar.multiselect(
            "Filtrer par materiau", materials_available
        )

        score_range = st.sidebar.slider(
            "Plage de score",
            0.0,
            1.0,
            (0.0, 1.0),
            step=0.01,
        )

        # Seuils de couleur carte
        st.sidebar.markdown("### Seuils carte")
        threshold_low = st.sidebar.slider("Seuil faible/moyen", 0.0, 1.0, 0.33, 0.01)
        threshold_high = st.sidebar.slider("Seuil moyen/eleve", 0.0, 1.0, 0.66, 0.01)

        # Apply filters
        df_view = df_scored.copy()
        if filter_selected_only:
            df_view = df_view[df_view["GID"].isin(selected_gids)]
        if filter_materials:
            df_view = df_view[df_view["MAT"].isin(filter_materials)]
        df_view = df_view[
            (df_view["risk_score"] >= score_range[0])
            & (df_view["risk_score"] <= score_range[1])
        ]

        st.write(f"**{len(df_view):,} troncons affiches** (sur {len(df_scored):,} total)")

        # Map
        df_full = _load_patrimoine()
        geo_available = has_geometry(df_full)

        if geo_available:
            geom_col = get_geometry_column(df_full)
            # Merge geometry into df_view
            df_view_geo = df_view.merge(
                df_full[["GID", geom_col]], on="GID", how="left"
            )
            render_map_folium(
                df_view_geo,
                selected_gids=selected_gids,
                thresholds=(threshold_low, threshold_high),
                geom_column=geom_col,
            )
        else:
            render_map_placeholder()

        st.markdown("---")

        # Table
        st.subheader("Tableau des troncons")
        render_table_view(df_view, selected_gids=selected_gids)

        # Single pipe details
        st.markdown("---")
        st.subheader("Detail d'un troncon")
        gid_input = st.number_input(
            "Entrez un GID",
            min_value=0,
            value=0,
            step=1,
            help="Saisissez le GID d'un troncon pour voir ses details",
        )

        if gid_input > 0:
            pipe_data = df_scored[df_scored["GID"] == gid_input]
            if len(pipe_data) == 0:
                st.warning(f"GID {gid_input} non trouve dans les donnees scorees.")
            else:
                pipe = pipe_data.iloc[0]
                is_selected = gid_input in selected_gids

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**GID:** {int(pipe['GID'])}")
                    st.write(f"**Score:** {pipe['risk_score']:.4f}")
                    st.write(f"**Rang:** {int(pipe['rank'])}")
                    st.write(f"**Selectionne:** {'OUI' if is_selected else 'NON'}")
                with col2:
                    st.write(f"**Materiau:** {pipe.get('MAT', '-')}")
                    st.write(f"**Diametre:** {pipe.get('DIAMETRE', '-')} mm")
                    st.write(f"**Longueur:** {pipe.get('LNG', 0):.0f} m")
                    st.write(f"**Age:** {pipe.get('age_at_freeze', 0):.1f} ans")
                with col3:
                    st.write(f"**Fuites total:** {int(pipe.get('n_fuites_total', 0))}")
                    st.write(f"**Fuites 1an:** {int(pipe.get('n_fuites_1y', 0))}")
                    st.write(f"**Fuites 3ans:** {int(pipe.get('n_fuites_3y', 0))}")
                    st.write(
                        f"**Jours depuis derniere fuite:** "
                        f"{int(pipe.get('days_since_last_fuite', 0))}"
                    )

                # Feature importance bar (top features)
                st.markdown("**Profil de risque (features principales)**")
                feature_cols = [
                    "age_at_freeze",
                    "n_fuites_total",
                    "ratio_age_median",
                    "overdue_years",
                    "has_recent_fuite",
                    "longueur_km",
                ]
                feature_cols = [f for f in feature_cols if f in pipe.index]
                if feature_cols:
                    import plotly.express as px

                    feat_df = pd.DataFrame(
                        {
                            "Feature": feature_cols,
                            "Valeur": [float(pipe[f]) for f in feature_cols],
                        }
                    )
                    fig = px.bar(
                        feat_df,
                        x="Valeur",
                        y="Feature",
                        orientation="h",
                        title=f"Valeurs des features principales (GID {gid_input})",
                        template="plotly_white",
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 3 : PLAN OPTIMISÉ
# ─────────────────────────────────────────────
elif page == "3 - Plan optimise":
    st.title("Plan optimise")

    result = st.session_state.optimized_result or st.session_state.scenario_result
    if result is None:
        st.warning("Generez d'abord un scenario (Page 1).")
    else:
        # Option to run optimizer from here
        if st.session_state.optimized_result is None:
            st.info(
                "Le plan affiche est le baseline ML. "
                "Pour un plan optimise, selectionnez 'Plan optimise (solveur)' en Page 1."
            )

            if st.button("Lancer l'optimisation maintenant"):
                df_scored = st.session_state.df_scored
                params = result.params

                if df_scored is not None:
                    with st.spinner("Optimisation en cours..."):
                        if ortools_solver.is_available():
                            solver_result = ortools_solver.solve_milp(
                                df=df_scored,
                                budget_total=params.budget_total,
                                lineaire_max_km=params.lineaire_max_km,
                                weight_mode=params.weight_mode,
                                excluded_materials=params.excluded_materials,
                            )
                        else:
                            solver_result = greedy_solver.solve_greedy(
                                df=df_scored,
                                budget_total=params.budget_total,
                                lineaire_max_km=params.lineaire_max_km,
                                weight_mode=params.weight_mode,
                                excluded_materials=params.excluded_materials,
                            )

                        selected_gids = solver_result["selected_gids"]
                        df_selected = df_scored[
                            df_scored["GID"].isin(selected_gids)
                        ].copy()
                        df_selected = df_selected.sort_values(
                            "risk_score", ascending=False
                        ).reset_index(drop=True)
                        df_selected["rank"] = range(1, len(df_selected) + 1)

                        opt_params = ScenarioParams(
                            freeze_date=params.freeze_date,
                            horizon_years=params.horizon_years,
                            plan_years=params.plan_years,
                            mode="optimized",
                            budget_total=params.budget_total,
                            lineaire_max_km=params.lineaire_max_km,
                            cost_per_km=params.cost_per_km,
                            excluded_materials=params.excluded_materials,
                            weight_mode=params.weight_mode,
                        )

                        opt_result = ScenarioResult(
                            params=opt_params,
                            df_scored=df_scored,
                            df_selected=df_selected,
                            solver_status=solver_result["status"],
                            solver_time_s=solver_result["solve_time_s"],
                        )
                        opt_result.compute_kpis()
                        st.session_state.optimized_result = opt_result
                        result = opt_result

                        for log_line in solver_result.get("logs", []):
                            st.text(log_line)

                        st.rerun()

        # Display the plan
        st.markdown("### Resume du plan")
        render_kpi_cards(result)

        st.markdown("---")

        # Solver info
        if result.solver_status:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Statut solveur:** {result.solver_status}")
            with col2:
                st.write(f"**Temps resolution:** {result.solver_time_s:.2f}s")

        # Comparison baseline vs optimized
        baseline = st.session_state.scenario_result
        optimized = st.session_state.optimized_result

        if baseline and optimized:
            st.markdown("### Comparaison Baseline vs Optimise")
            comp_df = pd.DataFrame(
                {
                    "KPI": [
                        "Troncons selectionnes",
                        "Risque traite",
                        "Risque traite (%)",
                        "Budget utilise (EUR)",
                        "Lineaire (km)",
                        "Lineaire (%)",
                    ],
                    "Baseline": [
                        baseline.n_selected,
                        f"{baseline.risk_treated:,.2f}",
                        f"{baseline.risk_treated/max(baseline.risk_total,1)*100:.1f}%",
                        f"{baseline.budget_used:,.0f}",
                        f"{baseline.coverage_length_km:,.1f}",
                        f"{baseline.coverage_length_pct:.1f}%",
                    ],
                    "Optimise": [
                        optimized.n_selected,
                        f"{optimized.risk_treated:,.2f}",
                        f"{optimized.risk_treated/max(optimized.risk_total,1)*100:.1f}%",
                        f"{optimized.budget_used:,.0f}",
                        f"{optimized.coverage_length_km:,.1f}",
                        f"{optimized.coverage_length_pct:.1f}%",
                    ],
                }
            )
            st.table(comp_df)

        # Selected pipes table
        st.markdown("### Troncons selectionnes (chantiers)")
        df_sel = result.df_selected
        if len(df_sel) > 0:
            display_cols = [
                "rank", "GID", "risk_score", "MAT", "DIAMETRE", "LNG",
                "longueur_km", "age_at_freeze", "n_fuites_total", "cost",
            ]
            display_cols = [c for c in display_cols if c in df_sel.columns]
            st.dataframe(
                df_sel[display_cols].style.format(
                    {
                        "risk_score": "{:.4f}",
                        "age_at_freeze": "{:.1f}",
                        "longueur_km": "{:.3f}",
                        "cost": "{:,.0f}",
                    },
                    na_rep="-",
                ),
                use_container_width=True,
                height=600,
            )

            # Constraints check
            st.markdown("### Verification des contraintes")
            total_cost = df_sel["cost"].sum() if "cost" in df_sel.columns else 0
            total_length = df_sel["longueur_km"].sum()

            checks = []
            checks.append(
                {
                    "Contrainte": "Budget",
                    "Limite": f"{result.params.budget_total:,.0f} EUR",
                    "Utilise": f"{total_cost:,.0f} EUR",
                    "Respectee": "OUI" if total_cost <= result.params.budget_total else "NON",
                }
            )

            if result.params.lineaire_max_km:
                checks.append(
                    {
                        "Contrainte": "Lineaire max",
                        "Limite": f"{result.params.lineaire_max_km:,.1f} km",
                        "Utilise": f"{total_length:,.1f} km",
                        "Respectee": (
                            "OUI"
                            if total_length <= result.params.lineaire_max_km
                            else "NON"
                        ),
                    }
                )

            st.table(pd.DataFrame(checks))
        else:
            st.warning("Aucun troncon selectionne.")


# ─────────────────────────────────────────────
# PAGE 4 : BILAN RISQUE
# ─────────────────────────────────────────────
elif page == "4 - Bilan Risque":
    st.title("Bilan Risque")

    result = st.session_state.optimized_result or st.session_state.scenario_result
    if result is None:
        st.warning("Generez d'abord un scenario (Page 1).")
    else:
        # KPI Cards
        render_kpi_cards(result)

        st.markdown("---")

        # Two columns for charts
        col1, col2 = st.columns(2)

        with col1:
            # Pie chart
            fig_pie = plot_treated_vs_untreated_pie(result)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Score distribution
            fig_dist = plot_score_distribution(
                result.df_scored, result.df_selected
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("---")

        # Capture curve
        st.subheader("Courbe de capture")
        fig_capture = plot_capture_curve(result.df_scored)
        st.plotly_chart(fig_capture, use_container_width=True)

        # Capture stats
        df_sorted = result.df_scored.sort_values(
            "risk_score", ascending=False
        ).reset_index(drop=True)
        n = len(df_sorted)
        exposure = df_sorted["longueur_km"].values
        weighted_risk = df_sorted["risk_score"].values * exposure
        total_risk = weighted_risk.sum()
        cum_risk = np.cumsum(weighted_risk)

        capture_stats = []
        for pct in [5, 10, 15, 20, 30, 50]:
            idx = min(int(n * pct / 100) - 1, n - 1)
            if idx >= 0:
                cap = cum_risk[idx] / total_risk * 100
                lift = cap / pct
                capture_stats.append(
                    {
                        "Top K%": f"{pct}%",
                        "N troncons": idx + 1,
                        "Capture (%)": f"{cap:.1f}%",
                        "Lift": f"{lift:.2f}x",
                    }
                )
        st.table(pd.DataFrame(capture_stats))

        st.markdown("---")

        # Decile analysis
        st.subheader("Analyse par decile")
        fig_decile = plot_decile_analysis(result.df_scored)
        st.plotly_chart(fig_decile, use_container_width=True)

        # Risk by material
        st.subheader("Risque par materiau")
        fig_mat = plot_risk_by_material(result.df_scored)
        st.plotly_chart(fig_mat, use_container_width=True)

        # Risk vs age
        st.subheader("Score vs Age")
        fig_age = plot_risk_by_age(result.df_scored)
        st.plotly_chart(fig_age, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 5 : EXPORT
# ─────────────────────────────────────────────
elif page == "5 - Export":
    st.title("Export")

    result = st.session_state.optimized_result or st.session_state.scenario_result
    if result is None:
        st.warning("Generez d'abord un scenario (Page 1).")
    else:
        st.subheader("Exports disponibles")

        col1, col2 = st.columns(2)

        with col1:
            # CSV plan
            st.markdown("#### CSV - Troncons selectionnes")
            csv_plan = export_plan_csv(result)
            st.download_button(
                label="Telecharger le plan (CSV)",
                data=csv_plan,
                file_name=f"plan_renouvellement_{result.params.freeze_date}.csv",
                mime="text/csv",
            )

            # CSV all scored
            st.markdown("#### CSV - Tous les troncons scores")
            csv_all = export_all_scored_csv(result)
            st.download_button(
                label="Telecharger tous les scores (CSV)",
                data=csv_all,
                file_name=f"scores_complets_{result.params.freeze_date}.csv",
                mime="text/csv",
            )

        with col2:
            # GeoJSON
            st.markdown("#### GeoJSON - Plan de renouvellement")
            geojson = export_geojson(result)
            st.download_button(
                label="Telecharger le plan (GeoJSON)",
                data=geojson,
                file_name=f"plan_renouvellement_{result.params.freeze_date}.geojson",
                mime="application/geo+json",
            )

            # Rapport
            st.markdown("#### Rapport scenario (Markdown)")
            report_md = generate_scenario_report(result)
            st.download_button(
                label="Telecharger le rapport",
                data=report_md,
                file_name=f"rapport_scenario_{result.params.freeze_date}.md",
                mime="text/markdown",
            )

        # Preview rapport
        st.markdown("---")
        st.subheader("Apercu du rapport")
        with st.expander("Voir le rapport", expanded=False):
            st.markdown(generate_scenario_report(result))

        # Export summary
        st.markdown("---")
        st.subheader("Resume de l'export")
        st.write(f"**Date de gel:** {result.params.freeze_date}")
        st.write(f"**Horizon:** {result.params.horizon_years} an(s)")
        st.write(f"**Mode:** {result.params.mode}")
        st.write(f"**Troncons selectionnes:** {result.n_selected:,}")
        st.write(f"**Risque traite:** {result.risk_treated:,.2f} ({result.risk_treated/max(result.risk_total,1)*100:.1f}%)")
        st.write(f"**Budget utilise:** {result.budget_used:,.0f} EUR")
