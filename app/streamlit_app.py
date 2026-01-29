"""
Optiplan - Optimisation du Renouvellement des Conduites
========================================================

Interface Streamlit pour l'optimisation sous contrainte budgetaire
et l'analyse what-if des scenarios de renouvellement.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import sys

# Ajouter le chemin parent pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from optimizer import BudgetOptimizer, OptimizationResult

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Optiplan - Renouvellement AEP",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalise
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
    }
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .kpi-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .kpi-delta {
        font-size: 1rem;
        padding: 0.25rem 0.5rem;
        border-radius: 0.5rem;
        background: rgba(255,255,255,0.2);
    }
    .stMetric {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CHARGEMENT DES DONNEES
# =============================================================================

@st.cache_data
def load_data():
    """Charge les donnees et le modele."""
    base_path = Path(__file__).parent.parent

    # Charger le dataset
    df = pd.read_pickle(base_path / "artifacts" / "dataset_freeze_h1.pkl")

    # Charger le modele
    model_path = base_path / "artifacts" / "model_v3_final.joblib"
    if model_path.exists():
        model_bundle = joblib.load(model_path)
    else:
        model_bundle = None

    return df, model_bundle


@st.cache_data
def prepare_features(df, model_bundle):
    """Prepare les features et calcule les scores."""
    from sklearn.preprocessing import LabelEncoder

    df = df.copy()

    # Encodings
    for col in ["materiau", "decade_install"]:
        if col in df.columns and col + "_enc" not in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    # Age corrections
    if "age_cap" not in df.columns:
        df["age_cap"] = df["age_at_freeze"].clip(upper=110)
        df["age_extreme"] = (df["age_at_freeze"] > 110).astype(int)
        p99 = df["age_at_freeze"].quantile(0.99)
        df["age_winsor"] = df["age_at_freeze"].clip(upper=p99)

    if "age_ratio_dec" not in df.columns:
        dec_med = df.groupby("decade_install")["age_at_freeze"].median()
        df["decade_med"] = df["decade_install"].map(dec_med)
        df["age_ratio_dec"] = df["age_at_freeze"] / df["decade_med"].replace(0, np.nan)
        df["age_ratio_dec"] = df["age_ratio_dec"].fillna(1.0)

    if "age_resid_mat" not in df.columns:
        mat_med = df.groupby("materiau")["age_at_freeze"].median()
        df["mat_med"] = df["materiau"].map(mat_med)
        df["age_resid_mat"] = df["age_at_freeze"] - df["mat_med"].fillna(df["age_at_freeze"].median())

    # Calculer les scores
    if model_bundle:
        features = model_bundle['features']
        model = model_bundle['model']
        calibrator = model_bundle['calibrator']

        X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
        raw_scores = model.predict_proba(X)[:, 1]
        scores = calibrator.transform(raw_scores)
    else:
        # Fallback: utiliser un score base sur l'age
        scores = (df['age_at_freeze'] / df['age_at_freeze'].max() * 0.1).values

    return df, scores


# =============================================================================
# COMPOSANTS UI
# =============================================================================

def render_kpi_card(label: str, value: str, delta: str = None, delta_color: str = "green"):
    """Affiche une carte KPI."""
    delta_html = ""
    if delta:
        color = "#10B981" if delta_color == "green" else "#EF4444"
        delta_html = f'<div style="color: {color}; font-size: 0.9rem; margin-top: 0.25rem;">{delta}</div>'

    st.markdown(f"""
    <div style="background: #f8fafc; padding: 1.25rem; border-radius: 0.75rem;
                border: 1px solid #e2e8f0; text-align: center;">
        <div style="color: #6B7280; font-size: 0.85rem; margin-bottom: 0.5rem;">{label}</div>
        <div style="color: #1E3A5F; font-size: 1.8rem; font-weight: 700;">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_header():
    """Affiche l'en-tete de l'application."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<p class="main-header">💧 Optiplan</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Optimisation du renouvellement des conduites AEP</p>',
                    unsafe_allow_html=True)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        scenario_name = st.text_input("Nom du scenario", value="Scenario 2024", label_visibility="collapsed")


def render_sidebar(df: pd.DataFrame):
    """Affiche la barre laterale de configuration."""
    st.sidebar.markdown("## ⚙️ Configuration Scenario")
    st.sidebar.markdown("---")

    # Budget
    st.sidebar.markdown("### 💰 Budget")
    budget_m = st.sidebar.slider(
        "Budget annuel (M€)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="Budget total disponible pour le renouvellement"
    )
    budget = budget_m * 1_000_000

    # Horizon
    st.sidebar.markdown("### 📅 Horizon")
    horizon = st.sidebar.slider(
        "Horizon temporel (annees)",
        min_value=1,
        max_value=20,
        value=5,
        help="Nombre d'annees pour la planification"
    )

    st.sidebar.markdown("---")

    # Filtres
    st.sidebar.markdown("### 🔧 Contraintes")

    # Materiaux
    materiaux = df['materiau'].unique().tolist()
    materiaux_selected = st.sidebar.multiselect(
        "Materiaux cibles",
        options=materiaux,
        default=['FT', 'FTG', 'ACIE'],
        help="Types de materiaux a prioriser"
    )

    # Diametre
    diam_range = st.sidebar.slider(
        "Diametre (mm)",
        min_value=int(df['diametre'].min()),
        max_value=int(df['diametre'].max()),
        value=(50, 300),
        help="Plage de diametres a considerer"
    )

    # Age minimum
    min_age = st.sidebar.slider(
        "Age minimum (annees)",
        min_value=0,
        max_value=100,
        value=30,
        help="Ne considerer que les conduites plus vieilles que..."
    )

    st.sidebar.markdown("---")

    # Bouton de lancement
    run_optimization = st.sidebar.button(
        "🚀 Lancer l'optimisation",
        type="primary",
        use_container_width=True
    )

    return {
        'budget': budget,
        'budget_m': budget_m,
        'horizon': horizon,
        'materials': materiaux_selected if materiaux_selected else None,
        'min_diameter': diam_range[0],
        'max_diameter': diam_range[1],
        'min_age': min_age,
        'run': run_optimization
    }


def render_results(result: OptimizationResult, config: dict):
    """Affiche les resultats de l'optimisation."""

    # KPIs principaux
    st.markdown("### 📊 Resultats de l'optimisation")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_kpi_card(
            "Budget Consomme",
            f"{result.total_cost / 1_000_000:.2f} M€",
            f"{result.budget_used_pct:.0f}% du budget"
        )

    with col2:
        render_kpi_card(
            "Lineaire Renouvele",
            f"{result.total_length / 1000:.1f} km",
            f"{len(result.selected_ids)} conduites"
        )

    with col3:
        risk_reduction = (1 - result.risk_after / result.risk_before) * 100 if result.risk_before > 0 else 0
        render_kpi_card(
            "Risque Initial",
            f"{result.risk_before:.1f}",
            None
        )

    with col4:
        render_kpi_card(
            "Risque Residuel",
            f"{result.risk_after:.1f}",
            f"-{risk_reduction:.0f}%",
            delta_color="green"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Graphiques
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        # Graphique de reduction du risque
        fig_risk = go.Figure()

        fig_risk.add_trace(go.Bar(
            name='Risque Elimine',
            x=['Avant', 'Apres'],
            y=[result.risk_before - result.risk_after, 0],
            marker_color='#10B981',
            text=[f'-{result.risk_before - result.risk_after:.1f}', ''],
            textposition='inside'
        ))

        fig_risk.add_trace(go.Bar(
            name='Risque Residuel',
            x=['Avant', 'Apres'],
            y=[result.risk_after, result.risk_after],
            marker_color='#EF4444',
            text=[f'{result.risk_after:.1f}', f'{result.risk_after:.1f}'],
            textposition='inside'
        ))

        fig_risk.update_layout(
            title="Reduction du Risque",
            barmode='stack',
            height=350,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis_title="Score de Risque Cumule"
        )

        st.plotly_chart(fig_risk, use_container_width=True)

    with col_chart2:
        # Defaillances evitees
        if result.n_failures_avoided > 0:
            fig_failures = go.Figure()

            fig_failures.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=result.n_failures_avoided,
                title={'text': "Defaillances Evitees (estimees)"},
                delta={'reference': 0, 'increasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [0, result.risk_before]},
                    'bar': {'color': "#10B981"},
                    'steps': [
                        {'range': [0, result.risk_before * 0.5], 'color': "#D1FAE5"},
                        {'range': [result.risk_before * 0.5, result.risk_before], 'color': "#FEE2E2"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': result.risk_before * 0.9
                    }
                }
            ))

            fig_failures.update_layout(height=350)
            st.plotly_chart(fig_failures, use_container_width=True)
        else:
            st.info("Aucune conduite selectionnee avec les contraintes actuelles.")


def render_what_if(optimizer: BudgetOptimizer, config: dict):
    """Affiche l'analyse what-if."""
    st.markdown("### 🔮 Analyse What-If")
    st.markdown("Impact des variations budgetaires sur la reduction du risque")

    # Analyse what-if
    what_if_results = optimizer.what_if_analysis(
        base_budget=config['budget'],
        variations=[-0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5],
        materials=config['materials'],
        min_diameter=config['min_diameter'],
        max_diameter=config['max_diameter'],
        min_age=config['min_age']
    )

    col1, col2 = st.columns(2)

    with col1:
        # Graphique budget vs risque
        fig_whatif = px.line(
            what_if_results,
            x='Budget (M€)',
            y='Reduction (%)',
            markers=True,
            title="Reduction du Risque vs Budget"
        )
        fig_whatif.update_traces(
            line_color='#6366F1',
            marker_size=10,
            line_width=3
        )
        fig_whatif.add_vline(
            x=config['budget_m'],
            line_dash="dash",
            line_color="red",
            annotation_text="Budget actuel"
        )
        fig_whatif.update_layout(height=400)
        st.plotly_chart(fig_whatif, use_container_width=True)

    with col2:
        # Graphique budget vs lineaire
        fig_linear = px.bar(
            what_if_results,
            x='Variation',
            y='Lineaire (km)',
            title="Lineaire Renouvelable par Scenario",
            color='Lineaire (km)',
            color_continuous_scale='Blues'
        )
        fig_linear.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_linear, use_container_width=True)

    # Tableau comparatif
    st.markdown("#### Comparaison des scenarios")
    display_cols = ['Variation', 'Budget (M€)', 'Conduites', 'Lineaire (km)',
                    'Reduction (%)', 'Defaillances Evitees']
    st.dataframe(
        what_if_results[display_cols].style.format({
            'Budget (M€)': '{:.2f}',
            'Lineaire (km)': '{:.1f}',
            'Reduction (%)': '{:.1f}%',
            'Defaillances Evitees': '{:.1f}'
        }).background_gradient(subset=['Reduction (%)'], cmap='Greens'),
        use_container_width=True,
        hide_index=True
    )


def render_details(result: OptimizationResult):
    """Affiche le detail des conduites selectionnees."""
    st.markdown("### 📋 Detail des Conduites Selectionnees")

    if len(result.details) > 0:
        # Statistiques
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre de conduites", len(result.details))
        with col2:
            st.metric("Cout moyen", f"{result.total_cost / len(result.details):,.0f} €")
        with col3:
            st.metric("Longueur moyenne", f"{result.total_length / len(result.details):.0f} m")

        # Tableau
        st.dataframe(
            result.details.head(100).style.format({
                'Score Risque': '{:.4f}',
                'Cout (€)': '{:,.0f}',
                'Efficacite': '{:.6f}'
            }).background_gradient(subset=['Score Risque'], cmap='Reds'),
            use_container_width=True,
            hide_index=True
        )

        if len(result.details) > 100:
            st.info(f"Affichage limite aux 100 premieres conduites sur {len(result.details)} selectionnees.")

        # Export
        csv = result.details.to_csv(index=False)
        st.download_button(
            label="📥 Exporter la liste (CSV)",
            data=csv,
            file_name="conduites_selectionnees.csv",
            mime="text/csv"
        )
    else:
        st.warning("Aucune conduite selectionnee avec les contraintes actuelles.")


# =============================================================================
# APPLICATION PRINCIPALE
# =============================================================================

def main():
    """Point d'entree principal."""

    render_header()

    # Charger les donnees
    with st.spinner("Chargement des donnees..."):
        df, model_bundle = load_data()
        df, scores = prepare_features(df, model_bundle)

    # Sidebar
    config = render_sidebar(df)

    # Creer l'optimiseur
    optimizer = BudgetOptimizer(df, scores)

    # Onglets
    tab1, tab2, tab3 = st.tabs(["📊 Resultats", "🔮 What-If", "📋 Details"])

    # Lancer l'optimisation
    if config['run'] or 'last_result' not in st.session_state:
        result = optimizer.optimize(
            budget=config['budget'],
            materials=config['materials'],
            min_diameter=config['min_diameter'],
            max_diameter=config['max_diameter'],
            min_age=config['min_age']
        )
        st.session_state['last_result'] = result
        st.session_state['last_config'] = config
    else:
        result = st.session_state.get('last_result')
        if result is None:
            result = optimizer.optimize(
                budget=config['budget'],
                materials=config['materials'],
                min_diameter=config['min_diameter'],
                max_diameter=config['max_diameter'],
                min_age=config['min_age']
            )

    with tab1:
        render_results(result, config)

    with tab2:
        render_what_if(optimizer, config)

    with tab3:
        render_details(result)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #9CA3AF; font-size: 0.85rem;">
            Optiplan v1.0 | Modele ML: HistGradientBoosting (Lift@10%: 6.57) |
            Donnees: 178,693 conduites
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
