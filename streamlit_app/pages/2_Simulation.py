"""
Page Simulation - Configuration et exécution des scénarios
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
from datetime import datetime

# Ajouter le chemin src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import Scenario
from data_loader import (
    load_troncons, save_scenario, load_scenario,
    load_parametres_couts, get_lineaire_total
)
from optimizer import simuler_scenario

st.set_page_config(
    page_title="Simulation - Optiplan",
    page_icon="🚀",
    layout="wide"
)

# Initialiser session state
if "troncons" not in st.session_state:
    st.session_state["troncons"] = None
if "resultat" not in st.session_state:
    st.session_state["resultat"] = None
if "scenario" not in st.session_state:
    st.session_state["scenario"] = Scenario()

# Charger un scénario existant si demandé
if "scenario_to_view" in st.session_state:
    scenario_id = st.session_state.pop("scenario_to_view")
    loaded = load_scenario(scenario_id)
    if loaded:
        st.session_state["scenario"] = loaded
        st.session_state["resultat"] = loaded.resultat


def load_data():
    """Charge les données si pas encore fait."""
    if st.session_state["troncons"] is None:
        with st.spinner("Chargement des données..."):
            params_couts = load_parametres_couts()
            st.session_state["troncons"] = load_troncons(params_couts)
    return st.session_state["troncons"]


# Charger les données
troncons = load_data()
lineaire_total_km = get_lineaire_total(troncons)

# Titre
st.title("🚀 Configuration Scénario")
st.markdown("Créez ou modifiez les paramètres de votre simulation de renouvellement.")

# Layout en 2 colonnes
col_config, col_results = st.columns([1, 2])

with col_config:
    st.subheader("Paramètres")

    scenario = st.session_state["scenario"]

    # Nom et description
    scenario.nom = st.text_input(
        "Nom du scénario",
        value=scenario.nom,
        placeholder="Ex: Scénario Optimisé T2 2024"
    )

    scenario.description = st.text_area(
        "Description",
        value=scenario.description,
        placeholder="Objectifs et contraintes...",
        height=80
    )

    st.markdown("---")

    # Horizon temporel
    st.markdown("**Horizon Temporel**")
    scenario.horizon_ans = st.slider(
        "Durée (années)",
        min_value=1,
        max_value=30,
        value=scenario.horizon_ans,
        help="Nombre d'années de planification"
    )

    st.markdown("---")

    # Budget
    st.markdown("**Budget Annuel**")
    budget_m = st.number_input(
        "Budget (M€)",
        min_value=0.1,
        max_value=100.0,
        value=scenario.budget_annuel / 1_000_000,
        step=0.1,
        format="%.1f"
    )
    scenario.budget_annuel = budget_m * 1_000_000

    budget_total = scenario.budget_annuel * scenario.horizon_ans
    st.caption(f"💡 Budget total estimé sur {scenario.horizon_ans} ans : **{budget_total/1e6:.1f} M€**")

    st.markdown("---")

    # Contraintes linéaires
    st.markdown("**Contraintes Linéaires**")
    st.caption(f"Réseau total : {lineaire_total_km:.0f} km")

    col_min, col_max = st.columns(2)
    with col_min:
        scenario.lineaire_min_pct = st.number_input(
            "Min (%)",
            min_value=0.0,
            max_value=10.0,
            value=scenario.lineaire_min_pct,
            step=0.1,
            format="%.1f",
            help="Linéaire minimum à renouveler par an"
        )
        lineaire_min_km = lineaire_total_km * scenario.lineaire_min_pct / 100
        st.caption(f"= {lineaire_min_km:.1f} km/an")

    with col_max:
        scenario.lineaire_max_pct = st.number_input(
            "Max (%)",
            min_value=0.1,
            max_value=20.0,
            value=scenario.lineaire_max_pct,
            step=0.1,
            format="%.1f",
            help="Capacité max des équipes"
        )
        lineaire_max_km = lineaire_total_km * scenario.lineaire_max_pct / 100
        st.caption(f"= {lineaire_max_km:.1f} km/an")

    # Validation
    if scenario.lineaire_min_pct > scenario.lineaire_max_pct:
        st.error("⚠️ Le minimum ne peut pas être supérieur au maximum !")

    st.markdown("---")

    # Coefficient Alpha
    st.markdown("**Pondération Objectif**")
    scenario.alpha = st.slider(
        "Coefficient α",
        min_value=0.0,
        max_value=1.0,
        value=scenario.alpha,
        step=0.1,
        help="α=1 : 100% risque ML | α=0 : 100% opportunités"
    )

    col_risk, col_opp = st.columns(2)
    with col_risk:
        st.metric("Risque ML", f"{scenario.alpha*100:.0f}%")
    with col_opp:
        st.metric("Opportunités", f"{(1-scenario.alpha)*100:.0f}%")

    st.markdown("---")

    # Bouton lancer
    if st.button("▶️ Lancer la simulation", type="primary", use_container_width=True):
        if scenario.lineaire_min_pct > scenario.lineaire_max_pct:
            st.error("Corrigez les contraintes de linéaire avant de lancer.")
        else:
            with st.spinner("Optimisation en cours..."):
                resultat = simuler_scenario(troncons, scenario)
                st.session_state["resultat"] = resultat
                scenario.resultat = resultat
                if resultat.succes:
                    scenario.statut = "termine"
                scenario.date_modification = datetime.now().isoformat()
                st.session_state["scenario"] = scenario


# Colonne résultats
with col_results:
    resultat = st.session_state.get("resultat")

    if resultat is None:
        st.info("👈 Configurez les paramètres et lancez la simulation pour voir les résultats.")

        # Afficher quelques stats sur les données
        st.subheader("📊 Aperçu du patrimoine")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tronçons", f"{len(troncons):,}")
        with col2:
            st.metric("Linéaire total", f"{lineaire_total_km:.0f} km")
        with col3:
            risque_moyen = sum(t.score_ml for t in troncons) / len(troncons) if troncons else 0
            st.metric("Risque moyen", f"{risque_moyen*100:.1f}/100")

    else:
        if not resultat.succes:
            st.error(f"❌ {resultat.message}")
        else:
            # Breadcrumb
            st.caption(f"Projets / Renouvellement 2024 / **{scenario.nom or 'Nouveau scénario'}**")

            # Actions
            col_title, col_actions = st.columns([3, 1])
            with col_title:
                st.subheader("📈 Visualisation des Résultats")
            with col_actions:
                col_exp, col_save = st.columns(2)
                with col_exp:
                    # Export CSV
                    if resultat.troncons_planifies:
                        df_export = pd.DataFrame([
                            {
                                "GID": tp.troncon.gid,
                                "Matériau": tp.troncon.materiau,
                                "Diamètre (mm)": tp.troncon.diametre,
                                "Longueur (m)": tp.troncon.longueur,
                                "Année pose": tp.troncon.annee_pose,
                                "Score ML": tp.troncon.score_ml,
                                "Score composite": tp.score_composite,
                                "Coût (€)": tp.troncon.cout_total,
                                "Année renouvellement": tp.annee_renouvellement,
                            }
                            for tp in resultat.troncons_planifies
                        ])
                        csv = df_export.to_csv(index=False)
                        st.download_button(
                            "📥 Export",
                            csv,
                            f"plan_renouvellement_{scenario.id}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                with col_save:
                    if st.button("💾 Sauver", use_container_width=True):
                        save_scenario(scenario)
                        st.success("Scénario sauvegardé !")

            # KPIs principaux
            st.markdown("---")
            kpi_cols = st.columns(4)

            with kpi_cols[0]:
                budget_pct = (resultat.budget_total_consomme / (scenario.budget_annuel * scenario.horizon_ans)) * 100
                st.metric(
                    "💰 Budget Consommé",
                    f"{resultat.budget_total_consomme/1e6:.1f} M€",
                    f"{budget_pct:.0f}% du budget alloué"
                )

            with kpi_cols[1]:
                km_par_an = resultat.lineaire_total_km / scenario.horizon_ans if scenario.horizon_ans > 0 else 0
                st.metric(
                    "📏 Linéaire Renouvelé",
                    f"{resultat.lineaire_total_km:.1f} km",
                    f"{km_par_an:.1f} km/an en moyenne"
                )

            with kpi_cols[2]:
                st.metric(
                    "⚠️ Risque Moyen Initial",
                    f"{resultat.risque_initial*100:.0f}/100"
                )

            with kpi_cols[3]:
                reduction = ((resultat.risque_initial - resultat.risque_residuel) / resultat.risque_initial * 100) if resultat.risque_initial > 0 else 0
                st.metric(
                    "✅ Risque Résiduel",
                    f"{resultat.risque_residuel*100:.0f}/100",
                    f"-{reduction:.0f}%",
                    delta_color="inverse"
                )

            # Graphique et tableau
            st.markdown("---")
            tab_graph, tab_annee, tab_troncons = st.tabs(["📊 Graphique", "📅 Détail par Année", "🔧 Tronçons Sélectionnés"])

            with tab_graph:
                if resultat.kpis_par_annee:
                    df_kpi = pd.DataFrame([
                        {
                            "Année": k.annee,
                            "Budget (M€)": k.budget_consomme / 1e6,
                            "Linéaire (km)": k.lineaire_km,
                            "Nb tronçons": k.nb_troncons
                        }
                        for k in resultat.kpis_par_annee
                    ])

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df_kpi["Année"],
                        y=df_kpi["Linéaire (km)"],
                        name="Linéaire (km)",
                        marker_color="#3B82F6"
                    ))
                    fig.update_layout(
                        title="Linéaire renouvelé par année",
                        xaxis_title="Année",
                        yaxis_title="km",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with tab_annee:
                st.markdown("**Détail par Année**")
                if resultat.kpis_par_annee:
                    df_annee = pd.DataFrame([
                        {
                            "Année": k.annee,
                            "Budget (M€)": f"{k.budget_consomme/1e6:.2f}",
                            "Linéaire (km)": f"{k.lineaire_km:.1f}",
                            "Nb tronçons": k.nb_troncons,
                            "Risque moyen": f"{k.risque_moyen*100:.0f}/100"
                        }
                        for k in resultat.kpis_par_annee
                    ])
                    st.dataframe(df_annee, use_container_width=True, hide_index=True)

            with tab_troncons:
                st.markdown(f"**{len(resultat.troncons_planifies)} tronçons sélectionnés**")
                if resultat.troncons_planifies:
                    # Trier par année puis score
                    troncons_tries = sorted(
                        resultat.troncons_planifies,
                        key=lambda tp: (tp.annee_renouvellement, -tp.score_composite)
                    )

                    df_troncons = pd.DataFrame([
                        {
                            "GID": tp.troncon.gid,
                            "Matériau": tp.troncon.materiau,
                            "Ø (mm)": int(tp.troncon.diametre),
                            "Long. (m)": f"{tp.troncon.longueur:.0f}",
                            "Posé en": tp.troncon.annee_pose,
                            "Score": f"{tp.score_composite*100:.0f}",
                            "Coût (€)": f"{tp.troncon.cout_total:,.0f}",
                            "Année renouv.": tp.annee_renouvellement,
                        }
                        for tp in troncons_tries[:500]  # Limiter l'affichage
                    ])

                    st.dataframe(df_troncons, use_container_width=True, hide_index=True)

                    if len(resultat.troncons_planifies) > 500:
                        st.caption(f"Affichage limité aux 500 premiers tronçons sur {len(resultat.troncons_planifies)}")
