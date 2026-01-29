"""
Page Scénarios - Liste et gestion des scénarios
"""
import streamlit as st
import sys
from pathlib import Path

# Ajouter le chemin src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import list_scenarios, delete_scenario, load_scenario
from models import Scenario

st.set_page_config(
    page_title="Scénarios - Optiplan",
    page_icon="📋",
    layout="wide"
)

st.title("📋 Liste des scénarios")
st.markdown("Gérez vos programmes de renouvellement et optimisez vos budgets.")

# Bouton nouveau scénario
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("➕ Nouveau scénario", type="primary", use_container_width=True):
        st.switch_page("pages/2_Simulation.py")

# Filtres
st.markdown("---")
col_filter1, col_filter2, col_filter3 = st.columns([2, 1, 1])

with col_filter1:
    search = st.text_input("🔍 Rechercher par nom, description...", placeholder="Rechercher...")

with col_filter2:
    statut_filter = st.selectbox(
        "Statut",
        ["Tout", "Brouillon", "Terminé"],
        index=0
    )

with col_filter3:
    sort_by = st.selectbox(
        "Trier par",
        ["Date (récent)", "Date (ancien)", "Nom", "Budget"],
        index=0
    )

# Charger les scénarios
scenarios = list_scenarios()

# Filtrer
if search:
    search_lower = search.lower()
    scenarios = [s for s in scenarios if search_lower in s.nom.lower() or search_lower in s.description.lower()]

if statut_filter == "Brouillon":
    scenarios = [s for s in scenarios if s.statut == "brouillon"]
elif statut_filter == "Terminé":
    scenarios = [s for s in scenarios if s.statut == "termine"]

# Trier
if sort_by == "Date (récent)":
    scenarios.sort(key=lambda s: s.date_modification, reverse=True)
elif sort_by == "Date (ancien)":
    scenarios.sort(key=lambda s: s.date_modification)
elif sort_by == "Nom":
    scenarios.sort(key=lambda s: s.nom.lower())
elif sort_by == "Budget":
    scenarios.sort(key=lambda s: s.budget_annuel, reverse=True)

# Afficher le tableau
st.markdown("---")

if len(scenarios) == 0:
    st.info("Aucun scénario trouvé. Créez votre premier scénario !")
else:
    # En-têtes
    cols = st.columns([2.5, 3, 1, 1.2, 1, 1.5])
    cols[0].markdown("**NOM DU SCÉNARIO**")
    cols[1].markdown("**DESCRIPTION**")
    cols[2].markdown("**HORIZON**")
    cols[3].markdown("**BUDGET**")
    cols[4].markdown("**STATUT**")
    cols[5].markdown("**ACTIONS**")

    st.markdown("---")

    # Lignes
    for scenario in scenarios:
        cols = st.columns([2.5, 3, 1, 1.2, 1, 1.5])

        with cols[0]:
            st.markdown(f"**{scenario.nom or 'Sans nom'}**")

        with cols[1]:
            desc = scenario.description[:50] + "..." if len(scenario.description) > 50 else scenario.description
            st.markdown(desc or "-")

        with cols[2]:
            st.markdown(f"{scenario.horizon_ans} ans")

        with cols[3]:
            budget_m = scenario.budget_annuel / 1_000_000
            st.markdown(f"{budget_m:.1f} M€")

        with cols[4]:
            if scenario.statut == "termine":
                st.success("Terminé", icon="✅")
            else:
                st.warning("Brouillon", icon="📝")

        with cols[5]:
            action_cols = st.columns(3)
            with action_cols[0]:
                if st.button("👁️", key=f"view_{scenario.id}", help="Voir"):
                    st.session_state["scenario_to_view"] = scenario.id
                    st.switch_page("pages/2_Simulation.py")
            with action_cols[1]:
                if st.button("📋", key=f"dup_{scenario.id}", help="Dupliquer"):
                    # Dupliquer le scénario
                    new_scenario = Scenario(
                        nom=f"{scenario.nom} (copie)",
                        description=scenario.description,
                        horizon_ans=scenario.horizon_ans,
                        budget_annuel=scenario.budget_annuel,
                        lineaire_min_pct=scenario.lineaire_min_pct,
                        lineaire_max_pct=scenario.lineaire_max_pct,
                        alpha=scenario.alpha,
                    )
                    from data_loader import save_scenario
                    save_scenario(new_scenario)
                    st.rerun()
            with action_cols[2]:
                if st.button("🗑️", key=f"del_{scenario.id}", help="Supprimer"):
                    delete_scenario(scenario.id)
                    st.rerun()

        st.markdown("---")

    # Pagination info
    st.caption(f"Affichage de 1 à {len(scenarios)} sur {len(scenarios)} résultats")
