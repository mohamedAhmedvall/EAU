"""
Optiplan - Application de planification des renouvellements de canalisations
============================================================================

Point d'entrée principal de l'application Streamlit.

Lancer avec: streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Optiplan",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/water.png", width=60)
    st.title("Optiplan")
    st.caption("Optimisation des renouvellements")
    st.markdown("---")

# Page d'accueil
st.title("💧 Optiplan")
st.subheader("Planification et optimisation des renouvellements de canalisations")

st.markdown("""
Bienvenue dans **Optiplan**, votre outil d'aide à la décision pour optimiser
les programmes de renouvellement de votre réseau d'eau potable.
""")

st.markdown("---")

# Cards de navigation
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📋 Scénarios
    Gérez vos programmes de renouvellement et comparez différentes stratégies.
    """)
    if st.button("Voir les scénarios", use_container_width=True):
        st.switch_page("pages/1_Scenarios.py")

with col2:
    st.markdown("""
    ### 🚀 Simulation
    Créez un nouveau scénario et lancez une optimisation.
    """)
    if st.button("Nouvelle simulation", type="primary", use_container_width=True):
        st.switch_page("pages/2_Simulation.py")

with col3:
    st.markdown("""
    ### ⚙️ Paramètres
    Configurez les coûts et les seuils de classement.
    """)
    if st.button("Configurer", use_container_width=True):
        st.switch_page("pages/3_Parametres.py")

st.markdown("---")

# Stats rapides
st.subheader("📊 Aperçu rapide")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from data_loader import load_troncons, list_scenarios, get_lineaire_total

    # Charger les données en cache
    @st.cache_data(ttl=300)
    def get_stats():
        troncons = load_troncons()
        scenarios = list_scenarios()
        lineaire = get_lineaire_total(troncons)
        risque_moyen = sum(t.score_ml for t in troncons) / len(troncons) if troncons else 0
        return len(troncons), lineaire, risque_moyen, len(scenarios)

    with st.spinner("Chargement des statistiques..."):
        nb_troncons, lineaire_km, risque_moyen, nb_scenarios = get_stats()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔧 Tronçons en service", f"{nb_troncons:,}")
    with col2:
        st.metric("📏 Linéaire total", f"{lineaire_km:,.0f} km")
    with col3:
        st.metric("⚠️ Risque moyen", f"{risque_moyen*100:.1f}/100")
    with col4:
        st.metric("📋 Scénarios créés", nb_scenarios)

except Exception as e:
    st.warning(f"Impossible de charger les statistiques: {e}")
    st.info("Les données seront chargées lors de votre première simulation.")

st.markdown("---")

# Guide rapide
with st.expander("📖 Guide rapide"):
    st.markdown("""
    ### Comment utiliser Optiplan ?

    1. **Configurez vos paramètres** (optionnel)
       - Définissez les coûts de renouvellement par matériau/diamètre
       - Ajustez les seuils de classement des risques

    2. **Créez un scénario de simulation**
       - Définissez l'horizon temporel (1-30 ans)
       - Fixez le budget annuel
       - Configurez les contraintes de linéaire (min/max)

    3. **Lancez l'optimisation**
       - Le moteur MILP sélectionne les tronçons optimaux
       - Maximise les défaillances évitées sous contraintes

    4. **Analysez les résultats**
       - Visualisez les KPIs (budget, linéaire, risque)
       - Exportez le plan de renouvellement en CSV

    ### Formule d'optimisation

    ```
    Maximiser : Σ (score_i × longueur_i × x_i)

    Contraintes :
      - Budget max : Σ (coût_i × x_i) ≤ budget
      - Linéaire min : Σ (longueur_i × x_i) ≥ min_km
      - Linéaire max : Σ (longueur_i × x_i) ≤ max_km
    ```

    où `score_i = α × risque_ML + (1-α) × opportunité`
    """)

# Footer
st.markdown("---")
st.caption("Optiplan v1.0 | Développé pour l'optimisation des réseaux AEP")
