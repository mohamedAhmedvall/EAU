"""
Page Paramètres - Configuration des coûts et seuils
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

# Ajouter le chemin src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import ParametresCouts, ParametresSeuils, RegleCout
from data_loader import load_parametres, save_parametres, get_default_couts

st.set_page_config(
    page_title="Paramètres - Optiplan",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Paramètres")
st.markdown("Configurez les coûts de renouvellement et les seuils de classement.")

# Charger les paramètres actuels
if "params_couts" not in st.session_state:
    couts, seuils = load_parametres()
    st.session_state["params_couts"] = couts
    st.session_state["params_seuils"] = seuils

params_couts = st.session_state["params_couts"]
params_seuils = st.session_state["params_seuils"]

# Onglets
tab_couts, tab_seuils = st.tabs(["💰 Coûts de Renouvellement", "📊 Seuils de Classement"])

# ============================================================================
# ONGLET COÛTS
# ============================================================================
with tab_couts:
    st.subheader("Paramètres Coûts de Renouvellement")
    st.markdown("Configurez les coûts unitaires pour le calcul des investissements. "
                "Définissez un coût par défaut et des règles spécifiques par matériau et diamètre.")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("### 💵 Coût par défaut")
        st.caption("Appliqué aux tronçons sans règle spécifique.")

        params_couts.cout_defaut = st.number_input(
            "Coût unitaire (€/m)",
            min_value=10.0,
            max_value=1000.0,
            value=float(params_couts.cout_defaut),
            step=10.0,
            format="%.0f"
        )

        st.markdown("---")

        st.markdown("### ℹ️ Règles de validation")
        st.info("""
        - Les plages de diamètres ne doivent pas se chevaucher pour un même matériau.
        - Tous les coûts doivent être positifs.
        """)

        st.markdown("---")

        # Matériaux configurés
        st.markdown("### 📦 Matériaux configurés")
        materiaux = list(set(r.materiau for r in params_couts.regles))
        for mat in sorted(materiaux):
            count = sum(1 for r in params_couts.regles if r.materiau == mat)
            st.markdown(f"- **{mat}** ({count})")

    with col_right:
        st.markdown("### 📋 Barèmes spécifiques")
        st.markdown("Définissez les coûts par matériau et diamètre.")

        # Filtrer par matériau
        all_materiaux = ["Tous les matériaux"] + sorted(set(r.materiau for r in params_couts.regles))
        filtre_mat = st.selectbox("Filtrer par matériau", all_materiaux)

        # Afficher les règles
        regles_affichees = params_couts.regles
        if filtre_mat != "Tous les matériaux":
            regles_affichees = [r for r in params_couts.regles if r.materiau == filtre_mat]

        # Convertir en DataFrame pour affichage
        if regles_affichees:
            df_regles = pd.DataFrame([
                {
                    "Matériau": r.materiau,
                    "Diamètre Min (mm)": int(r.diametre_min),
                    "Diamètre Max (mm)": int(r.diametre_max),
                    "Coût (€/m)": int(r.cout_unitaire),
                }
                for r in regles_affichees
            ])

            # Éditer avec st.data_editor
            edited_df = st.data_editor(
                df_regles,
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Matériau": st.column_config.SelectboxColumn(
                        options=["FT", "FTG", "PVC", "PEHD", "AC", "BTM", "POLY", "INOX", "AUTRE"],
                        required=True
                    ),
                    "Diamètre Min (mm)": st.column_config.NumberColumn(min_value=0, max_value=2000, step=10),
                    "Diamètre Max (mm)": st.column_config.NumberColumn(min_value=0, max_value=2000, step=10),
                    "Coût (€/m)": st.column_config.NumberColumn(min_value=10, max_value=1000, step=10),
                }
            )

            # Mettre à jour les règles si modifié
            if st.button("💾 Appliquer les modifications", key="save_couts"):
                nouvelles_regles = []
                for _, row in edited_df.iterrows():
                    nouvelles_regles.append(RegleCout(
                        materiau=row["Matériau"],
                        diametre_min=float(row["Diamètre Min (mm)"]),
                        diametre_max=float(row["Diamètre Max (mm)"]),
                        cout_unitaire=float(row["Coût (€/m)"])
                    ))

                # Si on filtrait, garder les autres règles
                if filtre_mat != "Tous les matériaux":
                    autres_regles = [r for r in params_couts.regles if r.materiau != filtre_mat]
                    nouvelles_regles = autres_regles + nouvelles_regles

                params_couts.regles = nouvelles_regles
                st.session_state["params_couts"] = params_couts
                save_parametres(params_couts, params_seuils)
                st.success("✅ Paramètres de coûts sauvegardés !")
                st.rerun()

        else:
            st.info("Aucune règle définie. Utilisez le bouton ci-dessous pour ajouter des règles.")

        # Bouton ajouter une règle
        st.markdown("---")
        col_add, col_reset = st.columns(2)
        with col_add:
            if st.button("➕ Ajouter une règle", use_container_width=True):
                params_couts.regles.append(RegleCout(
                    materiau="FT",
                    diametre_min=0,
                    diametre_max=150,
                    cout_unitaire=200
                ))
                st.session_state["params_couts"] = params_couts
                st.rerun()

        with col_reset:
            if st.button("🔄 Réinitialiser par défaut", use_container_width=True):
                st.session_state["params_couts"] = get_default_couts()
                save_parametres(st.session_state["params_couts"], params_seuils)
                st.success("Paramètres réinitialisés !")
                st.rerun()


# ============================================================================
# ONGLET SEUILS
# ============================================================================
with tab_seuils:
    st.subheader("Paramètres Seuils de Classement")
    st.markdown("Définissez les bornes des seuils **S3**, **S4** et **S5** pour calibrer "
                "la transformation du Score de Risque en note patrimoniale de 1 à 5.")

    col_config, col_preview = st.columns([1, 1])

    with col_config:
        st.markdown("### 🎚️ Configuration des intervalles")

        # Tableau des classes
        classes_data = [
            {"Classe": "1", "Description": "Non Pertinent (NR)", "Intervalle": "Score = 0", "Seuil": "-"},
            {"Classe": "2", "Description": "Bon état", "Intervalle": f"0 < Score ≤ S3", "Seuil": ""},
            {"Classe": "3", "Description": "État moyen", "Intervalle": "S3 < Score ≤ S4", "Seuil": ""},
            {"Classe": "4", "Description": "Mauvais état", "Intervalle": "S4 < Score ≤ S5", "Seuil": ""},
            {"Classe": "5", "Description": "Critique", "Intervalle": "Score > S5", "Seuil": "Max (100)"},
        ]

        st.markdown("#### Définition des seuils")

        col_s3, col_s4, col_s5 = st.columns(3)

        with col_s3:
            new_s3 = st.number_input(
                "S3 (Bon → Moyen)",
                min_value=1.0,
                max_value=99.0,
                value=float(params_seuils.s3),
                step=5.0,
                help="Seuil entre classe 2 (Bon) et classe 3 (Moyen)"
            )

        with col_s4:
            new_s4 = st.number_input(
                "S4 (Moyen → Mauvais)",
                min_value=1.0,
                max_value=99.0,
                value=float(params_seuils.s4),
                step=5.0,
                help="Seuil entre classe 3 (Moyen) et classe 4 (Mauvais)"
            )

        with col_s5:
            new_s5 = st.number_input(
                "S5 (Mauvais → Critique)",
                min_value=1.0,
                max_value=99.0,
                value=float(params_seuils.s5),
                step=5.0,
                help="Seuil entre classe 4 (Mauvais) et classe 5 (Critique)"
            )

        # Validation
        if new_s3 >= new_s4:
            st.error("⚠️ S3 doit être inférieur à S4")
        elif new_s4 >= new_s5:
            st.error("⚠️ S4 doit être inférieur à S5")
        else:
            params_seuils.s3 = new_s3
            params_seuils.s4 = new_s4
            params_seuils.s5 = new_s5

        st.markdown("---")

        # Boutons
        col_save, col_reset = st.columns(2)
        with col_save:
            if st.button("💾 Sauvegarder", key="save_seuils", use_container_width=True):
                st.session_state["params_seuils"] = params_seuils
                save_parametres(params_couts, params_seuils)
                st.success("✅ Seuils sauvegardés !")

        with col_reset:
            if st.button("🔄 Défaut", key="reset_seuils", use_container_width=True):
                st.session_state["params_seuils"] = ParametresSeuils()
                params_seuils = st.session_state["params_seuils"]
                save_parametres(params_couts, params_seuils)
                st.success("Seuils réinitialisés !")
                st.rerun()

    with col_preview:
        st.markdown("### 📊 Répartition estimée")
        st.caption("Visualisation de l'impact des seuils actuels sur la population de canalisations.")

        # Créer un graphique de répartition estimée
        # (simulation basée sur une distribution typique)
        import numpy as np

        # Simuler une distribution de scores (exponentielle décroissante typique)
        np.random.seed(42)
        scores_simules = np.random.exponential(scale=20, size=1000)
        scores_simules = np.clip(scores_simules, 0, 100)

        # Compter par classe
        c1 = np.sum(scores_simules == 0)
        c2 = np.sum((scores_simules > 0) & (scores_simules <= params_seuils.s3))
        c3 = np.sum((scores_simules > params_seuils.s3) & (scores_simules <= params_seuils.s4))
        c4 = np.sum((scores_simules > params_seuils.s4) & (scores_simules <= params_seuils.s5))
        c5 = np.sum(scores_simules > params_seuils.s5)

        # Graphique en barres horizontales
        fig = go.Figure()

        categories = ["C5: Critique", "C4: Mauvais", "C3: Moyen", "C2: Bon", "C1: NR"]
        values = [c5, c4, c3, c2, c1]
        colors = ["#EF4444", "#F97316", "#EAB308", "#22C55E", "#94A3B8"]

        fig.add_trace(go.Bar(
            y=categories,
            x=values,
            orientation='h',
            marker_color=colors,
            text=[f"{v}" for v in values],
            textposition='auto'
        ))

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Nombre de tronçons (simulé)",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Estimations km
        st.markdown("#### Estimations (simulation)")
        col_c5, col_c4 = st.columns(2)
        with col_c5:
            km_c5 = c5 * 0.05  # ~50m par tronçon
            st.metric("🔴 CRITIQUE (C5)", f"~{km_c5:.0f} km")
        with col_c4:
            km_c4 = c4 * 0.05
            st.metric("🟠 MAUVAIS (C4)", f"~{km_c4:.0f} km")

        # Vue d'ensemble visuelle
        st.markdown("---")
        st.markdown("#### Vue d'ensemble des intervalles")

        # Barre de couleur
        fig2 = go.Figure()

        # Segments de la barre
        segments = [
            (0, params_seuils.s3, "#22C55E", "C2: Bon"),
            (params_seuils.s3, params_seuils.s4, "#EAB308", "C3: Moyen"),
            (params_seuils.s4, params_seuils.s5, "#F97316", "C4: Mauvais"),
            (params_seuils.s5, 100, "#EF4444", "C5: Critique"),
        ]

        for start, end, color, label in segments:
            fig2.add_trace(go.Bar(
                x=[end - start],
                y=["Score"],
                orientation='h',
                marker_color=color,
                name=label,
                text=label,
                textposition='inside',
                hovertemplate=f"{label}<br>{start:.0f} - {end:.0f}<extra></extra>"
            ))

        fig2.update_layout(
            barmode='stack',
            height=100,
            margin=dict(l=20, r=20, t=10, b=10),
            showlegend=False,
            xaxis=dict(range=[0, 100], title="Score (0-100)"),
            yaxis=dict(visible=False)
        )

        # Ajouter les marqueurs de seuils
        for val, label in [(params_seuils.s3, "S3"), (params_seuils.s4, "S4"), (params_seuils.s5, "S5")]:
            fig2.add_vline(x=val, line_dash="dash", line_color="black", line_width=2)
            fig2.add_annotation(x=val, y=1.1, text=f"{label} ({val:.0f})", showarrow=False, yref="paper")

        st.plotly_chart(fig2, use_container_width=True)
