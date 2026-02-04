"""
Module de visualisation : charts et KPI cards pour l'app Streamlit.
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional


def render_kpi_cards(result):
    """Affiche les KPI cards du bilan risque."""
    cols = st.columns(4)

    with cols[0]:
        st.metric(
            "Risque total",
            f"{result.risk_total:,.1f}",
            help="Somme(score × longueur_km) sur tout le périmètre",
        )

    with cols[1]:
        pct_treated = (
            result.risk_treated / result.risk_total * 100
            if result.risk_total > 0
            else 0
        )
        st.metric(
            "Risque traité",
            f"{result.risk_treated:,.1f}",
            delta=f"{pct_treated:.1f}%",
        )

    with cols[2]:
        st.metric(
            "Risque non traité",
            f"{result.risk_untreated:,.1f}",
        )

    with cols[3]:
        st.metric(
            "Tronçons sélectionnés",
            f"{result.n_selected:,} / {result.n_total:,}",
        )

    cols2 = st.columns(3)
    with cols2[0]:
        st.metric(
            "Budget utilisé",
            f"{result.budget_used:,.0f} €",
            help="Somme des coûts proxy des tronçons sélectionnés",
        )
    with cols2[1]:
        st.metric(
            "Linéaire couvert",
            f"{result.coverage_length_km:,.1f} km",
            delta=f"{result.coverage_length_pct:.1f}%",
        )
    with cols2[2]:
        efficiency = (
            result.risk_treated / result.budget_used * 1_000_000
            if result.budget_used > 0
            else 0
        )
        st.metric(
            "Efficience",
            f"{efficiency:,.2f}",
            help="Risque traité par M€",
        )


def plot_score_distribution(df_scored: pd.DataFrame, df_selected: Optional[pd.DataFrame] = None):
    """Histogramme de distribution des scores de risque."""
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=df_scored["risk_score"],
            nbinsx=50,
            name="Tous les tronçons",
            marker_color="steelblue",
            opacity=0.7,
        )
    )

    if df_selected is not None and len(df_selected) > 0:
        fig.add_trace(
            go.Histogram(
                x=df_selected["risk_score"],
                nbinsx=50,
                name="Sélectionnés",
                marker_color="crimson",
                opacity=0.7,
            )
        )

    fig.update_layout(
        title="Distribution des scores de risque",
        xaxis_title="Score de risque",
        yaxis_title="Nombre de tronçons",
        barmode="overlay",
        template="plotly_white",
        height=400,
    )
    return fig


def plot_capture_curve(df_scored: pd.DataFrame):
    """
    Courbe de capture (Capture@K) : % du risque total capturé
    en ciblant les top K% des tronçons.
    """
    df = df_scored.sort_values("risk_score", ascending=False).reset_index(drop=True)

    exposure = df["longueur_km"].values if "longueur_km" in df.columns else np.ones(len(df))
    weighted_risk = df["risk_score"].values * exposure
    total_risk = weighted_risk.sum()

    n = len(df)
    # Calculate at various K percentages
    k_values = np.arange(1, n + 1)
    k_pct = k_values / n * 100
    cumulative_risk = np.cumsum(weighted_risk)
    capture_pct = cumulative_risk / total_risk * 100

    fig = go.Figure()

    # Capture curve
    fig.add_trace(
        go.Scatter(
            x=k_pct,
            y=capture_pct,
            mode="lines",
            name="Capture ML",
            line=dict(color="crimson", width=2),
        )
    )

    # Random baseline
    fig.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode="lines",
            name="Aléatoire",
            line=dict(color="gray", dash="dash"),
        )
    )

    # Markers at 5%, 10%, 20%
    for pct in [5, 10, 20]:
        idx = min(int(n * pct / 100) - 1, n - 1)
        if idx >= 0:
            cap = capture_pct[idx]
            fig.add_trace(
                go.Scatter(
                    x=[pct],
                    y=[cap],
                    mode="markers+text",
                    text=[f"{cap:.1f}%"],
                    textposition="top right",
                    marker=dict(size=10, color="crimson"),
                    name=f"Cap@{pct}%",
                    showlegend=False,
                )
            )

    fig.update_layout(
        title="Courbe de capture (risque pondéré)",
        xaxis_title="% des tronçons ciblés (top-K%)",
        yaxis_title="% du risque total capturé",
        template="plotly_white",
        height=450,
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[0, 100]),
    )
    return fig


def plot_decile_analysis(df_scored: pd.DataFrame):
    """Analyse par décile : risque moyen et nombre de tronçons par décile."""
    df = df_scored.copy()
    df["decile"] = pd.qcut(df["risk_score"], q=10, labels=False, duplicates="drop") + 1

    agg = df.groupby("decile").agg(
        n_troncons=("GID", "count"),
        score_mean=("risk_score", "mean"),
        longueur_total_km=("longueur_km", "sum"),
    ).reset_index()

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Score moyen par décile", "Linéaire par décile (km)"),
    )

    colors = [
        "#2ecc71" if d <= 3 else "#f39c12" if d <= 7 else "#e74c3c"
        for d in agg["decile"]
    ]

    fig.add_trace(
        go.Bar(
            x=agg["decile"],
            y=agg["score_mean"],
            marker_color=colors,
            name="Score moyen",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=agg["decile"],
            y=agg["longueur_total_km"],
            marker_color=colors,
            name="Linéaire (km)",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="Analyse par décile de risque",
        template="plotly_white",
        height=400,
    )
    fig.update_xaxes(title_text="Décile", row=1, col=1)
    fig.update_xaxes(title_text="Décile", row=1, col=2)
    return fig


def plot_risk_by_material(df_scored: pd.DataFrame):
    """Score de risque moyen par matériau."""
    agg = (
        df_scored.groupby("MAT")
        .agg(
            n=("GID", "count"),
            score_mean=("risk_score", "mean"),
            score_p90=("risk_score", lambda x: x.quantile(0.90)),
            longueur_km=("longueur_km", "sum"),
        )
        .reset_index()
        .sort_values("score_mean", ascending=True)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=agg["MAT"],
            x=agg["score_mean"],
            orientation="h",
            name="Score moyen",
            marker_color="steelblue",
            text=[f"n={n}" for n in agg["n"]],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=agg["MAT"],
            x=agg["score_p90"],
            mode="markers",
            name="P90",
            marker=dict(size=10, color="crimson", symbol="diamond"),
        )
    )

    fig.update_layout(
        title="Score de risque par matériau",
        xaxis_title="Score de risque",
        template="plotly_white",
        height=max(300, len(agg) * 35),
    )
    return fig


def plot_risk_by_age(df_scored: pd.DataFrame):
    """Scatter plot score vs âge."""
    sample = df_scored
    if len(df_scored) > 5000:
        sample = df_scored.sample(5000, random_state=42)

    fig = px.scatter(
        sample,
        x="age_at_freeze",
        y="risk_score",
        color="MAT",
        size="longueur_km",
        size_max=8,
        opacity=0.4,
        title="Score de risque vs Âge du tronçon",
        labels={"age_at_freeze": "Âge (années)", "risk_score": "Score de risque"},
        template="plotly_white",
        height=450,
    )
    return fig


def plot_treated_vs_untreated_pie(result):
    """Pie chart risque traité vs non traité."""
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Risque traité", "Risque non traité"],
                values=[result.risk_treated, result.risk_untreated],
                marker_colors=["#2ecc71", "#e74c3c"],
                hole=0.4,
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(
        title="Répartition du risque",
        template="plotly_white",
        height=350,
    )
    return fig
