"""
Module carte : affichage des tronçons avec Folium (si géométrie disponible).
Fournit un mode dégradé "sans carte" sinon.
"""
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional

# Folium import (optionnel)
try:
    import folium
    from streamlit_folium import st_folium

    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False


def get_risk_color(score: float, thresholds: tuple = (0.33, 0.66)) -> str:
    """Retourne une couleur selon le score de risque."""
    if score >= thresholds[1]:
        return "#e74c3c"  # Rouge
    elif score >= thresholds[0]:
        return "#f39c12"  # Orange
    else:
        return "#2ecc71"  # Vert


def render_map_placeholder():
    """Affiche un message quand la carte n'est pas disponible."""
    st.info(
        "**Mode sans carte** : les données ne contiennent pas de géométrie (WKT, lat/lon, GeoJSON). "
        "La carte sera disponible lorsque des données géographiques seront ajoutées.\n\n"
        "Colonnes attendues : `geometry` (WKT), ou `latitude`/`longitude`, ou un fichier GeoJSON."
    )


def render_map_folium(
    df: pd.DataFrame,
    selected_gids: Optional[set] = None,
    thresholds: tuple = (0.33, 0.66),
    geom_column: str = "geometry",
    max_features: int = 5000,
):
    """
    Affiche une carte Folium avec les tronçons colorés par risque.
    Nécessite une colonne de géométrie WKT (LineString/MultiLineString).

    Parameters
    ----------
    df : pd.DataFrame
        Doit contenir : GID, risk_score, et une colonne géométrie
    selected_gids : set, optional
        GIDs des tronçons sélectionnés (overlay)
    thresholds : tuple
        Seuils (bas, haut) pour la coloration
    geom_column : str
        Nom de la colonne contenant la géométrie WKT
    max_features : int
        Nombre max de features à afficher
    """
    if not FOLIUM_AVAILABLE:
        st.warning("Folium non installé. `pip install folium streamlit-folium`")
        return

    try:
        from shapely import wkt as shapely_wkt
        from shapely.geometry import mapping
    except ImportError:
        st.warning("Shapely non installé. `pip install shapely`")
        return

    # Limiter le nombre de features
    df_map = df.nlargest(max_features, "risk_score") if len(df) > max_features else df

    # Centre de la carte
    # Essayer de calculer le centroid moyen
    try:
        geoms = df_map[geom_column].apply(shapely_wkt.loads)
        centroids = geoms.apply(lambda g: (g.centroid.y, g.centroid.x))
        center_lat = centroids.apply(lambda c: c[0]).mean()
        center_lon = centroids.apply(lambda c: c[1]).mean()
    except Exception:
        center_lat, center_lon = 48.85, 2.35  # Paris par défaut

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    for _, row in df_map.iterrows():
        try:
            geom = shapely_wkt.loads(row[geom_column])
            coords = mapping(geom)

            gid = row["GID"]
            score = row["risk_score"]
            color = get_risk_color(score, thresholds)

            is_selected = selected_gids and gid in selected_gids

            # Style
            weight = 5 if is_selected else 3
            opacity = 1.0 if is_selected else 0.6
            dash = None

            # Tooltip
            tooltip_text = (
                f"<b>GID:</b> {gid}<br>"
                f"<b>Score:</b> {score:.3f}<br>"
                f"<b>MAT:</b> {row.get('MAT', 'N/A')}<br>"
                f"<b>DIAM:</b> {row.get('DIAMETRE', 'N/A')} mm<br>"
                f"<b>LNG:</b> {row.get('LNG', 'N/A')} m<br>"
                f"<b>Âge:</b> {row.get('age_at_freeze', 'N/A'):.1f} ans<br>"
                f"<b>Sélectionné:</b> {'OUI' if is_selected else 'NON'}"
            )

            if coords["type"] == "LineString":
                # Folium needs (lat, lon), shapely gives (lon, lat)
                line_coords = [(c[1], c[0]) for c in coords["coordinates"]]
                folium.PolyLine(
                    line_coords,
                    color=color,
                    weight=weight,
                    opacity=opacity,
                    tooltip=tooltip_text,
                ).add_to(m)

            elif coords["type"] == "MultiLineString":
                for line in coords["coordinates"]:
                    line_coords = [(c[1], c[0]) for c in line]
                    folium.PolyLine(
                        line_coords,
                        color=color,
                        weight=weight,
                        opacity=opacity,
                        tooltip=tooltip_text,
                    ).add_to(m)

        except Exception:
            continue

    # Légende
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background-color: white; padding: 10px; border: 2px solid grey;
                border-radius: 5px; font-size: 14px;">
        <b>Risque</b><br>
        <i style="background: #2ecc71; width: 20px; height: 10px; display: inline-block;"></i> Faible<br>
        <i style="background: #f39c12; width: 20px; height: 10px; display: inline-block;"></i> Moyen<br>
        <i style="background: #e74c3c; width: 20px; height: 10px; display: inline-block;"></i> Élevé<br>
        <i style="background: white; border: 3px solid black; width: 20px; height: 10px;
           display: inline-block;"></i> Sélectionné (trait épais)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, width=800, height=600)


def render_map_latlon(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    selected_gids: Optional[set] = None,
):
    """
    Carte simple avec st.map pour données lat/lon (points, pas lignes).
    """
    df_map = df[[lat_col, lon_col, "risk_score"]].copy()
    df_map.columns = ["lat", "lon", "risk_score"]
    df_map = df_map.dropna(subset=["lat", "lon"])

    st.map(df_map, latitude="lat", longitude="lon")


def render_table_view(
    df: pd.DataFrame,
    selected_gids: Optional[set] = None,
    page_size: int = 50,
):
    """
    Vue tableau filtrable des tronçons.
    """
    df_display = df.copy()

    # Ajouter flag sélection
    if selected_gids:
        df_display["selected"] = df_display["GID"].isin(selected_gids)
    else:
        df_display["selected"] = False

    # Colonnes à afficher
    display_cols = [
        "rank",
        "GID",
        "risk_score",
        "selected",
        "MAT",
        "DIAMETRE",
        "LNG",
        "age_at_freeze",
        "n_fuites_total",
        "days_since_last_fuite",
        "longueur_km",
    ]

    # Filtrer colonnes existantes
    display_cols = [c for c in display_cols if c in df_display.columns]

    # Formatage
    if "cost" in df_display.columns:
        display_cols.append("cost")

    st.dataframe(
        df_display[display_cols].style.format(
            {
                "risk_score": "{:.4f}",
                "age_at_freeze": "{:.1f}",
                "longueur_km": "{:.3f}",
                "days_since_last_fuite": "{:.0f}",
                "cost": "{:,.0f}",
            },
            na_rep="-",
        ),
        use_container_width=True,
        height=min(page_size * 35 + 50, 800),
    )
