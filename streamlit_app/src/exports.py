"""
Module d'export : CSV, GeoJSON, rapport scénario.
"""
import pandas as pd
import json
from datetime import datetime
from typing import Optional
from io import StringIO


def export_plan_csv(result) -> str:
    """
    Exporte le plan (tronçons sélectionnés) en CSV.
    """
    df = result.df_selected.copy()

    export_cols = [
        "GID", "rank", "risk_score", "MAT", "DIAMETRE", "LNG",
        "longueur_km", "age_at_freeze", "n_fuites_total",
        "days_since_last_fuite", "cost",
    ]
    export_cols = [c for c in export_cols if c in df.columns]

    return df[export_cols].to_csv(index=False)


def export_all_scored_csv(result) -> str:
    """
    Exporte tous les tronçons scorés en CSV.
    """
    df = result.df_scored.copy()
    df["selected"] = df["GID"].isin(set(result.df_selected["GID"]))

    export_cols = [
        "GID", "rank", "risk_score", "selected", "MAT", "DIAMETRE", "LNG",
        "longueur_km", "age_at_freeze", "n_fuites_total",
        "days_since_last_fuite", "cost",
    ]
    export_cols = [c for c in export_cols if c in df.columns]

    return df[export_cols].to_csv(index=False)


def export_geojson(result, geom_column: Optional[str] = None) -> str:
    """
    Exporte les tronçons sélectionnés en GeoJSON.
    Si pas de géométrie, exporte un GeoJSON vide avec propriétés.
    """
    df = result.df_selected.copy()

    features = []
    for _, row in df.iterrows():
        properties = {
            "GID": int(row["GID"]),
            "risk_score": round(float(row["risk_score"]), 4),
            "MAT": str(row.get("MAT", "")),
            "DIAMETRE": float(row.get("DIAMETRE", 0)),
            "LNG": float(row.get("LNG", 0)),
            "age_at_freeze": round(float(row.get("age_at_freeze", 0)), 1),
            "n_fuites_total": int(row.get("n_fuites_total", 0)),
        }
        if "cost" in row.index:
            properties["cost"] = round(float(row["cost"]), 0)

        geometry = None
        if geom_column and geom_column in row.index:
            try:
                from shapely import wkt
                from shapely.geometry import mapping

                geom = wkt.loads(row[geom_column])
                geometry = mapping(geom)
            except Exception:
                geometry = None

        # Si pas de géométrie, on met un point null
        if geometry is None:
            geometry = {"type": "Point", "coordinates": [0, 0]}

        features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": geometry,
            }
        )

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "n_features": len(features),
            "scenario": {
                "freeze_date": result.params.freeze_date,
                "horizon_years": result.params.horizon_years,
                "mode": result.params.mode,
            },
        },
    }

    return json.dumps(geojson, indent=2, default=str)


def generate_scenario_report(result) -> str:
    """
    Génère un rapport Markdown du scénario.
    """
    p = result.params
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# Rapport de scénario - Plan de renouvellement",
        f"",
        f"**Généré le :** {now}",
        f"",
        f"## Paramètres du scénario",
        f"",
        f"| Paramètre | Valeur |",
        f"|-----------|--------|",
        f"| Date de gel (freeze_date) | {p.freeze_date} |",
        f"| Horizon de prédiction | {p.horizon_years} an(s) |",
        f"| Mode | {p.mode} |",
        f"| Budget total | {p.budget_total:,.0f} € |",
        f"| Coût/km | {p.cost_per_km:,.0f} €/km |",
    ]

    if p.lineaire_max_km:
        lines.append(f"| Linéaire max | {p.lineaire_max_km:,.1f} km |")

    if p.mode == "baseline":
        lines.append(f"| Mode baseline | {p.baseline_mode} |")
        if p.baseline_mode == "top_k_pct":
            lines.append(f"| Top K% | {p.top_k_pct*100:.0f}% |")
        elif p.baseline_mode == "top_n":
            lines.append(f"| Top N | {p.top_n} |")
        elif p.baseline_mode == "top_length":
            lines.append(f"| Top longueur | {p.top_length_km:.1f} km |")

    if p.excluded_materials:
        lines.append(f"| Matériaux exclus | {', '.join(p.excluded_materials)} |")

    lines.extend([
        f"",
        f"## KPIs de résultat",
        f"",
        f"| KPI | Valeur |",
        f"|-----|--------|",
        f"| Tronçons total | {result.n_total:,} |",
        f"| Tronçons sélectionnés | {result.n_selected:,} |",
        f"| Risque total | {result.risk_total:,.2f} |",
        f"| Risque traité | {result.risk_treated:,.2f} ({result.risk_treated/max(result.risk_total,1)*100:.1f}%) |",
        f"| Risque non traité | {result.risk_untreated:,.2f} ({result.risk_untreated/max(result.risk_total,1)*100:.1f}%) |",
        f"| Budget utilisé | {result.budget_used:,.0f} € |",
        f"| Linéaire couvert | {result.coverage_length_km:,.1f} km ({result.coverage_length_pct:.1f}%) |",
    ])

    if result.solver_status:
        lines.extend([
            f"",
            f"## Informations solveur",
            f"",
            f"- Statut : {result.solver_status}",
            f"- Temps de résolution : {result.solver_time_s:.2f} s",
        ])

    # Distribution des matériaux sélectionnés
    if len(result.df_selected) > 0:
        mat_dist = (
            result.df_selected.groupby("MAT")
            .agg(
                n=("GID", "count"),
                longueur_km=("longueur_km", "sum"),
                score_moy=("risk_score", "mean"),
            )
            .sort_values("longueur_km", ascending=False)
        )

        lines.extend([
            f"",
            f"## Détail par matériau (tronçons sélectionnés)",
            f"",
            f"| Matériau | Nb tronçons | Linéaire (km) | Score moyen |",
            f"|----------|------------|---------------|-------------|",
        ])

        for mat, row in mat_dist.iterrows():
            lines.append(
                f"| {mat} | {int(row['n'])} | {row['longueur_km']:.1f} | {row['score_moy']:.4f} |"
            )

    # Top 20 tronçons sélectionnés
    if len(result.df_selected) > 0:
        top20 = result.df_selected.nlargest(20, "risk_score")
        lines.extend([
            f"",
            f"## Top 20 tronçons sélectionnés",
            f"",
            f"| Rang | GID | Score | MAT | DIAM | LNG (m) | Âge |",
            f"|------|-----|-------|-----|------|---------|-----|",
        ])

        for _, row in top20.iterrows():
            lines.append(
                f"| {int(row.get('rank', 0))} "
                f"| {int(row['GID'])} "
                f"| {row['risk_score']:.4f} "
                f"| {row.get('MAT', '-')} "
                f"| {row.get('DIAMETRE', '-')} "
                f"| {row.get('LNG', 0):.0f} "
                f"| {row.get('age_at_freeze', 0):.1f} |"
            )

    lines.extend([
        f"",
        f"---",
        f"*Rapport généré automatiquement par l'application de plan de renouvellement.*",
    ])

    return "\n".join(lines)
