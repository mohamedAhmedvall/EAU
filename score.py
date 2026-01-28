#!/usr/bin/env python3
"""
Script de scoring reproductible.

Usage:
  python score.py --assets data/v1_trafic_prepared.csv --anomalies data/historiqueanomalie.csv \
    --freeze-date 2024-01-01 --horizon 1 --output reports/scores.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

from predict_risk import predict_risk  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Score AEP assets for risk.")
    parser.add_argument("--assets", required=True, help="Chemin vers la table patrimoine (CSV).")
    parser.add_argument("--anomalies", required=True, help="Chemin vers la table anomalies (CSV).")
    parser.add_argument("--freeze-date", required=True, help="Date de freeze (YYYY-MM-DD).")
    parser.add_argument("--horizon", type=int, default=1, help="Horizon de prediction en annees.")
    parser.add_argument("--model-path", default=None, help="Chemin vers le modele joblib.")
    parser.add_argument("--output", required=True, help="Chemin de sortie CSV.")
    args = parser.parse_args()

    df_assets = pd.read_csv(args.assets)
    df_anomalies = pd.read_csv(args.anomalies)

    scores = predict_risk(
        df_assets,
        df_anomalies,
        freeze_date=args.freeze_date,
        horizon_years=args.horizon,
        model_path=args.model_path,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"GID": scores.index, "risk_score": scores.values}).to_csv(out, index=False)
    print(f"Scores sauvegardes: {out}")


if __name__ == "__main__":
    main()
