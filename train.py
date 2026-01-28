#!/usr/bin/env python3
"""
Pipeline d'entraînement reproductible.

Usage:
  python train.py --horizons 1,3,5 --build-dataset
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

import pandas as pd

SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config import HORIZONS, ARTIFACTS_DIR, REPORTS_DIR  # noqa: E402
from dataset_builder import load_raw_data, build_dataset_for_horizon  # noqa: E402
from modeling import run_experiment, save_best_model, create_benchmark_table  # noqa: E402


def parse_horizons(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def build_datasets(horizons: List[int]) -> None:
    df_trafic, df_anomalies, metadata = load_raw_data()
    max_obs_date = pd.Timestamp(metadata["max_observation_date"])
    for h in horizons:
        df = build_dataset_for_horizon(df_trafic, df_anomalies, max_obs_date, h)
        df.to_csv(ARTIFACTS_DIR / f"dataset_freeze_h{h}.csv", index=False)
        df.to_pickle(ARTIFACTS_DIR / f"dataset_freeze_h{h}.pkl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train models across horizons.")
    parser.add_argument(
        "--horizons",
        type=parse_horizons,
        default=HORIZONS,
        help="Liste d'horizons (ex: 1,3,5).",
    )
    parser.add_argument(
        "--build-dataset",
        action="store_true",
        help="Reconstruit les datasets freeze+horizon avant l'entraînement.",
    )
    args = parser.parse_args()

    horizons = args.horizons
    if args.build_dataset:
        build_datasets(horizons)

    all_results = {}
    for horizon in horizons:
        exp = run_experiment(horizon)
        all_results[horizon] = exp
        save_best_model(exp, horizon)

    benchmark = create_benchmark_table(all_results)
    benchmark_path = REPORTS_DIR / "benchmark_results.csv"
    benchmark.to_csv(benchmark_path, index=False)
    print(f"\nBenchmark sauvegarde: {benchmark_path}")

    best_row = benchmark.loc[benchmark["Lift@10%"].idxmax()]
    print("\nMeilleur modele global:")
    print(best_row.to_string())


if __name__ == "__main__":
    main()
