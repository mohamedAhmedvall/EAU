#!/usr/bin/env python3
"""
Pipeline d'évaluation reproductible.

Usage:
  python evaluate.py --phase phase1
  python evaluate.py --phase phase2 --horizon 3
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

from phase1_model_evaluation import main as phase1_main  # noqa: E402
from phase2_full import main as phase2_main  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation phases.")
    parser.add_argument(
        "--phase",
        choices=["phase1", "phase2"],
        default="phase1",
        help="Phase a executer.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Horizon en annees (utilise pour phase2).",
    )
    args = parser.parse_args()

    if args.phase == "phase1":
        phase1_main()
    else:
        phase2_main(args.horizon)


if __name__ == "__main__":
    main()
