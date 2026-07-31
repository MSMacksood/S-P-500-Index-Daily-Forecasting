#!/usr/bin/env python3
"""
Runs the full pipeline end to end: Task 2 -> Task 3 -> Task 4 -> Task 5.

Usage
-----
    python scripts/run_all.py                  # full real run (slow: SARIMA
                                                 # order search, LSTM/XGBoost/RF
                                                 # tuning, GARCH walk-forward)
    python scripts/run_all.py --quick           # fast end-to-end smoke test
    python scripts/run_all.py --synthetic       # offline (no yfinance/network)
    python scripts/run_all.py --quick --synthetic --test-limit 60   # fastest possible

Each task can also be run individually (see scripts/task2_preprocessing.py,
task3_point_forecasting.py, task4_volatility.py, task5_evaluation.py) - this
script is a convenience wrapper that chains them with a shared --quick flag.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

if "--quick" in sys.argv:
    os.environ["ATSF_QUICK"] = "1"

import argparse

from src.utils import get_logger, timer

log = get_logger("run_all")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic data instead of yfinance (Task 2).")
    p.add_argument("--test-limit", type=int, default=None, help="Cap the walk-forward test length (debugging).")
    p.add_argument("--models", type=str, default=None, help="Task 3 --models override.")
    p.add_argument("--skip-hybrid", action="store_true")
    p.add_argument("--skip-regime", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import scripts.task2_preprocessing as task2
    import scripts.task3_point_forecasting as task3
    import scripts.task4_volatility as task4
    import scripts.task5_evaluation as task5

    with timer("TASK 2", log):
        task2_argv = ["--synthetic"] if args.synthetic else []
        task2.main(task2_argv)

    with timer("TASK 3", log):
        task3_argv = []
        if args.quick:
            task3_argv.append("--quick")
        if args.models:
            task3_argv += ["--models", args.models]
        if args.test_limit:
            task3_argv += ["--test-limit", str(args.test_limit)]
        task3.main(task3_argv)

    with timer("TASK 4", log):
        task4_argv = []
        if args.quick:
            task4_argv.append("--quick")
        if args.test_limit:
            task4_argv += ["--test-limit", str(args.test_limit)]
        task4.main(task4_argv)

    with timer("TASK 5", log):
        task5_argv = []
        if args.quick:
            task5_argv.append("--quick")
        if args.skip_hybrid:
            task5_argv.append("--skip-hybrid")
        if args.skip_regime:
            task5_argv.append("--skip-regime")
        task5.main(task5_argv)

    log.info("Pipeline complete. See results/ for metrics, figures, and tuned_params/.")


if __name__ == "__main__":
    main()
