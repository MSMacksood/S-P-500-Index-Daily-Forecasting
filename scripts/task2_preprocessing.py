#!/usr/bin/env python3
"""
Task 2 - Time Series Exploration and Preprocessing.

Acquires ^GSPC (Listing 2.1), runs the full diagnostic suite (Listings 2.2
and 2.4), builds the leakage-safe feature matrix (Listing 2.3), and writes
the strict chronological 70/15/15 split (Listing 3.1) to data/processed/
for Tasks 3-5 to consume.

Usage
-----
    python scripts/task2_preprocessing.py
    python scripts/task2_preprocessing.py --synthetic          # offline smoke test
    python scripts/task2_preprocessing.py --force-refresh
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config, data, diagnostics, features
from src.utils import get_logger, save_json, timer

log = get_logger("task2")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthetic", action="store_true",
                    help="Use a synthetic OHLCV series instead of yfinance (offline smoke test).")
    p.add_argument("--force-refresh", action="store_true", help="Re-download even if a cache exists.")
    p.add_argument("--horizon", type=int, default=config.FORECAST_HORIZON, help="Forecast horizon in trading days.")
    p.add_argument("--skip-diagnostics", action="store_true", help="Skip STL/ACF/stationarity plots (faster).")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)

    with timer("Task 2: acquisition", log):
        df = data.load_or_download(force_refresh=args.force_refresh, synthetic=args.synthetic)
        log.info("Loaded %d rows spanning %s -> %s", len(df), df.index.min().date(), df.index.max().date())

    with timer("Task 2: validation + anomaly screening", log):
        calendar_report = data.validate_calendar(df)
        df = data.flag_anomalies(df)
        n_anom = int(df["anomaly_flag"].sum())
        log.info("%d anomalous (|z|>%.1f) return days retained (not removed)", n_anom, config.ANOMALY_Z_THRESH)

    diagnostics_report = {}
    if not args.skip_diagnostics:
        with timer("Task 2: STL / ACF / stationarity / ARCH-LM diagnostics", log):
            diagnostics_report = diagnostics.run_full_diagnostics(df)

    with timer("Task 2: feature matrix + chronological split", log):
        feat = features.build_feature_matrix(df, horizon=args.horizon)
        train, val, test = features.chronological_split(feat)
        log.info(
            "Feature matrix: %d rows x %d cols | train=%d val=%d test=%d (%.1f/%.1f/%.1f%%)",
            *feat.shape, len(train), len(val), len(test),
            100 * len(train) / len(feat), 100 * len(val) / len(feat), 100 * len(test) / len(feat),
        )

    with timer("Task 2: persisting processed artifacts", log):
        df.to_parquet(config.PROCESSED_DATA_DIR / "prices_clean.parquet")
        feat.to_parquet(config.PROCESSED_DATA_DIR / "features_full.parquet")
        train.to_parquet(config.PROCESSED_DATA_DIR / "train.parquet")
        val.to_parquet(config.PROCESSED_DATA_DIR / "val.parquet")
        test.to_parquet(config.PROCESSED_DATA_DIR / "test.parquet")

    summary = {
        "n_rows_raw": int(len(df)),
        "date_range": [str(df.index.min().date()), str(df.index.max().date())],
        "n_anomalies_flagged": n_anom,
        "calendar_validation": calendar_report,
        "feature_matrix_shape": list(feat.shape),
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "diagnostics": diagnostics_report,
        "synthetic_data": bool(args.synthetic),
        "horizon": args.horizon,
    }
    save_json(summary, config.RESULTS_DIR / "task2_summary.json")
    log.info("Task 2 complete. Summary -> %s", config.RESULTS_DIR / "task2_summary.json")
    return summary


if __name__ == "__main__":
    main()
