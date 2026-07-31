#!/usr/bin/env python3
"""
Task 4 - Volatility Modeling with GARCH.

Fits GARCH(1,1), GJR-GARCH, and EGARCH (Student-t errors) on train+val
percent-returns - the same 85% cumulative split boundary Task 3 uses
(Listing 4.1: "align test block with Task 3") - then produces efficient
rolling 1-day-ahead variance forecasts over the identical test block,
alongside a naive 21-day rolling-variance benchmark every GARCH variant
must beat.

Usage
-----
    python scripts/task4_volatility.py
    python scripts/task4_volatility.py --quick             # fast smoke test
    python scripts/task4_volatility.py --refit-every 1     # literal Listing 4.1 (slow)
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.models import garch_models as gm
from src.utils import get_logger, save_json, timer

log = get_logger("task4")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="Widen the refit cadence for a fast smoke test.")
    p.add_argument("--refit-every", type=int, default=None, help=f"Override refit cadence (default {config.GARCH_REFIT_EVERY}).")
    p.add_argument("--test-limit", type=int, default=None, help="Only forecast the first N test rows (debugging).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    refit_every = args.refit_every or config.refit_every(config.GARCH_REFIT_EVERY)

    df = pd.read_parquet(config.PROCESSED_DATA_DIR / "prices_clean.parquet")
    r = df["ret_pct"]

    # Align the split to Task 3's *actual* test-block start date rather than
    # independently recomputing `int(len(r)*0.85)` on the raw price series:
    # the feature matrix (what Task 3 splits) drops a warmup window at the
    # front, so the two "85%" points land ~9 trading days apart if computed
    # separately - Listing 4.1's own comment says to "align test block with
    # Task 3", so we read Task 3's boundary directly.
    test_path = config.PROCESSED_DATA_DIR / "test.parquet"
    if test_path.exists():
        test_start = pd.read_parquet(test_path).index[0]
        split = df.index.get_loc(test_start)
    else:
        log.warning("data/processed/test.parquet not found - falling back to int(len(r)*0.85); run scripts/task2_preprocessing.py first to align exactly with Task 3.")
        split = int(len(r) * config.VAL_FRAC)

    r_tr, r_te = r.iloc[:split], r.iloc[split:]
    if args.test_limit:
        r_te = r_te.iloc[: args.test_limit]
    log.info(
        "GARCH fit window: %d obs (%s -> %s) | test window: %d obs (%s -> %s) | refit_every=%d",
        len(r_tr), r_tr.index.min().date(), r_tr.index.max().date(),
        len(r_te), r_te.index.min().date(), r_te.index.max().date(), refit_every,
    )

    with timer("Fit GARCH(1,1) / GJR-GARCH / EGARCH", log):
        fitted = gm.fit_garch_family(r_tr)
        best_name = gm.select_best_by_bic(fitted)
        log.info("Lowest-BIC model: %s", best_name)

    param_summary = {name: gm.summarize_params(res) for name, res in fitted.items()}
    param_summary["_selected_by_bic"] = best_name

    forecasts = {}
    with timer("Rolling 1-day-ahead variance forecasts (all 3 variants)", log):
        r_full_for_test = pd.concat([r_tr, r_te])
        for name, spec in gm.GARCH_SPECS.items():
            log.info("  walk-forward: %s", name)
            forecasts[name] = gm.rolling_variance_forecast(
                r_full_for_test, split, spec, refit_every=refit_every
            )

    forecasts["Naive 21d Rolling Variance"] = gm.naive_rolling_variance_benchmark(
        pd.concat([r_tr, r_te]), split
    )

    realized = gm.realized_variance_proxies(r)

    out = pd.DataFrame(forecasts)
    out = out.join(realized, how="left")
    out.to_parquet(config.RESULTS_DIR / "predictions_task4.parquet")
    save_json(param_summary, config.TUNED_PARAMS_DIR / "task4_garch_params.json")

    _plot_forecast_vs_realized(out, best_name)

    log.info("Task 4 predictions saved -> %s", config.RESULTS_DIR / "predictions_task4.parquet")
    return out, param_summary


def _plot_forecast_vs_realized(out: pd.DataFrame, best_name: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    realized_vol = out["sq_ret"].rolling(5).mean().pow(0.5)
    ax.plot(out.index, realized_vol, label="Realized vol (5d avg of |r|, rolling)", color="gray", lw=0.8, alpha=0.7)
    for name in gm.GARCH_SPECS:
        ax.plot(out.index, out[name] ** 0.5, label=f"{name} forecast vol", lw=1.0)
    ax.plot(out.index, out["Naive 21d Rolling Variance"] ** 0.5, label="Naive 21d", lw=0.8, ls="--", color="black")
    ax.set_title(f"Forecast vs realized volatility (test block) - lowest BIC: {best_name}")
    ax.set_ylabel("Daily vol (%, return scale)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "garch_forecast_vs_realized.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
