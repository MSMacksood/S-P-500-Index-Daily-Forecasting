#!/usr/bin/env python3
"""
Task 5 - Model Comparison, Insights, and Critical Reflection.

Builds the "Hybrid (+GARCH sigma feature)" model (bonus, referenced in the
blueprint's own Table 3), assembles the final results matrix over every
point-forecast model - including the mandatory naive baseline - with the
Diebold-Mariano test vs naive and a full pairwise DM p-value matrix, scores
the Task 4 volatility models on QLIKE/variance-MSE/Mincer-Zarnowitz against
a naive 21-day benchmark, runs the sec. 5.3 toy long/flat backtest, and (if
VIX data is reachable) buckets errors by VIX tercile.

Requires scripts/task3_point_forecasting.py and scripts/task4_volatility.py
to have already been run (reads their parquet outputs from results/).

Usage
-----
    python scripts/task5_evaluation.py
    python scripts/task5_evaluation.py --quick --skip-hybrid --skip-regime
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
import numpy as np
import pandas as pd

from src import config, evaluate as ev, features
from src.models import garch_models as gm
from src.models import hybrid, tree_models
from src.utils import get_logger, save_json, timer

log = get_logger("task5")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="Shrink trials/refit cadence for a fast smoke test.")
    p.add_argument("--skip-hybrid", action="store_true", help="Skip the bonus GARCH-sigma-feature hybrid model.")
    p.add_argument("--skip-regime", action="store_true", help="Skip the VIX-regime error breakdown.")
    p.add_argument("--cost-bps", type=float, default=5.0, help="One-way transaction cost (bps) for the toy backtest.")
    return p.parse_args(argv)


def build_hybrid_model(df, train, val, test, train_val):
    """Fold the best-BIC GARCH's conditional volatility (in-sample fitted
    values for train+val, walk-forward OOS forecasts for test) into the
    Task 2 feature matrix as `garch_sigma`, then tune/train/walk-forward an
    XGBoost model on top of it exactly like the plain XGBoost model."""
    r_pct = df["ret_pct"]
    # Align to Task 3's actual test-block start (see task4_volatility.py's
    # module docstring for why `int(len(r_pct)*VAL_FRAC)` alone is ~9
    # trading days off from the feature matrix's own split boundary).
    split = df.index.get_loc(test.index[0])

    fitted = gm.fit_garch_family(r_pct.iloc[:split])
    best_name = gm.select_best_by_bic(fitted)
    best_res = fitted[best_name]
    in_sample_sigma = pd.Series(np.asarray(best_res.conditional_volatility), index=r_pct.iloc[:split].index)

    task4_path = config.RESULTS_DIR / "predictions_task4.parquet"
    if not task4_path.exists():
        raise FileNotFoundError(f"{task4_path} not found - run scripts/task4_volatility.py first.")
    task4_preds = pd.read_parquet(task4_path)
    if best_name not in task4_preds.columns:
        raise KeyError(f"{best_name} not in {task4_path} - re-run task4 (make sure it wasn't run with a --test-limit that dropped it).")
    oos_sigma = np.sqrt(task4_preds[best_name].dropna())

    full_sigma_pct = pd.concat([in_sample_sigma, oos_sigma[~oos_sigma.index.isin(in_sample_sigma.index)]]).sort_index()
    log.info("Hybrid model uses %s as the GARCH-sigma feature (%d in-sample + %d OOS points).", best_name, len(in_sample_sigma), len(oos_sigma))

    train_h = hybrid.add_garch_sigma_feature(train, full_sigma_pct)
    val_h = hybrid.add_garch_sigma_feature(val, full_sigma_pct)
    test_h = hybrid.add_garch_sigma_feature(test, full_sigma_pct)
    train_val_h = pd.concat([train_h, val_h])

    params, _study = hybrid.tune_hybrid_xgboost(train_h, val_h)
    fit_fn = lambda h: hybrid.train_hybrid_xgboost(params, h)  # noqa: E731
    ret_preds = hybrid.walk_forward_predict_hybrid(
        fit_fn, train_val_h, test_h, refit_every=config.refit_every(config.TREE_REFIT_EVERY)
    )
    price_preds = features.reconstruct_price_from_returns(df["log_close"], ret_preds)
    hybrid_price = features.shift_to_target_dates(price_preds, df.index).rename("Hybrid (+GARCH sigma)")
    return hybrid_price, params, best_name


def implied_returns(pred_df: pd.DataFrame, df: pd.DataFrame, model_cols) -> tuple[pd.Series, pd.DataFrame]:
    """Back out each model's implied predicted log-return (and the actual
    realized log-return) from its price-level forecast, for the sec. 5.3
    toy backtest - `pred_df` stores price levels only, so this reconstructs
    the return each model was implicitly predicting relative to the last
    actual price before its target date."""
    prev_actual = df["Adj Close"].shift(1).reindex(pred_df.index)
    actual_ret = np.log(pred_df["Actual"]) - np.log(prev_actual)
    pred_rets = {c: np.log(pred_df[c]) - np.log(prev_actual) for c in model_cols}
    return actual_ret, pd.DataFrame(pred_rets)


def main(argv=None):
    args = parse_args(argv)

    pred_path = config.RESULTS_DIR / "predictions_task3.parquet"
    vol_path = config.RESULTS_DIR / "predictions_task4.parquet"
    if not pred_path.exists() or not vol_path.exists():
        raise FileNotFoundError(
            "Run scripts/task3_point_forecasting.py and scripts/task4_volatility.py before task5_evaluation.py."
        )
    pred_df = pd.read_parquet(pred_path)
    vol_df = pd.read_parquet(vol_path)
    model_names = [c for c in pred_df.columns if c != "Actual"]
    log.info("Point-forecast models found: %s", model_names)

    # ------------------------------------------------------------- hybrid
    if not args.skip_hybrid:
        try:
            with timer("Hybrid (+GARCH sigma) model", log):
                df = pd.read_parquet(config.PROCESSED_DATA_DIR / "prices_clean.parquet")
                train = pd.read_parquet(config.PROCESSED_DATA_DIR / "train.parquet")
                val = pd.read_parquet(config.PROCESSED_DATA_DIR / "val.parquet")
                test = pd.read_parquet(config.PROCESSED_DATA_DIR / "test.parquet")
                train_val = pd.concat([train, val])
                hybrid_price, hybrid_params, garch_used = build_hybrid_model(df, train, val, test, train_val)
                pred_df = pred_df.join(hybrid_price, how="left")
                model_names.append(hybrid_price.name)
                save_json(
                    {"garch_variant_used": garch_used, "xgboost_params": hybrid_params},
                    config.TUNED_PARAMS_DIR / "task5_hybrid_params.json",
                )
        except Exception as e:
            log.warning("Skipping hybrid model: %s", e)

    pred_df.to_parquet(config.RESULTS_DIR / "predictions_task3_with_hybrid.parquet")

    # ------------------------------------------------------ comparison table
    with timer("Point-forecast comparison table + DM matrix", log):
        summary, dm_matrix = ev.build_comparison_table(pred_df, model_names)
        summary.to_csv(config.RESULTS_DIR / "metrics_point_forecasts.csv")
        dm_matrix.to_csv(config.RESULTS_DIR / "dm_pvalue_matrix.csv")
        log.info("\n%s", summary.round(4).to_string())

    cse = ev.cumulative_squared_error(pred_df, model_names)
    fig, ax = plt.subplots(figsize=(11, 5))
    cse.plot(ax=ax, lw=1.1)
    ax.set_title("Cumulative squared error over the test block")
    ax.set_ylabel("Cumulative (price - forecast)^2")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "cumulative_squared_error.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pred_df.index, pred_df["Actual"], color="black", lw=1.0, label="Actual")
    for name in model_names:
        ax.plot(pred_df.index, pred_df[name], lw=0.8, alpha=0.85, label=name)
    ax.set_title("Forecast vs actual price - test block")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "forecast_overlay.png", dpi=120)
    plt.close(fig)

    # --------------------------------------------------- volatility models
    with timer("Volatility model comparison", log):
        vol_cols = list(gm.GARCH_SPECS.keys()) + ["Naive 21d Rolling Variance"]
        vol_cols = [c for c in vol_cols if c in vol_df.columns]
        vol_summary = ev.build_volatility_comparison(vol_df, vol_cols, realized_col="sq_ret")
        vol_summary.to_csv(config.RESULTS_DIR / "metrics_volatility.csv")
        log.info("\n%s", vol_summary.round(4).to_string())

    # -------------------------------------------------------- toy backtest
    with timer("Toy long/flat backtest", log):
        actual_ret, pred_rets = implied_returns(pred_df, pd.read_parquet(config.PROCESSED_DATA_DIR / "prices_clean.parquet"), model_names)
        backtest_rows = {}
        for name in model_names:
            backtest_rows[name] = ev.toy_long_flat_backtest(actual_ret, pred_rets[name], cost_bps=args.cost_bps)
            backtest_rows[f"{name} (no cost)"] = ev.toy_long_flat_backtest(actual_ret, pred_rets[name], cost_bps=0.0)
        backtest_df = pd.DataFrame(backtest_rows).T
        backtest_df.to_csv(config.RESULTS_DIR / "toy_backtest.csv")
        log.info("\n%s", backtest_df.round(4).to_string())

    # ------------------------------------------------------- regime bars
    if not args.skip_regime:
        try:
            with timer("VIX-regime error breakdown", log):
                from src import data as data_mod

                vix_cache = config.RAW_DATA_DIR / "VIX_raw.parquet"
                vix_df = data_mod.download_prices(ticker=config.VIX_TICKER, cache_path=vix_cache)
                vix = vix_df["Adj Close"]

                regime_rows = {}
                for name in model_names:
                    sub = pred_df[[name, "Actual"]].dropna()
                    abs_err = (sub["Actual"] - sub[name]).abs()
                    regime_rows[name] = ev.error_by_regime(abs_err, vix)["mean_abs_error"]
                regime_df = pd.DataFrame(regime_rows)
                regime_df.to_csv(config.RESULTS_DIR / "error_by_vix_regime.csv")
                log.info("\n%s", regime_df.round(4).to_string())
        except Exception as e:
            log.warning("Skipping VIX-regime breakdown (network/data unavailable): %s", e)

    log.info("Task 5 complete. Results written to %s", config.RESULTS_DIR)
    return summary, dm_matrix, vol_summary


if __name__ == "__main__":
    main()
