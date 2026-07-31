#!/usr/bin/env python3
"""
Task 3 - Model Development: Traditional, Deep Learning, and ML.

Trains/tunes the naive baseline, SARIMA (+ optional Prophet), LSTM/GRU, and
XGBoost (+ Random Forest baseline) on the Task 2 splits, walk-forward
evaluates every model over the identical test block, reconstructs price
levels, and writes predictions + tuned hyperparameters for Task 5.

Index convention (important - see src/features.py docstrings): every
model's prediction is stored keyed by the DATE IT IS A FORECAST FOR (not
the date the forecast was made from). Naive and SARIMA are already indexed
that way by construction; return-predicting models (LSTM/GRU, XGBoost, RF,
hybrid) are explicitly relabeled via `features.shift_to_target_dates`
after price reconstruction. This makes the final `predictions_task3`
table directly comparable column-for-column - Task 5 just aligns on the
union of dates and drops the handful of boundary rows a given model
doesn't cover.

Usage
-----
    python scripts/task3_point_forecasting.py
    python scripts/task3_point_forecasting.py --quick                     # fast smoke test
    python scripts/task3_point_forecasting.py --models naive,sarima,xgboost
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# --quick must flip ATSF_QUICK *before* `src.config` (and anything that
# imports it) is loaded, since QUICK_MODE is read once at import time.
if "--quick" in sys.argv:
    os.environ["ATSF_QUICK"] = "1"

import argparse

import numpy as np
import pandas as pd

from src import config, features
from src.models import naive, sarima_model, tree_models
from src.utils import get_logger, save_json, timer

log = get_logger("task3")

ALL_MODELS = ["naive", "sarima", "prophet", "lstm_gru", "xgboost", "random_forest"]
DEFAULT_MODELS = "naive,sarima,lstm_gru,xgboost,random_forest"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="Shrink trials/epochs/refit cadence for a fast smoke test.")
    p.add_argument("--models", type=str, default=DEFAULT_MODELS, help=f"Comma-separated subset of {ALL_MODELS}")
    p.add_argument("--test-limit", type=int, default=None, help="Only walk-forward the first N test rows (debugging).")
    return p.parse_args(argv)


def main(argv=None) -> pd.DataFrame:
    args = parse_args(argv)
    models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models_to_run:
        if m not in ALL_MODELS:
            raise ValueError(f"Unknown model '{m}'. Choose from {ALL_MODELS}.")

    df = pd.read_parquet(config.PROCESSED_DATA_DIR / "prices_clean.parquet")
    train = pd.read_parquet(config.PROCESSED_DATA_DIR / "train.parquet")
    val = pd.read_parquet(config.PROCESSED_DATA_DIR / "val.parquet")
    test = pd.read_parquet(config.PROCESSED_DATA_DIR / "test.parquet")
    train_val = pd.concat([train, val])

    if args.test_limit:
        test = test.iloc[: args.test_limit]

    log.info(
        "train=%d val=%d test=%d | test spans %s -> %s",
        len(train), len(val), len(test), test.index.min().date(), test.index.max().date(),
    )

    predictions: dict[str, pd.Series] = {}
    tuned_params: dict = {}

    # ---------------------------------------------------------------- naive
    if "naive" in models_to_run:
        with timer("naive", log):
            predictions["Naive Random Walk"] = naive.naive_price_forecast(df["Adj Close"], test.index)

    # ---------------------------------------------------------------- sarima
    if "sarima" in models_to_run:
        with timer("SARIMA", log):
            train_val_log_price = df.loc[train_val.index, "log_close"]
            test_log_price = df.loc[test.index, "log_close"]
            arima = sarima_model.fit_auto_arima(train_val_log_price)
            sarima_log_preds = sarima_model.walk_forward_sarima(arima, test_log_price)
            predictions["SARIMA"] = np.exp(sarima_log_preds).rename("SARIMA")
            tuned_params["sarima"] = {
                "order": list(arima.order),
                "seasonal_order": list(getattr(arima, "seasonal_order", (0, 0, 0, 0))),
            }

    # ---------------------------------------------------------------- prophet (optional secondary benchmark)
    if "prophet" in models_to_run:
        try:
            from src.models import prophet_model

            with timer("Prophet", log):
                train_val_log_price = df.loc[train_val.index, "log_close"]
                m = prophet_model.fit_prophet(train_val_log_price)
                prophet_log_preds = prophet_model.predict_prophet(m, test.index)
                predictions["Prophet"] = np.exp(prophet_log_preds).rename("Prophet")
        except ImportError as e:
            log.warning("Skipping Prophet (optional secondary benchmark): %s", e)

    # ---------------------------------------------------------------- lstm/gru
    if "lstm_gru" in models_to_run:
        try:
            from src.models import dl_models

            with timer("LSTM/GRU", log):
                train_ret = df.loc[train.index.min() : train.index.max(), "ret"]
                val_ret = df.loc[val.index.min() : val.index.max(), "ret"]
                full_ret = df["ret"]

                best_params, _study = dl_models.tune_hyperparameters(train_ret, val_ret)
                models, scaler, seed_summary = dl_models.run_multi_seed(train_ret, val_ret, best_params)
                tuned_params["lstm_gru"] = {"best_params": best_params, "seed_summary": seed_summary}

                # Headline series comes from the median-performing seed -
                # a single arbitrary seed would misrepresent the 5-seed
                # spread the blueprint asks us to report (sec. 3.3).
                order = np.argsort(seed_summary["per_seed"])
                chosen_model = models[int(order[len(order) // 2])]

                ret_preds = dl_models.walk_forward_predict_dl(
                    chosen_model, scaler, full_ret, test.index, best_params["lookback"]
                )
                price_preds = features.reconstruct_price_from_returns(df["log_close"], ret_preds)
                predictions["LSTM/GRU"] = features.shift_to_target_dates(price_preds, df.index).rename("LSTM/GRU")
        except ImportError as e:
            log.warning("Skipping LSTM/GRU (tensorflow not installed): %s", e)

    # ---------------------------------------------------------------- xgboost
    if "xgboost" in models_to_run:
        with timer("XGBoost", log):
            xgb_params, _study = tree_models.tune_xgboost(train, val)
            tuned_params["xgboost"] = xgb_params

            fit_fn = lambda h: tree_models.train_xgboost(xgb_params, h)  # noqa: E731
            ret_preds = tree_models.walk_forward_predict_tree(
                fit_fn, train_val, test, refit_every=config.refit_every(config.TREE_REFIT_EVERY)
            )
            price_preds = features.reconstruct_price_from_returns(df["log_close"], ret_preds)
            predictions["XGBoost"] = features.shift_to_target_dates(price_preds, df.index).rename("XGBoost")

            final_model = tree_models.train_xgboost(xgb_params, train_val)
            fi = tree_models.feature_importance(final_model, tree_models.get_feature_columns(train_val))
            fi.to_csv(config.RESULTS_DIR / "xgboost_feature_importance.csv")

            shap_vals = tree_models.shap_summary(final_model, train_val[tree_models.get_feature_columns(train_val)])
            if shap_vals is not None:
                shap_vals.abs().mean().sort_values(ascending=False).to_csv(
                    config.RESULTS_DIR / "xgboost_shap_importance.csv"
                )

    # ---------------------------------------------------------------- random forest
    if "random_forest" in models_to_run:
        with timer("Random Forest", log):
            rf_params, _study = tree_models.tune_random_forest(train, val)
            tuned_params["random_forest"] = rf_params

            fit_fn = lambda h: tree_models.train_random_forest(rf_params, h)  # noqa: E731
            ret_preds = tree_models.walk_forward_predict_tree(
                fit_fn, train_val, test, refit_every=config.refit_every(config.TREE_REFIT_EVERY)
            )
            price_preds = features.reconstruct_price_from_returns(df["log_close"], ret_preds)
            predictions["Random Forest"] = features.shift_to_target_dates(price_preds, df.index).rename("Random Forest")

    # ------------------------------------------------------------- persist
    pred_df = pd.DataFrame(predictions)
    pred_df["Actual"] = df["Adj Close"].reindex(pred_df.index)
    pred_df.to_parquet(config.RESULTS_DIR / "predictions_task3.parquet")
    save_json(tuned_params, config.TUNED_PARAMS_DIR / "task3_tuned_params.json")

    n_common = pred_df.dropna().shape[0]
    log.info(
        "Task 3 predictions saved -> %s (%d rows total, %d with every model present)",
        config.RESULTS_DIR / "predictions_task3.parquet", len(pred_df), n_common,
    )
    return pred_df


if __name__ == "__main__":
    main()
