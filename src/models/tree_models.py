"""
Task 3.4 - XGBoost with Optuna tuning (Listing 3.4) on the Task 2 feature
matrix, plus a Random Forest sanity baseline with equivalent tuning.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from src import config
from src.utils import get_logger

log = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

TARGET_COLUMN = "y"


def get_feature_columns(frame: pd.DataFrame) -> List[str]:
    return [c for c in frame.columns if c != TARGET_COLUMN]


def tune_xgboost(
    train: pd.DataFrame, val: pd.DataFrame, n_trials: Optional[int] = None, seed: int = config.DEFAULT_SEED
):
    """Optuna tuning exactly per Listing 3.4's search space."""
    cols = get_feature_columns(train)
    Xtr, ytr = train[cols].values, train[TARGET_COLUMN].values
    Xva, yva = val[cols].values, val[TARGET_COLUMN].values
    n_trials = config.n_trials(config.XGB_N_TRIALS if n_trials is None else n_trials)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1500),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("lr", 1e-3, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample", 0.5, 1.0),
            min_child_weight=trial.suggest_int("mcw", 1, 20),
            reg_lambda=trial.suggest_float("lambda", 1e-3, 10, log=True),
        )
        model = xgb.XGBRegressor(**params, tree_method="hist", random_state=seed)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return float(mean_squared_error(yva, model.predict(Xva)) ** 0.5)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    log.info("Best XGBoost params: %s (val RMSE=%.6f)", study.best_params, study.best_value)
    return _map_xgb_params(study.best_params), study


def _map_xgb_params(raw: dict) -> dict:
    """Optuna trial param names (lr/mcw/colsample/lambda) -> XGBRegressor kwargs."""
    return dict(
        n_estimators=raw["n_estimators"], max_depth=raw["max_depth"],
        learning_rate=raw["lr"], subsample=raw["subsample"],
        colsample_bytree=raw["colsample"], min_child_weight=raw["mcw"],
        reg_lambda=raw["lambda"],
    )


def train_xgboost(params: dict, train: pd.DataFrame, seed: int = config.DEFAULT_SEED) -> xgb.XGBRegressor:
    cols = get_feature_columns(train)
    model = xgb.XGBRegressor(**params, tree_method="hist", random_state=seed)
    model.fit(train[cols].values, train[TARGET_COLUMN].values, verbose=False)
    return model


def feature_importance(model: xgb.XGBRegressor, feature_names: List[str]) -> pd.DataFrame:
    """Gain-based importance (sec. 3.4): expect rolling volatility + short
    lags to dominate, calendar features marginal."""
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    mapped = {
        (feature_names[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k): v
        for k, v in gain.items()
    }
    s = pd.Series(mapped).reindex(feature_names).fillna(0.0).sort_values(ascending=False)
    return s.rename("gain").to_frame()


def shap_summary(model: xgb.XGBRegressor, X: pd.DataFrame, max_samples: int = 2000):
    """SHAP values (sec. 3.4) - guarded: shap is an optional dependency, and
    (being a fast-moving library) occasionally breaks against the newest
    xgboost release; either failure degrades gracefully rather than
    crashing the run, since gain-based `feature_importance()` already
    covers the same "which features matter" question.
    """
    try:
        import shap

        sample = X.sample(min(max_samples, len(X)), random_state=config.DEFAULT_SEED)
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(sample)
        return pd.DataFrame(values, columns=X.columns, index=sample.index)
    except ImportError:
        log.warning("shap not installed - skipping SHAP importance (gain-based feature_importance() still works).")
        return None
    except Exception as e:  # pragma: no cover - defensive: shap/xgboost version skew
        log.warning("shap failed (%s) - skipping SHAP importance (gain-based feature_importance() still works).", e)
        return None


def tune_random_forest(
    train: pd.DataFrame, val: pd.DataFrame, n_trials: Optional[int] = None, seed: int = config.DEFAULT_SEED
):
    """RandomForestRegressor with equivalent tuning - sec. 3.4's "required
    alternative/baseline check"."""
    cols = get_feature_columns(train)
    Xtr, ytr = train[cols].values, train[TARGET_COLUMN].values
    Xva, yva = val[cols].values, val[TARGET_COLUMN].values
    n_trials = config.n_trials(config.RF_N_TRIALS if n_trials is None else n_trials)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 50),
            max_features=trial.suggest_float("max_features", 0.3, 1.0),
        )
        model = RandomForestRegressor(**params, n_jobs=-1, random_state=seed)
        model.fit(Xtr, ytr)
        return float(mean_squared_error(yva, model.predict(Xva)) ** 0.5)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    log.info("Best RandomForest params: %s (val RMSE=%.6f)", study.best_params, study.best_value)
    return study.best_params, study


def train_random_forest(params: dict, train: pd.DataFrame, seed: int = config.DEFAULT_SEED) -> RandomForestRegressor:
    cols = get_feature_columns(train)
    model = RandomForestRegressor(**params, n_jobs=-1, random_state=seed)
    model.fit(train[cols].values, train[TARGET_COLUMN].values)
    return model


def walk_forward_predict_tree(
    fit_fn, train_val_frame: pd.DataFrame, test_frame: pd.DataFrame, refit_every: int = config.TREE_REFIT_EVERY
) -> pd.Series:
    """Periodic-refit walk-forward for any sklearn-API model (sec. 3.1:
    "... refit ... at fixed intervals, e.g., every 21 days...").

    `fit_fn(frame) -> fitted_model` encapsulates a specific model + its
    already-chosen hyperparameters (see scripts/task3_point_forecasting.py).
    `train_val_frame` should already be train+val combined, so the model's
    history has no gap right up to the first test date.
    """
    from src.walkforward import refit_checkpoints

    cols = get_feature_columns(test_frame)
    history = train_val_frame.copy()
    chunks = []

    for start, end in refit_checkpoints(len(test_frame), refit_every):
        model = fit_fn(history)
        block = test_frame.iloc[start:end]
        chunks.append(pd.Series(model.predict(block[cols].values), index=block.index))
        history = pd.concat([history, block])

    return pd.concat(chunks).rename("pred")
