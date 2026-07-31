"""
Task 3.3 - LSTM/GRU (Listing 3.3) on scaled returns: early stopping on the
validation block, Optuna hyperparameter tuning, and 5-seed reporting (the
blueprint's stability requirement - "single-seed deep-learning results on
this series are unstable enough to flip model rankings"). Predictions are
reconstructed to price level via `features.reconstruct_price_from_returns`
for comparability with SARIMA/naive.

TensorFlow is an optional (if heavy) dependency: every public function
raises a clear ImportError if it is missing, rather than making this whole
module - and anything that imports it - unusable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config
from src.utils import get_logger, set_global_seed

log = get_logger(__name__)

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False


def _require_tf() -> None:
    if not TF_AVAILABLE:
        raise ImportError(
            "tensorflow is not installed. Install it (`pip install tensorflow`) "
            "to run the LSTM/GRU model, or drop it from --models on "
            "scripts/task3_point_forecasting.py."
        )


def make_windows(series_1d: np.ndarray, lookback: int):
    """Listing 3.3's `windows` helper: turns a 1-D array into overlapping
    (lookback,)-length sequences and their next-step targets.

    `X[k] = series[k : k+lookback]` predicts `y[k] = series[k+lookback]`,
    i.e. the lookback window INCLUDES the most recent observation available
    (position `k+lookback-1`) and predicts the very next one - the same
    "no unnecessary gap" convention Listing 3.3 uses (note this differs
    from the tree feature matrix's lag-1..21 convention in Listing 2.3,
    which deliberately starts at lag 1; each is implemented faithfully to
    its own listing).
    """
    if len(series_1d) <= lookback:
        raise ValueError(f"Series length {len(series_1d)} must exceed lookback {lookback}.")
    X = np.stack([series_1d[i - lookback : i] for i in range(lookback, len(series_1d))])
    y = series_1d[lookback:]
    return X[..., None], y


def build_model(units=64, layers=2, dropout=0.2, lr=1e-3, cell="LSTM", lookback=60):
    """Listing 3.3's `build` helper."""
    _require_tf()
    cell_layer = tf.keras.layers.LSTM if cell == "LSTM" else tf.keras.layers.GRU
    m = tf.keras.Sequential([tf.keras.Input((lookback, 1))])
    for i in range(layers):
        m.add(cell_layer(units, return_sequences=(i < layers - 1)))
        m.add(tf.keras.layers.Dropout(dropout))
    m.add(tf.keras.layers.Dense(1))
    m.compile(tf.keras.optimizers.Adam(lr), loss="mse")
    return m


@dataclass
class DLDataset:
    scaler: StandardScaler
    Xtr: np.ndarray
    ytr: np.ndarray
    Xva: np.ndarray
    yva: np.ndarray
    lookback: int


def prepare_dataset(train_ret: pd.Series, val_ret: pd.Series, lookback: int) -> DLDataset:
    """StandardScaler fit on TRAIN ONLY (Listing 3.3), applied to val too.
    The val window is built by prepending the tail of train so the first
    `lookback` val points are not wasted.
    """
    scaler = StandardScaler().fit(train_ret.to_frame())
    train_scaled = scaler.transform(train_ret.to_frame()).ravel()
    val_scaled = scaler.transform(val_ret.to_frame()).ravel()

    Xtr, ytr = make_windows(train_scaled, lookback)

    val_input = np.concatenate([train_scaled[-lookback:], val_scaled])
    Xva, yva = make_windows(val_input, lookback)

    return DLDataset(scaler=scaler, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, lookback=lookback)


def train_model(model, ds: DLDataset, epochs=None, batch_size=None, patience=None, verbose=0):
    """Fit with EarlyStopping on the validation block (Listing 3.3)."""
    _require_tf()
    epochs = config.n_epochs(config.LSTM_MAX_EPOCHS if epochs is None else epochs)
    batch_size = batch_size or config.LSTM_BATCH_SIZE
    patience = config.LSTM_PATIENCE if patience is None else patience

    callback = tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
    return model.fit(
        ds.Xtr, ds.ytr,
        validation_data=(ds.Xva, ds.yva),
        epochs=epochs, batch_size=batch_size,
        callbacks=[callback], verbose=verbose,
    )


def tune_hyperparameters(
    train_ret: pd.Series, val_ret: pd.Series, n_trials: Optional[int] = None, seed: int = config.DEFAULT_SEED
):
    """Optuna tuning (objective = val RMSE) over the exact grid in sec. 3.3:
    units, layers, dropout, lookback, lr, cell."""
    _require_tf()
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_trials = config.n_trials(config.LSTM_N_TUNING_TRIALS if n_trials is None else n_trials)

    def objective(trial: "optuna.Trial") -> float:
        lookback = trial.suggest_categorical("lookback", list(config.LSTM_LOOKBACK_OPTIONS))
        units = trial.suggest_categorical("units", list(config.LSTM_UNITS_OPTIONS))
        layers = trial.suggest_categorical("layers", list(config.LSTM_LAYERS_OPTIONS))
        dropout = trial.suggest_categorical("dropout", list(config.LSTM_DROPOUT_OPTIONS))
        lr = trial.suggest_categorical("lr", list(config.LSTM_LR_OPTIONS))
        cell = trial.suggest_categorical("cell", list(config.LSTM_CELL_OPTIONS))

        set_global_seed(seed)
        ds = prepare_dataset(train_ret, val_ret, lookback)
        model = build_model(units=units, layers=layers, dropout=dropout, lr=lr, cell=cell, lookback=lookback)
        train_model(model, ds, epochs=config.n_epochs(60), patience=8, verbose=0)
        val_pred = model.predict(ds.Xva, verbose=0).ravel()
        return float(np.sqrt(np.mean((ds.yva - val_pred) ** 2)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info("Best LSTM/GRU params: %s (val RMSE=%.6f)", study.best_params, study.best_value)
    return study.best_params, study


def run_multi_seed(train_ret: pd.Series, val_ret: pd.Series, best_params: dict, seeds=config.RANDOM_SEEDS):
    """Report mean +/- std over multiple random seeds (sec. 3.3)."""
    _require_tf()
    lookback = best_params["lookback"]
    ds = prepare_dataset(train_ret, val_ret, lookback)

    val_rmses, models = [], []
    for seed in seeds:
        set_global_seed(seed)
        model = build_model(
            units=best_params["units"], layers=best_params["layers"],
            dropout=best_params["dropout"], lr=best_params["lr"],
            cell=best_params["cell"], lookback=lookback,
        )
        train_model(model, ds, verbose=0)
        val_pred = model.predict(ds.Xva, verbose=0).ravel()
        rmse = float(np.sqrt(np.mean((ds.yva - val_pred) ** 2)))
        val_rmses.append(rmse)
        models.append(model)
        log.info("  seed=%d val RMSE=%.6f", seed, rmse)

    summary = dict(
        mean_val_rmse=float(np.mean(val_rmses)), std_val_rmse=float(np.std(val_rmses)), per_seed=val_rmses
    )
    log.info("LSTM/GRU %d-seed val RMSE: %.6f +/- %.6f", len(seeds), summary["mean_val_rmse"], summary["std_val_rmse"])
    return models, ds.scaler, summary


def walk_forward_predict_dl(
    model, scaler: StandardScaler, full_ret: pd.Series, test_index: pd.Index, lookback: int
) -> pd.Series:
    """One-step-ahead walk-forward *inference* over the test block: at each
    test date `t`, feed the model the `lookback` most recent ACTUAL returns
    up to and including `t` (already known at that point) and predict the
    return realized at `t+1`. The model's weights are frozen (no refitting
    inside this loop) - standard practice for evaluating an already-trained
    sequence model walk-forward; all `test_index` windows are batched into
    a single `model.predict()` call for speed.
    """
    _require_tf()
    scaled = pd.Series(scaler.transform(full_ret.to_frame()).ravel(), index=full_ret.index)

    windows, valid_dates = [], []
    for date in test_index:
        w = scaled.loc[:date].iloc[-lookback:]
        if len(w) == lookback:
            windows.append(w.values)
            valid_dates.append(date)

    if not windows:
        return pd.Series(np.nan, index=test_index, name="lstm_ret")

    X = np.stack(windows)[..., None]
    preds_scaled = model.predict(X, verbose=0).ravel()
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    return pd.Series(preds, index=pd.Index(valid_dates), name="lstm_ret").reindex(test_index)
