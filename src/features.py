"""
Task 2.4 / 3.1 - Leakage-safe feature matrix (Listing 2.3) and the strict
chronological 70/15/15 split (Listing 3.1).

Every column is built only from information available strictly before the
date it occupies, and the prediction target sits `horizon` steps ahead of
that date. See `_calendar_features` for one deliberate, documented
departure from the blueprint's illustrative Listing 2.3.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src import config


def _lag_features(ret: pd.Series, lags: Sequence[int]) -> pd.DataFrame:
    """ret_lag{k}[t] = ret[t-k] - known at t, so safe for a target at t+H."""
    return pd.DataFrame({f"ret_lag{k}": ret.shift(k) for k in lags})


def _rolling_features(ret: pd.Series, windows: Sequence[int]) -> pd.DataFrame:
    """Rolling mean/std computed through t-1 (the trailing .shift(1) means
    the window ending "today" is never included), then aligned to row t."""
    out = {}
    for w in windows:
        out[f"roll_mean_{w}"] = ret.rolling(w).mean().shift(1)
        out[f"roll_std_{w}"] = ret.rolling(w).std().shift(1)
    return pd.DataFrame(out)


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Day-of-week / month / turn-of-month / COVID-window dummy.

    NOTE ON SHIFTING: the blueprint's illustrative Listing 2.3 builds these
    from `df.index` unshifted. The project's own critical constraints are
    stricter than the listing here ("... calendar indicators ... must be
    strictly shifted by at least one time step to prevent future data
    leakage") and take precedence, so every calendar column below is
    shifted by one trading day before being used, exactly like every
    numeric feature. This costs nothing in practice - calendar attributes
    are deterministic and can never leak information either way - but it
    keeps a single, uniformly-verifiable contract for the whole feature
    matrix: "row t uses only information dated on or before t-1".
    """
    dow = pd.Series(index.dayofweek, index=index, name="dow")
    month = pd.Series(index.month, index=index, name="month")
    tom = pd.Series(
        index.day.isin([*range(28, 32), 1, 2, 3]).astype(int), index=index, name="tom"
    )
    covid = pd.Series(
        ((index >= config.COVID_START) & (index <= config.COVID_END)).astype(int),
        index=index,
        name="covid",
    )
    cal = pd.concat([dow, month, tom, covid], axis=1)
    return cal.shift(1)


def build_feature_matrix(
    df: pd.DataFrame,
    horizon: int = config.FORECAST_HORIZON,
    lags: Sequence[int] = config.RET_LAGS,
    roll_windows: Sequence[int] = config.ROLL_WINDOWS,
    momentum_window: int = config.MOMENTUM_WINDOW,
) -> pd.DataFrame:
    """Build the Listing 2.3 feature matrix. `y` is the `horizon`-day-ahead
    log-return; returns a single NaN-dropped DataFrame of features + `y`.

    Parameters
    ----------
    df: output of `data.add_log_returns` (must contain ret, Adj Close, High,
        Low, Close).
    horizon: forecast horizon in trading days. 1 = primary target; 5 =
        blueprint's secondary "1 trading week ahead" horizon (sec. 1.2).
    """
    feat = pd.DataFrame(index=df.index)
    feat = feat.join(_lag_features(df["ret"], lags))
    feat = feat.join(_rolling_features(df["ret"], roll_windows))

    feat["mom_63"] = (df["Adj Close"] / df["Adj Close"].shift(momentum_window) - 1).shift(1)
    feat["hl_range"] = ((df["High"] - df["Low"]) / df["Close"]).shift(1)

    feat = feat.join(_calendar_features(df.index))

    y = df["ret"].shift(-horizon).rename("y")
    data = feat.join(y).dropna()
    return data


def chronological_split(
    data: pd.DataFrame,
    train_frac: float = config.TRAIN_FRAC,
    val_frac: float = config.VAL_FRAC,
):
    """Strict, un-shuffled 70/15/15 split (Listing 3.1). No cross-validation
    shuffling anywhere; use TimeSeriesSplit for any CV inside the training
    block if needed."""
    n = len(data)
    i_tr, i_va = int(n * train_frac), int(n * val_frac)
    train, val, test = data.iloc[:i_tr], data.iloc[i_tr:i_va], data.iloc[i_va:]
    assert train.index.max() < val.index.min(), "train/val overlap or misorder"
    assert val.index.max() < test.index.min(), "val/test overlap or misorder"
    return train, val, test


def align_price_series(df: pd.DataFrame, target_index: pd.Index, price_col: str = "Adj Close") -> pd.Series:
    """Reindex the raw price series onto the feature matrix's index (i.e.
    the price *at* each feature row's own date `t`) so every model is
    scored against the same set of "as-of" dates."""
    return df.loc[target_index, price_col]


def future_price(df: pd.DataFrame, target_index: pd.Index, horizon: int = config.FORECAST_HORIZON,
                  price_col: str = "Adj Close") -> pd.Series:
    """The *actual* price `horizon` steps ahead of each feature row's date -
    i.e. the ground truth that a reconstructed price forecast should be
    compared against. Equivalent to `y` in price-level terms."""
    return df[price_col].shift(-horizon).reindex(target_index)


def target_dates(origin_index: pd.DatetimeIndex, full_index: pd.DatetimeIndex, horizon: int = config.FORECAST_HORIZON) -> pd.DatetimeIndex:
    """The calendar dates `horizon` trading days after each date in
    `origin_index`, looked up positionally against the complete trading
    index `full_index` (e.g. `df.index`)."""
    positions = full_index.get_indexer(origin_index)
    if (positions < 0).any():
        raise KeyError("Some dates in origin_index are not present in full_index.")
    target_positions = positions + horizon
    if (target_positions >= len(full_index)).any():
        raise IndexError(f"horizon={horizon} runs past the end of full_index for some dates.")
    return full_index[target_positions]


def shift_to_target_dates(series: pd.Series, full_index: pd.DatetimeIndex, horizon: int = config.FORECAST_HORIZON) -> pd.Series:
    """Relabel a series indexed by *origin* date t (a forecast made using
    information through t) onto the *target* date t+horizon it is actually
    a forecast for.

    This is the convention used for the final model-comparison table: row =
    the date being forecast, matching the naive benchmark's natural
    `yhat_t = y_{t-1}` indexing and SARIMA's walk-forward loop, both of
    which are target-date-indexed by construction (their "history" ends
    exactly one step before the date they are about to forecast, so no
    relabeling is needed for them - only for the return-predicting models,
    whose feature-matrix rows are naturally origin-indexed).
    """
    new_index = target_dates(series.index, full_index, horizon)
    return pd.Series(series.values, index=new_index, name=series.name)


def reconstruct_price_from_returns(
    anchor_log_price: pd.Series, predicted_return: pd.Series
) -> pd.Series:
    """P_hat[t+H] = exp(log P[t] + r_hat) - Task 3.2/3.3's price
    reconstruction, so returns-based models (LSTM/GRU, XGBoost, RF, hybrid)
    are comparable to SARIMA/naive on price level (MAPE is defined on price
    levels only).

    `anchor_log_price` should be `df["log_close"]` **unshifted** - row `t`'s
    target `y[t] = ret[t+H]` is the *single* day's return realized on
    t+H (Listing 2.3), so the correct anchor is today's own log-price,
    `log_close[t]`, not yesterday's. This reconstruction is exact for the
    primary horizon H=1; it is intentionally not used for the secondary
    5-day horizon (see `build_cumulative_return_target`).

    The result stays indexed by the *feature* date `t`; compare it against
    `future_price(df, index, horizon=H)`, which is aligned the same way.
    """
    anchor = anchor_log_price.reindex(predicted_return.index)
    return np.exp(anchor + predicted_return)


def build_cumulative_return_target(df: pd.DataFrame, horizon: int = config.SECONDARY_HORIZON) -> pd.Series:
    """Direct target for the blueprint's *secondary* horizon (sec. 1.2: "5
    trading days ahead, forecast recursively/directly"). Unlike the primary
    `y = ret.shift(-1)`, a single day's return H days out is not a useful
    "H-day-ahead" target, so this instead sums the H individual daily
    returns between t and t+H: `Y[t] = log P[t+H] - log P[t]`, the exact
    cumulative log-return a direct H-day model should predict, and which
    reconstructs with the same `reconstruct_price_from_returns` anchor
    (`log_close[t]`, unshifted) used for H=1.
    """
    return (df["log_close"].shift(-horizon) - df["log_close"]).rename("y_cum")
