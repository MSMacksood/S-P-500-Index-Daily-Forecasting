"""
Mandatory benchmark (Task 3 / sec. 3.5) - naive random walk: yhat_t = y_{t-1}
on the PRICE level. Every other model must beat this to claim skill.
"""
from __future__ import annotations

import pandas as pd


def naive_price_forecast(full_price_series: pd.Series, target_index: pd.Index) -> pd.Series:
    """yhat[t] = price[t-1]: tomorrow's price is predicted to equal today's
    actual price ("no change").

    Pass the FULL history (e.g. `df["Adj Close"]`), not a pre-sliced test
    block, so the first test-block prediction can borrow the last
    training-period actual value.
    """
    return full_price_series.shift(1).reindex(target_index).rename("naive")


def naive_return_forecast(target_index: pd.Index) -> pd.Series:
    """Return-space equivalent of the price random walk: a constant
    zero predicted return (i.e. "price unchanged")."""
    return pd.Series(0.0, index=target_index, name="naive_ret")
