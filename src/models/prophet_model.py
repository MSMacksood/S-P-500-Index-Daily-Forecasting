"""
Task 3.2 - Prophet secondary benchmark (sec. 3.2; optional, guarded import).

Prophet assumes continuous time and handles calendar gaps (weekends,
holidays) natively via its `ds` column, so - unlike SARIMA/ARIMA on the
native trading-day index - training data can be left with gaps in place;
forecasts must only ever be requested for actual future trading days.
"""
from __future__ import annotations

import pandas as pd

from src.utils import get_logger

log = get_logger(__name__)

try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    Prophet = None
    PROPHET_AVAILABLE = False


def fit_prophet(
    train_log_price: pd.Series,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    us_holidays: bool = True,
):
    """Fit on (ds, y=log price) with weekly/yearly seasonality + US holidays."""
    if not PROPHET_AVAILABLE:
        raise ImportError(
            "prophet is not installed. It is an optional secondary benchmark - "
            "install with `pip install prophet` to use it, or drop it from "
            "--models on scripts/task3_point_forecasting.py."
        )
    m = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=False,
    )
    if us_holidays:
        m.add_country_holidays(country_name="US")
    df = pd.DataFrame({"ds": train_log_price.index, "y": train_log_price.values})
    log.info("Fitting Prophet on %d points ...", len(df))
    m.fit(df)
    return m


def predict_prophet(model, future_trading_dates: pd.DatetimeIndex) -> pd.Series:
    """Forecast only for the given future *trading* dates - never weekends,
    since Prophet's trend/seasonality would otherwise happily fabricate a
    Saturday forecast that has no real counterpart to evaluate against."""
    future = pd.DataFrame({"ds": future_trading_dates})
    fcst = model.predict(future)
    return pd.Series(fcst["yhat"].values, index=future_trading_dates, name="prophet_log_price")
