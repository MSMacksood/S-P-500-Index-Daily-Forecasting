"""
Task 3.2 - SARIMA via pmdarima.auto_arima (Listing 3.2): order selection on
log-price with d=1 (equivalently ARMA on returns), then rolling-origin
(walk-forward) one-step forecasts across the test block via `.update()`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.stats.diagnostic import acorr_ljungbox

from src import config
from src.utils import get_logger

log = get_logger(__name__)


def fit_auto_arima(train_log_price: pd.Series, **kwargs):
    """auto_arima order selection, exactly per Listing 3.2's search space
    (d=1, seasonal m=5 weekly terms tested but expected weak, AIC-driven
    stepwise search)."""
    kw = {**config.SARIMA_KW, **kwargs}
    log.info("Fitting pmdarima.auto_arima on %d training points ...", len(train_log_price))
    model = pm.auto_arima(train_log_price, **kw)
    log.info("Selected order=%s seasonal_order=%s", model.order, getattr(model, "seasonal_order", None))
    return model


def residual_diagnostics(model, lags=(10, 20, 30)) -> pd.DataFrame:
    """Ljung-Box on SARIMA residuals - should show no remaining autocorrelation."""
    resid = model.arima_res_.resid
    return acorr_ljungbox(resid, lags=list(lags), return_df=True)


def walk_forward_sarima(
    model,
    test_log_price: pd.Series,
    refit_every: Optional[int] = None,
    train_log_price: Optional[pd.Series] = None,
) -> pd.Series:
    """Rolling-origin one-step forecasts over the test block.

    Default (`refit_every=None`) reproduces Listing 3.2's exact loop: at
    each step, `model.predict(1)` then `model.update(actual)` (a cheap
    state refresh, not a full re-search). Passing `refit_every=N` adds an
    optional, more expensive full `auto_arima` re-identification every N
    steps (sec. 3.1's compute-bounding guidance) - requires
    `train_log_price` so there is a growing history to re-fit on.

    Returns predicted LOG-PRICE (the model was fit on log-price levels);
    exponentiate for the price level.
    """
    from src.walkforward import refit_checkpoints

    if refit_every and train_log_price is None:
        raise ValueError("train_log_price is required when refit_every is set.")

    n = len(test_log_price)
    checkpoints = list(refit_checkpoints(n, refit_every)) if refit_every else [(0, n)]
    history = train_log_price.copy() if refit_every else None

    preds, idx = [], []
    for block_start, block_end in checkpoints:
        if history is not None and block_start > 0:
            log.info("Re-identifying SARIMA order at test step %d/%d ...", block_start, n)
            model = fit_auto_arima(history)

        for t in range(block_start, block_end):
            yhat = model.predict(1)
            preds.append(float(np.asarray(yhat).ravel()[0]))
            idx.append(test_log_price.index[t])
            actual = test_log_price.iloc[t : t + 1]
            model.update(actual)
            if history is not None:
                history = pd.concat([history, actual])

    return pd.Series(preds, index=pd.Index(idx), name="sarima_log_price")
