"""
Synthetic index generator - FOR OFFLINE TESTING / DEMOS ONLY.

This module has no bearing on the production pipeline: `src/data.py` always
sources real data from Yahoo Finance via yfinance per the project's data
directive. `generate_synthetic_ohlcv` exists solely so the pipeline's
*logic* (features, splitting, models, evaluation) can be smoke-tested
end-to-end in environments without outbound access to Yahoo Finance (e.g.
locked-down CI/sandboxes), and so anyone can dry-run the whole pipeline in
seconds before pointing it at real data.

Never use this for a result you intend to report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_ohlcv(
    start: str = "1990-01-01",
    end: str | None = None,
    seed: int = 7,
    start_price: float = 350.0,
    mu_annual: float = 0.08,
) -> pd.DataFrame:
    """Simulate a GJR-GARCH-driven price path shaped like the S&P 500.

    Produces the same OHLCV schema `data.download_prices` returns (Open,
    High, Low, Close, Adj Close, Volume on a business-day DatetimeIndex), so
    it is a drop-in stand-in for real data in tests and `--synthetic` runs.
    Fat-tailed (Student-t) innovations + an asymmetric variance process give
    it volatility clustering and a leverage effect, and two amplified
    windows stand in for "crisis" regimes - none of this is fit to real
    events, it is only meant to exercise the pipeline's code paths.
    """
    rng = np.random.default_rng(seed)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    idx = pd.bdate_range(start=start, end=end_ts)
    n = len(idx)
    if n < 300:
        raise ValueError("Need at least ~300 business days for a usable synthetic series.")

    mu = mu_annual / 252
    omega, alpha, gamma, beta = 2e-6, 0.05, 0.08, 0.90  # GJR-GARCH(1,1)-like
    dof = 6  # fat-tailed Student-t innovations

    sigma2 = np.empty(n)
    eps = np.empty(n)
    ret = np.empty(n)
    sigma2[0] = omega / max(1e-6, (1 - alpha - beta - gamma / 2))
    t_shock = rng.standard_t(dof, size=n) / np.sqrt(dof / (dof - 2))  # unit-variance Student-t

    for t in range(n):
        if t > 0:
            leverage = gamma * eps[t - 1] ** 2 * (eps[t - 1] < 0)
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + leverage + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * t_shock[t]
        ret[t] = mu + eps[t]

    # A couple of amplified, slightly-negatively-biased windows to mimic
    # "crisis" vol regimes (purely cosmetic - not calibrated to real events).
    for frac_start, frac_len, mult in [(0.32, 0.02, 5.0), (0.75, 0.015, 7.0)]:
        a = int(n * frac_start)
        b = a + max(5, int(n * frac_len))
        ret[a:b] = ret[a:b] * mult - np.abs(ret[a:b]).mean() * 0.5

    log_price = np.log(start_price) + np.cumsum(ret)
    close = np.exp(log_price)
    adj_close = close.copy()

    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1] * (1 + rng.normal(0, 0.001, n - 1))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    volume = rng.lognormal(mean=19.5, sigma=0.3, size=n).astype(np.int64)

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": adj_close,
            "Volume": volume,
        },
        index=pd.DatetimeIndex(idx, name="Date"),
    )
    return df
