"""
Task 2.1 / 2.3 - Data acquisition, cleaning, and anomaly screening.

Listing 2.1 (acquisition + log-returns) is implemented in `download_prices`
/ `add_log_returns` exactly as specified in the blueprint. This module is
deliberately thin and I/O-adjacent; the leakage-sensitive feature
construction lives in `features.py`, which operates on plain DataFrames and
is unit-testable without any network access.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src import config
from src.utils import get_logger

log = get_logger(__name__)

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def download_prices(
    ticker: str = config.TICKER,
    start: str = config.START_DATE,
    end: Optional[str] = config.END_DATE,
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch daily OHLCV via yfinance exactly as in Listing 2.1.

    Parameters
    ----------
    cache_path:
        If given, a parquet cache path. Subsequent calls read from cache
        unless `force_refresh=True`, so re-running scripts/notebooks does
        not hit the network every time.
    """
    if cache_path is not None and cache_path.exists() and not force_refresh:
        log.info("Loading cached raw prices from %s", cache_path)
        return pd.read_parquet(cache_path)

    import yfinance as yf  # imported lazily so this module stays importable
    # in environments without network/yfinance for anything downstream of
    # raw data acquisition (tests, synthetic-data runs, etc.).

    log.info("Downloading %s from %s via yfinance ...", ticker, start)
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        # Some yfinance versions return a (field, ticker) MultiIndex even
        # for a single symbol; flatten to match Listing 2.1's expectations.
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        raise RuntimeError(
            f"yfinance returned no rows for {ticker}. Check network access "
            "and the ticker symbol (or pass --synthetic to smoke-test the "
            "pipeline offline)."
        )

    df = df[REQUIRED_COLUMNS].dropna(subset=["Adj Close"])
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df.index.name = "Date"

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        log.info("Cached raw prices to %s", cache_path)

    return df


def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Log price / log return / percent-return columns (Listing 2.1)."""
    out = df.copy()
    out["log_close"] = np.log(out["Adj Close"])
    out["ret"] = out["log_close"].diff()
    out["ret_pct"] = 100 * out["ret"]  # arch-friendly percent-return scale
    out = out.dropna()
    return out


def validate_calendar(df: pd.DataFrame) -> dict:
    """Sanity checks from Task 2.3: missing values, duplicate dates,
    non-positive prices, Adj Close vs Close divergence, trading days/year.
    Never forward-fills across closed-market gaps - those are not missing
    data.
    """
    report: dict = {}
    report["n_rows"] = int(len(df))
    report["n_duplicate_index"] = int(df.index.duplicated().sum())
    report["n_missing_adj_close"] = int(df["Adj Close"].isna().sum())
    report["n_nonpositive_price"] = int(
        (df[["Open", "High", "Low", "Close", "Adj Close"]] <= 0).any(axis=1).sum()
    )

    divergence = (df["Adj Close"] - df["Close"]).abs() / df["Close"]
    report["max_adjclose_close_divergence_pct"] = float(divergence.max() * 100)

    per_year = df.groupby(df.index.year).size()
    report["trading_days_per_year"] = {int(k): int(v) for k, v in per_year.items()}
    low_years = per_year[per_year < 200]
    report["partial_or_low_count_years"] = {int(k): int(v) for k, v in low_years.items()}

    if report["n_duplicate_index"]:
        log.warning("%d duplicate index entries found", report["n_duplicate_index"])
    if report["n_nonpositive_price"]:
        log.warning("%d rows with non-positive OHLC prices", report["n_nonpositive_price"])

    return report


def flag_anomalies(df: pd.DataFrame, z_thresh: float = config.ANOMALY_Z_THRESH) -> pd.DataFrame:
    """Flag |z-score(return)| > z_thresh (Task 2.3). Diagnostic only: these
    rows are *kept*, never winsorized. Real, large market moves instead get
    an explicit `covid` event dummy in the feature matrix (Listing 2.3)
    rather than being scrubbed from the target.
    """
    out = df.copy()
    mu, sigma = out["ret"].mean(), out["ret"].std()
    out["ret_zscore"] = (out["ret"] - mu) / sigma
    out["anomaly_flag"] = out["ret_zscore"].abs() > z_thresh
    n_flagged = int(out["anomaly_flag"].sum())
    log.info("Flagged %d/%d rows with |z-score| > %.1f", n_flagged, len(out), z_thresh)
    return out


def load_or_download(force_refresh: bool = False, synthetic: bool = False) -> pd.DataFrame:
    """Convenience entry point used by scripts/task2_preprocessing.py."""
    cache_path = config.RAW_DATA_DIR / f"{config.TICKER.strip('^')}_raw.parquet"

    if synthetic:
        from src.synthetic import generate_synthetic_ohlcv

        log.warning(
            "Using SYNTHETIC data (--synthetic): for offline pipeline testing "
            "only. Never use this for a reported result."
        )
        df = generate_synthetic_ohlcv(start=config.START_DATE)
    else:
        df = download_prices(cache_path=cache_path, force_refresh=force_refresh)

    df = add_log_returns(df)
    return df
