"""
Task 2.2 / 2.5 - Visualization, STL decomposition, ACF/PACF, stationarity
and heteroscedasticity diagnostics (Listings 2.2 and 2.4).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: scripts may run without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

from src import config
from src.utils import get_logger, save_json

log = get_logger(__name__)


def plot_level_and_returns(df: pd.DataFrame, out_dir: Path = config.FIGURES_DIR) -> Path:
    """Level plot (log scale) with crisis shading + return plot (Task 2.2)."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    axes[0].plot(df.index, df["Adj Close"], lw=0.8, color="tab:blue")
    axes[0].set_yscale("log")
    axes[0].set_title(f"{config.TICKER} Adjusted Close (log scale)")
    for _name, (s, e) in config.KNOWN_EVENTS.items():
        axes[0].axvspan(pd.Timestamp(s), pd.Timestamp(e), color="tab:red", alpha=0.15)

    axes[1].plot(df.index, df["ret"], lw=0.5, color="tab:gray")
    axes[1].set_title("Daily log-return")
    axes[1].axhline(0, color="black", lw=0.5)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "level_and_returns.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def run_stl_decomposition(
    log_price: pd.Series, period: int = 252, robust: bool = True, out_dir: Path = config.FIGURES_DIR
):
    """STL decomposition on log-price (Listing 2.2): trend / seasonal(252) / resid."""
    stl = STL(log_price, period=period, robust=robust).fit()
    fig = stl.plot()
    fig.set_size_inches(10, 8)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "stl_decomposition.png", dpi=120)
    plt.close(fig)
    return stl


def plot_acf_pacf_diagnostics(ret: pd.Series, lags: int = 40, out_dir: Path = config.FIGURES_DIR) -> Path:
    """ACF/PACF of returns (~white noise) vs squared returns (slow decay -> ARCH)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_acf(ret, lags=lags, ax=axes[0, 0], title="ACF: returns")
    plot_pacf(ret, lags=lags, ax=axes[0, 1], title="PACF: returns")
    plot_acf(ret**2, lags=lags, ax=axes[1, 0], title="ACF: squared returns")
    plot_pacf(ret**2, lags=lags, ax=axes[1, 1], title="PACF: squared returns")
    fig.tight_layout()
    path = out_dir / "acf_pacf.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def rolling_realized_vol(ret: pd.Series, window: int = 21) -> pd.Series:
    """Annualized realized-vol proxy (Listing 2.2)."""
    return ret.rolling(window).std() * np.sqrt(252)


def distribution_diagnostics(ret: pd.Series, out_dir: Path = config.FIGURES_DIR) -> dict:
    """Histogram + QQ-plot + Jarque-Bera test (Task 2.2: fat tails, skew)."""
    from scipy import stats

    clean = ret.dropna()
    jb_stat, jb_p = stats.jarque_bera(clean)
    skew = stats.skew(clean)
    kurt = stats.kurtosis(clean)  # excess kurtosis (Fisher, normal -> 0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(clean, bins=100, color="tab:blue", alpha=0.8)
    axes[0].set_title("Return distribution")
    stats.probplot(clean, dist="norm", plot=axes[1])
    axes[1].set_title("QQ-plot vs Normal")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "distribution_diagnostics.png", dpi=120)
    plt.close(fig)

    return dict(
        jarque_bera_stat=float(jb_stat),
        jarque_bera_p=float(jb_p),
        skew=float(skew),
        excess_kurtosis=float(kurt),
    )


def stationarity_tests(series: pd.Series, name: str) -> dict:
    """ADF (H0: unit root) + KPSS (H0: stationary) - Listing 2.4."""
    s = series.dropna()
    adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
    with np.errstate(all="ignore"):
        kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
    result = dict(
        name=name,
        adf_stat=float(adf_stat),
        adf_p=float(adf_p),
        kpss_stat=float(kpss_stat),
        kpss_p=float(kpss_p),
    )
    log.info("%s: ADF p=%.4f  KPSS p=%.4f", name, adf_p, kpss_p)
    return result


def arch_lm_test(ret: pd.Series, nlags: int = 12) -> dict:
    """Engle ARCH-LM test on returns - the formal bridge to Task 4 (Listing 2.4)."""
    stat, p, f_stat, f_p = het_arch(ret.dropna(), nlags=nlags)
    log.info("ARCH-LM p-value = %.6g (expect ~0: strong ARCH effects)", p)
    return dict(lm_stat=float(stat), lm_p=float(p), f_stat=float(f_stat), f_p=float(f_p))


def ljung_box_test(series: pd.Series, lags=(10, 20, 30)) -> pd.DataFrame:
    return acorr_ljungbox(series.dropna(), lags=list(lags), return_df=True)


def run_full_diagnostics(df: pd.DataFrame, out_dir: Path = config.RESULTS_DIR) -> dict:
    """Runs every Task 2.2/2.5 diagnostic, writes figures + a JSON report."""
    log.info("Running Task 2 diagnostics ...")
    plot_level_and_returns(df)
    run_stl_decomposition(df["log_close"])
    plot_acf_pacf_diagnostics(df["ret"])

    report = {
        "stationarity": [
            stationarity_tests(df["log_close"], "log_close"),
            stationarity_tests(df["ret"], "ret"),
        ],
        "arch_lm_on_returns": arch_lm_test(df["ret"]),
        "ljung_box_returns": ljung_box_test(df["ret"]).to_dict(orient="index"),
        "ljung_box_squared_returns": ljung_box_test(df["ret"] ** 2).to_dict(orient="index"),
        "distribution": distribution_diagnostics(df["ret"]),
    }
    out_path = out_dir / "task2_diagnostics_report.json"
    save_json(report, out_path)
    log.info("Diagnostics report saved to %s", out_path)
    return report
