"""
Task 5 - Model Comparison, Insights, and Critical Reflection.

The metric suite (Listing 3.5), the Diebold-Mariano significance test vs
the naive random-walk benchmark (sec. 3.5 / 5.1), volatility-forecast
evaluation (QLIKE, Mincer-Zarnowitz - sec. 4.4), a toy long/flat backtest
(sec. 5.3), and the final results-matrix assembly (sec. 5.1).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Point-forecast metrics (Listing 3.5)
# --------------------------------------------------------------------------- #
def metrics(y, yhat) -> dict:
    """Listing 3.5's metric suite, implemented exactly as specified. `y`,
    `yhat` should be PRICE LEVELS (MAPE is undefined/explosive on returns,
    which cross zero - sec. 3.5's own note)."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    e = y - yhat
    mae = np.mean(np.abs(e))
    rmse = np.sqrt(np.mean(e**2))
    mape = np.mean(np.abs(e / y)) * 100
    smape = np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(yhat))) * 100
    if len(y) >= 2:
        da = np.mean(np.sign(np.diff(y)) == np.sign(np.diff(yhat)))
    else:
        da = float("nan")
    return dict(MAE=float(mae), RMSE=float(rmse), MAPE=float(mape), sMAPE=float(smape), DirAcc=float(da))


# --------------------------------------------------------------------------- #
# Diebold-Mariano test
# --------------------------------------------------------------------------- #
def _newey_west_long_run_variance(d: np.ndarray, lags: int) -> float:
    """Bartlett-kernel (Newey-West) long-run variance of the mean of `d`."""
    n = len(d)
    dc = d - d.mean()
    var = float(np.sum(dc**2)) / n
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        gamma = float(np.sum(dc[lag:] * dc[:-lag])) / n
        var += 2 * weight * gamma
    return var


def diebold_mariano(
    e1: np.ndarray, e2: np.ndarray, h: int = 1, loss: str = "squared", small_sample_correction: bool = True
) -> Tuple[float, float]:
    """Diebold & Mariano (1995) test for equal predictive accuracy, with the
    Harvey-Leybourne-Newbold (1997) small-sample correction (recommended
    unless T is very large; harmless when it is).

    `e1`, `e2` are forecast errors (actual - predicted) from two models
    over the SAME dates. `loss="squared"` compares squared-error
    differentials - sec. 3.5's stated protocol ("Diebold-Mariano test on
    pairwise squared-error differentials"); `loss="absolute"` is also
    available. Always test every model against the naive random walk
    (sec. 3.5: "... which every model must beat to claim skill").

    Returns (dm_stat, p_value). A significantly negative stat means model 1
    is more accurate (lower loss) than model 2; significantly positive
    means the reverse.
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    if e1.shape != e2.shape:
        raise ValueError("e1 and e2 must be the same length (aligned on identical dates).")
    n = len(e1)
    if n < 5:
        log.warning("DM test with only %d observations - result will be unreliable.", n)

    if loss == "squared":
        g1, g2 = e1**2, e2**2
    elif loss == "absolute":
        g1, g2 = np.abs(e1), np.abs(e2)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    d = g1 - g2
    dbar = float(d.mean())
    lags = max(h - 1, 0)
    long_run_var = _newey_west_long_run_variance(d, lags)

    if long_run_var <= 0:
        # Degenerate: e.g. two identical forecasts -> no detectable difference.
        return 0.0, 1.0

    dm_stat = dbar / np.sqrt(long_run_var / n)

    if small_sample_correction:
        correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        dm_stat = dm_stat * correction
        pvalue = 2 * stats.t.sf(np.abs(dm_stat), df=n - 1)
    else:
        pvalue = 2 * stats.norm.sf(np.abs(dm_stat))

    return float(dm_stat), float(pvalue)


# --------------------------------------------------------------------------- #
# Volatility-forecast evaluation (sec. 4.4)
# --------------------------------------------------------------------------- #
def qlike(realized_var, forecast_var) -> float:
    """QLIKE loss - preferred over MSE for variance forecasts because it is
    far less sensitive to noise in the realized-variance proxy (sec. 4.4)."""
    realized_var = np.asarray(realized_var, dtype=float)
    forecast_var = np.asarray(forecast_var, dtype=float)
    return float(np.mean(np.log(forecast_var) + realized_var / forecast_var))


def variance_mse(realized_var, forecast_var) -> float:
    realized_var = np.asarray(realized_var, dtype=float)
    forecast_var = np.asarray(forecast_var, dtype=float)
    return float(np.mean((realized_var - forecast_var) ** 2))


def mincer_zarnowitz(realized_var, forecast_var) -> dict:
    """Regress realized on forecast variance: alpha=0, beta=1 is the ideal
    (unbiased, efficient) forecast (sec. 4.4)."""
    import statsmodels.api as sm

    X = sm.add_constant(np.asarray(forecast_var, dtype=float))
    y = np.asarray(realized_var, dtype=float)
    model = sm.OLS(y, X).fit()
    alpha, beta = model.params
    alpha_se, beta_se = model.bse
    return dict(
        alpha=float(alpha), beta=float(beta), r2=float(model.rsquared),
        alpha_se=float(alpha_se), beta_se=float(beta_se),
    )


def build_volatility_comparison(
    vol_df: pd.DataFrame, model_cols: Sequence[str], realized_col: str = "sq_ret"
) -> pd.DataFrame:
    """QLIKE + variance-MSE + Mincer-Zarnowitz for each volatility model vs
    a realized-variance proxy (sec. 5.1: "Volatility models are compared
    separately ... on QLIKE and variance-MSE")."""
    rows = []
    for col in model_cols:
        sub = vol_df[[col, realized_col]].dropna()
        mz = mincer_zarnowitz(sub[realized_col].values, sub[col].values)
        rows.append(
            {
                "Model": col,
                "QLIKE": qlike(sub[realized_col].values, sub[col].values),
                "Variance_MSE": variance_mse(sub[realized_col].values, sub[col].values),
                "MZ_alpha": mz["alpha"], "MZ_beta": mz["beta"], "MZ_R2": mz["r2"],
                "N": len(sub),
            }
        )
    return pd.DataFrame(rows).set_index("Model").sort_values("QLIKE")


# --------------------------------------------------------------------------- #
# Toy long/flat backtest (sec. 5.3)
# --------------------------------------------------------------------------- #
def toy_long_flat_backtest(actual_return: pd.Series, predicted_return: pd.Series, cost_bps: float = 0.0) -> dict:
    """Long whenever the predicted return is positive, else flat; both
    series are log-returns aligned on identical target dates. `cost_bps`
    (one-way, in basis points) is charged on every position change - sec.
    5.3: "destroyed by transaction costs at daily frequency - quantify
    this with the toy-strategy backtest before making any claims."""
    common = pd.concat([actual_return.rename("actual"), predicted_return.rename("pred")], axis=1).dropna()
    if len(common) < 2:
        return dict(cumulative_return_strategy=float("nan"), cumulative_return_buy_and_hold=float("nan"),
                    annualized_sharpe=float("nan"), hit_rate=float("nan"), avg_daily_turnover=float("nan"),
                    total_cost_drag=float("nan"), n=len(common))

    position = (common["pred"] > 0).astype(float)
    strategy_ret = position * common["actual"]
    turnover = position.diff().abs()
    turnover.iloc[0] = position.iloc[0]
    cost = turnover * (cost_bps / 10_000)
    net_ret = strategy_ret - cost

    cum_strategy = np.exp(net_ret.cumsum().iloc[-1]) - 1
    cum_bh = np.exp(common["actual"].cumsum().iloc[-1]) - 1
    sharpe = float(np.sqrt(252) * net_ret.mean() / net_ret.std()) if net_ret.std() > 0 else float("nan")

    return dict(
        cumulative_return_strategy=float(cum_strategy),
        cumulative_return_buy_and_hold=float(cum_bh),
        annualized_sharpe=sharpe,
        hit_rate=float((np.sign(common["pred"]) == np.sign(common["actual"])).mean()),
        avg_daily_turnover=float(turnover.mean()),
        total_cost_drag=float(cost.sum()),
        n=len(common),
    )


# --------------------------------------------------------------------------- #
# Regime analysis (sec. 5.1: "error-by-regime bars (high vs low VIX terciles)")
# --------------------------------------------------------------------------- #
def error_by_regime(abs_errors: pd.Series, vix: pd.Series, n_terciles: int = 3) -> pd.DataFrame:
    aligned_vix = vix.reindex(abs_errors.index)
    common = pd.concat([abs_errors.rename("abs_error"), aligned_vix.rename("vix")], axis=1).dropna()
    labels = [f"T{i + 1}_{'low' if i == 0 else ('high' if i == n_terciles - 1 else 'mid')}" for i in range(n_terciles)]
    common["regime"] = pd.qcut(common["vix"], n_terciles, labels=labels)
    return common.groupby("regime")["abs_error"].agg(["mean", "count"]).rename(columns={"mean": "mean_abs_error"})


# --------------------------------------------------------------------------- #
# Final comparison table + DM matrix (sec. 5.1)
# --------------------------------------------------------------------------- #
def build_comparison_table(
    pred_df: pd.DataFrame,
    model_names: Sequence[str],
    actual_col: str = "Actual",
    naive_col: str = "Naive Random Walk",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The Table-3-shaped summary (Model | MAE | RMSE | MAPE% | sMAPE% |
    DirAcc | DM vs naive) plus a full pairwise DM p-value matrix (sec. 5.1).
    Each model is scored on its own valid (non-NaN) rows - walk-forward
    reconstruction leaves a handful of boundary rows some models don't
    cover (see scripts/task3_point_forecasting.py's module docstring).
    """
    rows = []
    for name in model_names:
        sub = pred_df[[name, actual_col]].dropna()
        m = metrics(sub[actual_col].values, sub[name].values)

        dm_stat, dm_p = float("nan"), float("nan")
        if name != naive_col and naive_col in pred_df.columns:
            common = pred_df[[name, naive_col, actual_col]].dropna()
            if len(common) >= 5:
                e_model = (common[actual_col] - common[name]).values
                e_naive = (common[actual_col] - common[naive_col]).values
                dm_stat, dm_p = diebold_mariano(e_model, e_naive)

        rows.append({"Model": name, **m, "DM_stat_vs_naive": dm_stat, "DM_p_vs_naive": dm_p, "N": len(sub)})

    summary = pd.DataFrame(rows).set_index("Model")

    dm_matrix = pd.DataFrame(index=list(model_names), columns=list(model_names), dtype=float)
    for a in model_names:
        for b in model_names:
            if a == b:
                continue
            common = pred_df[[a, b, actual_col]].dropna()
            if len(common) < 5:
                continue
            e_a = (common[actual_col] - common[a]).values
            e_b = (common[actual_col] - common[b]).values
            _, p = diebold_mariano(e_a, e_b)
            dm_matrix.loc[a, b] = p

    return summary, dm_matrix


def cumulative_squared_error(pred_df: pd.DataFrame, model_names: Sequence[str], actual_col: str = "Actual") -> pd.DataFrame:
    """Cumulative-squared-error curves (sec. 5.1) - reveal *when* a model
    fails, not just its average error."""
    out = {}
    for name in model_names:
        sub = pred_df[[name, actual_col]].dropna()
        out[name] = ((sub[actual_col] - sub[name]) ** 2).cumsum()
    return pd.DataFrame(out)
