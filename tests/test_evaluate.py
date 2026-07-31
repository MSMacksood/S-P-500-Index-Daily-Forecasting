"""Task 3.5 / 5.1 - metric suite and Diebold-Mariano correctness."""
import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    build_comparison_table,
    diebold_mariano,
    metrics,
    qlike,
    toy_long_flat_backtest,
    variance_mse,
)


def test_metrics_matches_hand_calculation():
    y = np.array([100.0, 101.0, 102.0, 101.0])
    yhat = np.array([100.0, 100.0, 103.0, 102.0])
    m = metrics(y, yhat)

    e = y - yhat
    assert m["MAE"] == pytest.approx(np.mean(np.abs(e)))
    assert m["RMSE"] == pytest.approx(np.sqrt(np.mean(e**2)))
    assert m["MAPE"] == pytest.approx(np.mean(np.abs(e / y)) * 100)
    assert m["DirAcc"] == pytest.approx(2 / 3)


def test_dm_test_identical_forecasts_gives_no_difference():
    rng = np.random.default_rng(0)
    e = rng.normal(0, 1, 300)
    dm_stat, p_value = diebold_mariano(e, e.copy())
    assert dm_stat == pytest.approx(0.0)
    assert p_value == pytest.approx(1.0)


def test_dm_test_detects_a_clearly_better_model():
    rng = np.random.default_rng(1)
    n = 1000
    e_good = rng.normal(0, 1, n)       # smaller errors
    e_bad = rng.normal(0, 5, n)        # larger errors, same process otherwise
    dm_stat, p_value = diebold_mariano(e_good, e_bad)
    assert p_value < 0.01
    assert dm_stat < 0  # model 1 (good) has lower loss -> negative statistic


def test_dm_test_requires_equal_length():
    with pytest.raises(ValueError):
        diebold_mariano(np.zeros(10), np.zeros(11))


def test_qlike_prefers_accurate_forecast_over_constant():
    rng = np.random.default_rng(2)
    realized = np.abs(rng.normal(1, 0.3, 200)) ** 2
    accurate = realized * (1 + rng.normal(0, 0.05, 200))
    constant = np.full_like(realized, realized.mean())
    assert qlike(realized, accurate) < qlike(realized, constant)
    assert variance_mse(realized, realized) == pytest.approx(0.0)


def test_build_comparison_table_flags_significant_improvement_over_naive():
    idx = pd.date_range("2021-01-01", periods=300, freq="B")
    rng = np.random.default_rng(3)
    actual = 100 + np.cumsum(rng.normal(0, 1, 300))
    naive = np.roll(actual, 1)
    naive[0] = actual[0]
    good_model = actual + rng.normal(0, 0.1, 300)  # much more accurate than naive

    pred_df = pd.DataFrame(
        {"Naive Random Walk": naive, "GoodModel": good_model, "Actual": actual}, index=idx
    )
    summary, dm_matrix = build_comparison_table(pred_df, ["Naive Random Walk", "GoodModel"])

    assert summary.loc["GoodModel", "RMSE"] < summary.loc["Naive Random Walk", "RMSE"]
    assert summary.loc["GoodModel", "DM_p_vs_naive"] < 0.05
    assert np.isnan(dm_matrix.loc["Naive Random Walk", "Naive Random Walk"])


def test_toy_backtest_naive_zero_signal_never_trades():
    idx = pd.date_range("2021-01-01", periods=50, freq="B")
    actual_ret = pd.Series(np.random.default_rng(4).normal(0, 0.01, 50), index=idx)
    zero_signal = pd.Series(0.0, index=idx)  # naive's implied predicted return
    result = toy_long_flat_backtest(actual_ret, zero_signal, cost_bps=5.0)
    assert result["cumulative_return_strategy"] == pytest.approx(0.0)
    assert result["avg_daily_turnover"] == pytest.approx(0.0)
