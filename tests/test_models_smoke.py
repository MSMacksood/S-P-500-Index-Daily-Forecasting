"""Fast smoke tests for every model wrapper on tiny synthetic data. These
check that each model's fit/predict plumbing and index alignment are
correct, not real forecasting skill. TensorFlow-dependent tests are
skipped automatically (`pytest.importorskip`) if it is not installed.
"""
import numpy as np
import pandas as pd
import pytest

from src.features import build_feature_matrix, chronological_split
from src.models import garch_models as gm
from src.models import tree_models
from src.models.naive import naive_price_forecast, naive_return_forecast
from src.walkforward import refit_checkpoints


def test_naive_forecast_is_shifted_actual(clean_df):
    idx = clean_df.index[-50:]
    pred = naive_price_forecast(clean_df["Adj Close"], idx)
    expected = clean_df["Adj Close"].shift(1).reindex(idx)
    pd.testing.assert_series_equal(pred, expected, check_names=False)
    assert (naive_return_forecast(idx) == 0.0).all()


def test_refit_checkpoints_cover_range_without_gaps_or_overlap():
    checkpoints = list(refit_checkpoints(100, 21))
    assert checkpoints[0][0] == 0
    assert checkpoints[-1][1] == 100
    for (s0, e0), (s1, e1) in zip(checkpoints, checkpoints[1:]):
        assert e0 == s1  # contiguous, no gap or overlap

    with pytest.raises(ValueError):
        list(refit_checkpoints(10, 0))


def test_xgboost_fit_predict_walk_forward(clean_df):
    feat = build_feature_matrix(clean_df)
    train, val, test = chronological_split(feat)
    train_val = pd.concat([train, val])

    params = dict(n_estimators=20, max_depth=2, learning_rate=0.1, subsample=1.0, colsample_bytree=1.0, min_child_weight=1, reg_lambda=1.0)
    model = tree_models.train_xgboost(params, train_val)
    cols = tree_models.get_feature_columns(test)
    preds = model.predict(test[cols].values)
    assert len(preds) == len(test)
    assert np.isfinite(preds).all()

    fit_fn = lambda h: tree_models.train_xgboost(params, h)  # noqa: E731
    wf = tree_models.walk_forward_predict_tree(fit_fn, train_val, test.iloc[:40], refit_every=20)
    assert len(wf) == 40
    assert wf.index.equals(test.iloc[:40].index)


def test_random_forest_fit_predict(clean_df):
    feat = build_feature_matrix(clean_df)
    train, val, test = chronological_split(feat)
    params = dict(n_estimators=20, max_depth=4, min_samples_leaf=5, max_features=0.5)
    model = tree_models.train_random_forest(params, train)
    cols = tree_models.get_feature_columns(test)
    preds = model.predict(test[cols].values[:10])
    assert len(preds) == 10 and np.isfinite(preds).all()


def test_garch_family_fits_and_forecasts_positive_variance(clean_df):
    r = clean_df["ret_pct"]
    split = int(len(r) * 0.85)
    fitted = gm.fit_garch_family(r.iloc[:split])
    assert set(fitted) == set(gm.GARCH_SPECS)
    best = gm.select_best_by_bic(fitted)
    assert best in fitted

    summary = gm.summarize_params(fitted[best])
    assert 0 < summary["persistence"] < 1.5  # sanity bound, not a tight spec

    vf = gm.rolling_variance_forecast(r, split, gm.GARCH_SPECS[best], refit_every=10)
    assert (vf > 0).all()
    assert vf.index.equals(r.iloc[split:].index)


def test_dl_windowing_is_backend_free():
    from src.models.dl_models import make_windows

    s = np.arange(20.0)
    X, y = make_windows(s, lookback=5)
    assert X.shape == (15, 5, 1)
    assert y[0] == 5.0
    np.testing.assert_array_equal(X[0].ravel(), [0, 1, 2, 3, 4])


def test_lstm_build_and_fit_tiny():
    tf = pytest.importorskip("tensorflow", reason="tensorflow not installed - LSTM/GRU is an optional heavy dependency")
    from src.models import dl_models

    rng = np.random.default_rng(0)
    ret = pd.Series(rng.normal(0, 0.01, 400), index=pd.bdate_range("2020-01-01", periods=400))
    train_ret, val_ret = ret.iloc[:300], ret.iloc[300:]

    ds = dl_models.prepare_dataset(train_ret, val_ret, lookback=10)
    model = dl_models.build_model(units=4, layers=1, dropout=0.0, lr=1e-2, cell="GRU", lookback=10)
    dl_models.train_model(model, ds, epochs=2, patience=1, verbose=0)
    preds = model.predict(ds.Xva, verbose=0)
    assert np.isfinite(preds).all()
