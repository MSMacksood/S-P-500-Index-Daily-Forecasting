"""Task 2.4 / 3.1 - feature matrix leakage-safety and split correctness."""
import numpy as np
import pandas as pd

from src.features import (
    build_feature_matrix,
    chronological_split,
    reconstruct_price_from_returns,
    shift_to_target_dates,
    target_dates,
)


def test_feature_matrix_has_no_nans(clean_df):
    feat = build_feature_matrix(clean_df)
    assert feat.isna().sum().sum() == 0


def test_lag_columns_match_manual_shift(clean_df):
    feat = build_feature_matrix(clean_df)
    # ret_lag1[t] must equal ret[t-1] for every retained row.
    manual = clean_df["ret"].shift(1).reindex(feat.index)
    pd.testing.assert_series_equal(feat["ret_lag1"], manual, check_names=False)


def test_rolling_features_exclude_current_day(clean_df):
    feat = build_feature_matrix(clean_df)
    # roll_mean_5[t] must equal mean(ret[t-5..t-1]), i.e. NOT include ret[t].
    manual = clean_df["ret"].rolling(5).mean().shift(1).reindex(feat.index)
    pd.testing.assert_series_equal(feat["roll_mean_5"], manual, check_names=False)


def test_target_is_next_day_return(clean_df):
    feat = build_feature_matrix(clean_df, horizon=1)
    manual_y = clean_df["ret"].shift(-1).reindex(feat.index)
    pd.testing.assert_series_equal(feat["y"], manual_y, check_names=False)


def test_calendar_features_are_shifted_by_one(clean_df):
    feat = build_feature_matrix(clean_df)
    manual_dow = pd.Series(clean_df.index.dayofweek, index=clean_df.index).shift(1).reindex(feat.index)
    pd.testing.assert_series_equal(feat["dow"], manual_dow, check_names=False)


def test_chronological_split_proportions_and_order(clean_df):
    feat = build_feature_matrix(clean_df)
    train, val, test = chronological_split(feat, train_frac=0.70, val_frac=0.85)

    assert len(train) + len(val) + len(test) == len(feat)
    assert abs(len(train) / len(feat) - 0.70) < 0.01
    assert abs(len(val) / len(feat) - 0.15) < 0.01
    assert abs(len(test) / len(feat) - 0.15) < 0.01

    # strictly chronological, no overlap, no shuffling
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()
    assert train.index.is_monotonic_increasing
    assert val.index.is_monotonic_increasing
    assert test.index.is_monotonic_increasing


def test_target_dates_positional_lookup(clean_df):
    full_index = clean_df.index
    origins = full_index[100:110]
    targets = target_dates(origins, full_index, horizon=1)
    # each target must be exactly the next entry in the full index
    expected = full_index[101:111]
    assert list(targets) == list(expected)


def test_reconstruct_price_from_returns_perfect_foresight(clean_df):
    """Feeding the TRUE next-day return back in must reproduce the true
    next-day price exactly - this is the sanity check that caught the
    original off-by-one anchor bug during development."""
    feat = build_feature_matrix(clean_df, horizon=1)
    _, _, test = chronological_split(feat)

    recon = reconstruct_price_from_returns(clean_df["log_close"], test["y"])
    shifted = shift_to_target_dates(recon, clean_df.index, horizon=1)

    actual = clean_df.loc[shifted.index, "Adj Close"]
    np.testing.assert_allclose(shifted.values, actual.values, rtol=1e-10)
