"""Shared fixtures: a small, fast synthetic price/return series so the
whole test suite runs in seconds without any network access."""
import pytest

from src.data import add_log_returns
from src.synthetic import generate_synthetic_ohlcv


@pytest.fixture(scope="session")
def raw_df():
    return generate_synthetic_ohlcv(start="2000-01-01", end="2010-12-31", seed=11)


@pytest.fixture(scope="session")
def clean_df(raw_df):
    return add_log_returns(raw_df)
