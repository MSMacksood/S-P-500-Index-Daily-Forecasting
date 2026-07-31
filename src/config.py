"""
Central configuration for the Advanced Time Series Forecasting pipeline.

Every tunable constant used across Tasks 2-5 lives here so the whole
pipeline can be re-parameterized (ticker, date range, split ratios, search
spaces, quick-mode) from one file instead of hunting through modules.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

# --------------------------------------------------------------------------- #
# Paths (Listing 5.1 repo layout)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TUNED_PARAMS_DIR = RESULTS_DIR / "tuned_params"

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, FIGURES_DIR, TUNED_PARAMS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Quick mode: ATSF_QUICK=1 (or --quick on any script) shrinks epochs / optuna
# trials / refit frequency for fast, cheap end-to-end smoke tests of the
# pipeline's *logic*. Never use it for a result you intend to report.
# --------------------------------------------------------------------------- #
QUICK_MODE = os.environ.get("ATSF_QUICK", "0") == "1"


def n_trials(n: int) -> int:
    """Shrink a trial count under quick mode."""
    return min(n, 3) if QUICK_MODE else n


def n_epochs(n: int) -> int:
    """Shrink an epoch count under quick mode."""
    return min(n, 3) if QUICK_MODE else n


def refit_every(n: int) -> int:
    """Widen a refit cadence under quick mode (fewer, bigger walk-forward steps)."""
    return max(n, 60) if QUICK_MODE else n


# --------------------------------------------------------------------------- #
# Data acquisition (Task 2.1 / Listing 2.1)
# --------------------------------------------------------------------------- #
TICKER = "^GSPC"
VIX_TICKER = "^VIX"
START_DATE = "1990-01-01"
END_DATE: str | None = None  # None = through "today"

KNOWN_EVENTS = {
    "Dot-com bust": ("2000-03-01", "2002-10-01"),
    "Global Financial Crisis": ("2007-10-01", "2009-03-01"),
    "Flash Crash": ("2010-05-06", "2010-05-07"),
    "COVID-19 Crash": ("2020-02-20", "2020-04-30"),
    "2022 Rate-Hike Bear Market": ("2022-01-01", "2022-10-01"),
}

# --------------------------------------------------------------------------- #
# Train / validation / test split (Task 3.1 / Listing 3.1): strict,
# chronological, un-shuffled 70/15/15.
# --------------------------------------------------------------------------- #
TRAIN_FRAC = 0.70
VAL_FRAC = 0.85  # cumulative fraction; val block is the half-open (TRAIN_FRAC, VAL_FRAC]

# --------------------------------------------------------------------------- #
# Feature engineering (Task 2.4 / Listing 2.3)
# --------------------------------------------------------------------------- #
FORECAST_HORIZON = 1     # H: primary target is next-day (h=1)
SECONDARY_HORIZON = 5    # 1 trading week ahead (blueprint sec. 1.2)
RET_LAGS: Sequence[int] = (1, 2, 3, 5, 10, 21)
ROLL_WINDOWS: Sequence[int] = (5, 21, 63)
MOMENTUM_WINDOW = 63
ANOMALY_Z_THRESH = 5.0
COVID_START = "2020-02-20"
COVID_END = "2020-04-30"

# --------------------------------------------------------------------------- #
# Walk-forward evaluation cadence (trading days). Sec. 3.1: "models refit or
# state-updated at fixed intervals, e.g., every 21 days, to keep compute
# bounded"; sec. 4.3: "(In practice refit weekly ...)" for GARCH.
# --------------------------------------------------------------------------- #
SARIMA_REFIT_EVERY = 21
GARCH_REFIT_EVERY = 5
TREE_REFIT_EVERY = 21
DL_REFIT_EVERY = 21

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEEDS: Sequence[int] = (0, 1, 2, 3, 4)
DEFAULT_SEED = 42

# --------------------------------------------------------------------------- #
# Model search spaces / defaults
# --------------------------------------------------------------------------- #
SARIMA_KW = dict(
    d=1, start_p=0, max_p=5, start_q=0, max_q=5,
    seasonal=True, m=5, D=0, max_P=2, max_Q=2,
    information_criterion="aic", stepwise=True, suppress_warnings=True,
)

LSTM_LOOKBACK_OPTIONS = (20, 60, 120)
LSTM_UNITS_OPTIONS = (32, 64, 128)
LSTM_LAYERS_OPTIONS = (1, 2)
LSTM_DROPOUT_OPTIONS = (0.0, 0.2, 0.4)
LSTM_LR_OPTIONS = (1e-2, 1e-3, 1e-4)
LSTM_CELL_OPTIONS = ("LSTM", "GRU")
LSTM_MAX_EPOCHS = 200
LSTM_PATIENCE = 15
LSTM_BATCH_SIZE = 64
LSTM_N_TUNING_TRIALS = 30

XGB_N_TRIALS = 100
RF_N_TRIALS = 50

GARCH_DIST = "t"
