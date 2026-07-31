# Advanced Time Series Forecasting - S&P 500 (^GSPC)

End-to-end pipeline implementing the *Advanced Time Series Forecasting Project
Blueprint*: daily S&P 500 point forecasting (naive, SARIMA, LSTM/GRU,
XGBoost, Random Forest, and a GARCH-hybrid) and volatility forecasting
(GARCH, GJR-GARCH, EGARCH), evaluated walk-forward on a strict, un-shuffled
70/15/15 chronological split.

## Setup

```bash
conda env create -f environment.yml && conda activate atsf
# or: pip install -r requirements.txt
```

`tensorflow` (LSTM/GRU) and `prophet` (secondary SARIMA benchmark) are the
two heavy/optional dependencies - every script degrades gracefully (skips
that model with a warning) if either is missing.

## Running the pipeline

```bash
python scripts/task2_preprocessing.py       # acquisition, diagnostics, features, split
python scripts/task3_point_forecasting.py   # naive, SARIMA, LSTM/GRU, XGBoost, RF
python scripts/task4_volatility.py          # GARCH(1,1), GJR-GARCH, EGARCH
python scripts/task5_evaluation.py          # hybrid model, final comparison, backtest

# or all four in sequence:
python scripts/run_all.py
```

Every script/`run_all.py` accepts `--quick`, which shrinks Optuna
trials/LSTM epochs and widens refit cadences so the whole pipeline finishes
in well under a minute - use it to sanity-check the plumbing, **never** for
a result you intend to report. `scripts/task2_preprocessing.py --synthetic`
(and `run_all.py --synthetic`) swaps in a locally-simulated GJR-GARCH price
path instead of calling yfinance, for offline development or CI. Most
scripts also take `--test-limit N` to cap the walk-forward length while
debugging.

Outputs land in `results/` (`metrics_*.csv`, `dm_pvalue_matrix.csv`,
`toy_backtest.csv`, `figures/`, `tuned_params/*.json`) and
`data/processed/` (cleaned prices, feature matrix, train/val/test splits).

## Repository layout

```
data/            raw/ (yfinance cache) and processed/ (features, splits) - gitignored
notebooks/       01_eda ... 07_comparison - thin, interactive wrappers around src/
src/             the tested, reusable pipeline logic
  config.py        every tunable constant (dates, split ratios, search spaces, quick-mode)
  data.py          acquisition (Listing 2.1), calendar validation, anomaly screening
  diagnostics.py   STL, ACF/PACF, ADF/KPSS, ARCH-LM, Jarque-Bera (Listings 2.2/2.4)
  features.py      leakage-safe feature matrix + split (Listing 2.3, 3.1) + index-alignment helpers
  evaluate.py       metrics, Diebold-Mariano, QLIKE, Mincer-Zarnowitz, toy backtest (Listing 3.5)
  walkforward.py   shared refit-checkpoint helper for walk-forward evaluation
  synthetic.py     offline test-data generator (never used for reported results)
  models/          naive, sarima_model, prophet_model, dl_models, tree_models, hybrid, garch_models
scripts/         task2-5 runnable entry points + run_all.py
tests/           pytest suite (feature leakage, metric correctness, model smoke tests)
results/         metrics.csv, figures/, tuned_params/ - regenerated, gitignored
report/          this blueprint
environment.yml / requirements.txt
```

## Design notes worth knowing before you extend this

**Index convention.** Every model's prediction is stored keyed by the date
it is a forecast *for*, not the date it was made from. Naive and SARIMA are
target-date-indexed by construction (their fitted history always ends
exactly one step before the date they're about to forecast); LSTM/XGBoost/
RF/hybrid predict from an origin-indexed feature row and are explicitly
relabeled via `features.shift_to_target_dates` after price reconstruction.
`scripts/task3_point_forecasting.py`'s module docstring has the full
rationale - and why a handful of boundary rows are `NaN` for some models
(the union of two 1-day-shifted index conventions).

**Leakage prevention.** Every feature is shifted so a row only uses
information dated strictly before the date it predicts - including the
calendar indicators, which the blueprint's own illustrative Listing 2.3
leaves unshifted; the project's stated critical constraints are stricter
and win out (see `features._calendar_features`'s docstring). Scalers
(`StandardScaler` for the LSTM) are fit on the training block only.

**Walk-forward compute-bounding.** A literal "refit at every single test
day" (as Listings 3.2/4.1 illustrate) is prohibitively slow at full scale.
Every model instead refits periodically (`config.SARIMA_REFIT_EVERY`,
`TREE_REFIT_EVERY`, `GARCH_REFIT_EVERY`, `DL_REFIT_EVERY`) and filters
forward with fixed parameters in between - exactly the tradeoff the
blueprint's own prose calls for in sec. 3.1 and 4.3. Pass `--refit-every 1`
to `task4_volatility.py` to reproduce Listing 4.1 literally on a short
window if you want to confirm the two are numerically close (they are,
to ~0.03% in testing).

**SARIMA/GARCH fit on train+val, not train alone.** Both need no held-out
set for their own tuning (auto_arima's order search and the GARCH MLE both
use only their own fit window), so - to avoid a multi-year blind gap
between where `train` ends and `test` begins - they fit on `train+val`
combined and walk forward from there, matching Listing 4.1's own
`split = int(len(r)*0.85)`. `val` stays genuinely held out for XGBoost/RF
(`eval_set`) and the LSTM (`validation_data`/early stopping).

**Primary horizon is 1 day.** The secondary 5-day horizon (sec. 1.2) isn't
wired into a full model-training pipeline; `features.build_cumulative_return_target`
implements the *direct* cumulative-return target it needs (a single day's
return 5 days out is not a useful 5-day-ahead target - see that function's
docstring) for anyone extending this.

## Testing

```bash
pytest tests/ -v
```

Covers feature-matrix leakage safety (including the exact off-by-one bug
this pipeline was debugged against during development), metric/DM-test
correctness against hand-computed and synthetic ground truth, and a fast
fit/predict smoke test for every model family. The LSTM test is skipped
automatically if `tensorflow` isn't installed.
