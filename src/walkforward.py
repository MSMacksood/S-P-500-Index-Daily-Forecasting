"""
Shared walk-forward (rolling-origin) evaluation helper.

Every model wrapper (SARIMA, GARCH, tree ensembles, LSTM/GRU) evaluates on
the same final 15% test block using a periodic-refit walk-forward scheme:
fit/tune once on train(+val), then step through the test block producing
one-step-ahead forecasts, re-fitting every `refit_every` trading days
rather than after every single day - the compute-bounding the blueprint
calls for in sec. 3.1 ("models refit or state-updated at fixed intervals,
e.g., every 21 days") and sec. 4.3 ("in practice refit weekly ...").

This module only computes the *checkpoint boundaries* generically; each
model's fit/predict mechanics differ too much (pmdarima's `.update()`,
arch's `.fix()`, XGBoost's full retrain, Keras' frozen-weights inference)
to usefully share more than that.
"""
from __future__ import annotations

from typing import Iterator, Tuple


def refit_checkpoints(test_len: int, refit_every: int) -> Iterator[Tuple[int, int]]:
    """Yield (block_start, block_end) index pairs, relative to the test
    block and with `block_end` exclusive, at which a model should be
    (re)fit and then used to forecast forward until the next checkpoint.
    """
    if refit_every <= 0:
        raise ValueError("refit_every must be a positive number of trading days")
    start = 0
    while start < test_len:
        end = min(start + refit_every, test_len)
        yield start, end
        start = end
