"""
Task 4 - Volatility Modeling with GARCH (Listing 4.1): GARCH(1,1),
GJR-GARCH, and EGARCH, all with Student-t errors, plus efficient rolling
1-day-ahead variance forecasts over the test block.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from arch import arch_model

from src import config
from src.utils import get_logger

log = get_logger(__name__)

# vol="GARCH" with o=0 is plain GARCH(1,1); o=1 adds the GJR leverage term.
GARCH_SPECS = {
    "GARCH(1,1)": dict(vol="GARCH", p=1, o=0, q=1),
    "GJR-GARCH(1,1,1)": dict(vol="GARCH", p=1, o=1, q=1),
    "EGARCH(1,1,1)": dict(vol="EGARCH", p=1, o=1, q=1),
}


def fit_garch_family(returns_pct_train: pd.Series, dist: str = config.GARCH_DIST) -> Dict[str, object]:
    """Fit GARCH(1,1), GJR-GARCH, and EGARCH - all Student-t - on the same
    training window (Listing 4.1). `returns_pct_train` should be train+val
    combined (percent returns), mirroring Listing 4.1's own
    `split = int(len(r)*0.85)` (i.e. it deliberately folds val into the
    GARCH fit, the same choice made for SARIMA - see sarima_model.py).
    Returns {name: fitted ARCHModelResult}.
    """
    fitted = {}
    for name, spec in GARCH_SPECS.items():
        log.info("Fitting %s (dist=%s) on %d points ...", name, dist, len(returns_pct_train))
        res = arch_model(returns_pct_train, dist=dist, **spec).fit(disp="off")
        fitted[name] = res
        log.info("  %s: AIC=%.2f BIC=%.2f LL=%.2f", name, res.aic, res.bic, res.loglikelihood)
    return fitted


def select_best_by_bic(fitted: Dict[str, object]) -> str:
    """Lowest BIC wins (Listing 4.1: "Model choice: lowest BIC + significant
    gamma -> expect GJR to win")."""
    return min(fitted, key=lambda k: fitted[k].bic)


def summarize_params(res) -> dict:
    """omega / alpha / beta / gamma / nu, persistence, and shock half-life
    (sec. 4.4): half-life = ln(0.5)/ln(persistence) days.

    Persistence is alpha+beta for GARCH/GJR-GARCH (the standard variance
    recursion). EGARCH's recursion operates in LOG-variance space, where
    persistence is governed by beta alone, not alpha+beta - using the same
    alpha+beta formula for it would overstate persistence (it can exceed 1
    without implying explosive variance), so this is detected from the
    fitted model type and computed accordingly.
    """
    p = res.params
    omega = float(p.get("omega", np.nan))
    alpha = float(p.get("alpha[1]", 0.0))
    beta = float(p.get("beta[1]", 0.0))
    gamma = float(p.get("gamma[1]", 0.0))
    nu = float(p.get("nu", np.nan))

    is_egarch = type(res.model.volatility).__name__.upper() == "EGARCH"
    persistence = beta if is_egarch else (alpha + beta)
    half_life = float(np.log(0.5) / np.log(persistence)) if 0 < persistence < 1 else float("nan")

    return dict(
        omega=omega, alpha=alpha, beta=beta, gamma=gamma, nu=nu,
        persistence=persistence, persistence_definition=("beta (EGARCH, log-variance)" if is_egarch else "alpha+beta"),
        half_life_days=half_life,
        aic=float(res.aic), bic=float(res.bic), loglikelihood=float(res.loglikelihood),
        pvalues={str(k): float(v) for k, v in res.pvalues.items()},
    )


def rolling_variance_forecast(
    returns_pct_full: pd.Series,
    split_idx: int,
    spec: dict,
    dist: str = config.GARCH_DIST,
    refit_every: int = config.GARCH_REFIT_EVERY,
) -> pd.Series:
    """1-day-ahead rolling variance forecasts over the test block
    (positions `split_idx:` of `returns_pct_full`).

    Reproduces Listing 4.1's walk-forward, but bounded the way the
    blueprint's own prose calls for (sec. 4.3: "In practice refit weekly
    and use fixed-parameter filtering between refits."): parameters are
    re-optimized only every `refit_every` steps; within a block, the
    conditional-variance recursion is *filtered* forward with those fixed
    parameters via `arch`'s `.fix()` + a single vectorized `.forecast(...,
    start=...)` call covering the whole block, rather than looping day by
    day (`refit_every=1` reproduces Listing 4.1 literally - full
    re-estimation at every step - through the same code path, just with
    one-row blocks; only use it on short test windows).

    Returned series is indexed by the ORIGIN date (the date whose
    information the forecast is conditioned on), matching every other
    "known as of t" column in the pipeline (see `features.py`); each value
    is the forecast for t+1. Use `features.shift_to_target_dates` if you
    need it aligned to the date it forecasts instead.
    """
    from src.walkforward import refit_checkpoints

    test_len = len(returns_pct_full) - split_idx
    chunks = []

    for start, end in refit_checkpoints(test_len, refit_every):
        fit_data = returns_pct_full.iloc[: split_idx + start]
        res = arch_model(fit_data, dist=dist, **spec).fit(disp="off")

        block_data = returns_pct_full.iloc[: split_idx + end]
        fixed = arch_model(block_data, dist=dist, **spec).fix(res.params)
        fc = fixed.forecast(horizon=1, start=split_idx + start, reindex=False)
        chunks.append(fc.variance.iloc[:, 0])

    variances = pd.concat(chunks)
    variances.index = returns_pct_full.index[split_idx : split_idx + len(variances)]
    return variances.rename("garch_variance_pct2")


def realized_variance_proxies(ret_pct: pd.Series, window: int = 21) -> pd.DataFrame:
    """Realized-volatility proxies to evaluate GARCH forecasts against
    (sec. 4.4): squared return, |return|, and a trailing `window`-day
    realized variance - all on the same percent-return scale as
    `rolling_variance_forecast`'s output (variance units, i.e. percent^2).
    """
    return pd.DataFrame(
        {
            "abs_ret": ret_pct.abs(),
            "sq_ret": ret_pct**2,
            f"realized_var_{window}d": ret_pct.rolling(window).var(),
        },
        index=ret_pct.index,
    )


def naive_rolling_variance_benchmark(ret_pct_full: pd.Series, split_idx: int, window: int = 21) -> pd.Series:
    """Naive volatility benchmark for sec. 5.1's comparison: the trailing
    `window`-day variance through origin date t stands in as the "forecast"
    for t+1's variance (no GARCH dynamics at all) - what every GARCH
    variant must beat to claim skill on the variance side, mirroring the
    naive price random walk's role in Task 3. Origin-indexed (Convention
    B), exactly like `rolling_variance_forecast`, so the two line up
    directly for QLIKE/MSE comparison.
    """
    rolling_var = ret_pct_full.rolling(window).var()
    return rolling_var.iloc[split_idx:].rename(f"naive_rolling_var_{window}d")
