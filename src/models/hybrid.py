"""
Bonus model referenced in the blueprint's own results table (sec. 5.1,
Table 3: "Hybrid (+GARCH sigma feature)") and motivated in sec. 4.4: feed
the GARCH conditional-volatility forecast into the Task 3 feature matrix as
an extra XGBoost feature (Kim & Won 2018; Roszyk & Slepaczuk 2024).
"""
from __future__ import annotations

import pandas as pd

from src.models import tree_models


def add_garch_sigma_feature(frame: pd.DataFrame, sigma_pct: pd.Series) -> pd.DataFrame:
    """Join a 1-day-ahead GARCH conditional-vol forecast onto a feature
    frame. `sigma_pct` must already be leakage-safe: at date `t` it should
    hold the forecast MADE using information through `t` for the return at
    `t+H` (exactly the alignment `garch_models.rolling_variance_forecast`
    produces), matching every other column's "known as of t" contract.
    Rows without a sigma forecast available (e.g. outside the GARCH
    forecasting window) are dropped.
    """
    out = frame.join(sigma_pct.rename("garch_sigma"), how="left")
    return out.dropna(subset=["garch_sigma"])


# Hyperparameter tuning / training / walk-forward for the hybrid model
# reuse the tree_models machinery unchanged - only the input feature frame
# differs (it carries the extra `garch_sigma` column), so there is nothing
# hybrid-specific to add beyond the join above.
tune_hybrid_xgboost = tree_models.tune_xgboost
train_hybrid_xgboost = tree_models.train_xgboost
walk_forward_predict_hybrid = tree_models.walk_forward_predict_tree
