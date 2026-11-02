# ----------------------------------------------------------------------
#  Scoring helpers shared by the panel (analyzer/panel.py) and validation.
#  The weight optimizers that used to live here were removed: their gate
#  rejected them on every run (last: beat equal weight 3 of 7 held-out years,
#  permutation p=0.149), so live scoring ran equal metric weights regardless.
#  analyzer/evaluation.py (--evaluate) asks the question this data can answer
#  -- is the score useful at all -- instead.
# ----------------------------------------------------------------------

import sys
import os
import warnings

import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from analyzer.summary_manager import SummaryManager
from analyzer.data_processing import calculate_score
from analyzer.metrics import (
    HIGHEST_WEIGHT_METRICS,
    HIGH_WEIGHT_METRICS,
    LOW_WEIGHT_METRICS,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ======================================================================
# Helpers
# ======================================================================


def _quintile_spread(scores, returns):
    """Long-short decision-relevance metric: mean(top bucket) - mean(bottom bucket).

    Sort by score, take the top and bottom buckets, return the difference in
    their mean forward returns. Uses quintiles when n >= 25, terciles below
    that (terciles need ~6 points for 2 per bucket). Returns np.nan when there
    are too few points to split at all (n < 6).
    """
    s = pd.Series(list(scores) if not isinstance(scores, pd.Series) else scores).astype(float)
    r = pd.Series(list(returns) if not isinstance(returns, pd.Series) else returns).astype(float)
    if len(s) != len(r):
        return np.nan
    # Align positionally when indices differ (e.g. plain lists/arrays)
    if not s.index.equals(r.index):
        s = s.reset_index(drop=True)
        r = r.reset_index(drop=True)
    valid = s.notna() & r.notna()
    s, r = s[valid], r[valid]
    n = len(s)
    if n < 6:
        return np.nan
    frac = 5 if n >= 25 else 3
    q = max(n // frac, 1)
    order = s.sort_values(ascending=False)
    top = r.loc[order.index[:q]].mean()
    bot = r.loc[order.index[-q:]].mean()
    return float(top - bot)

def _all_scored_metrics():
    """Every scored metric, in a stable order.

    `sorted`, not `list(set(...))`: Python randomizes string hashing per
    process. The HIGHEST/HIGH/LOW tiers in metrics.py now only define *which*
    metrics are scored -- live and the panel both weight them equally (see
    equal_metric_weights).
    """
    return sorted(
        set(HIGHEST_WEIGHT_METRICS)
        | set(HIGH_WEIGHT_METRICS)
        | set(LOW_WEIGHT_METRICS)
    )


def _score_snapshot(df_slice, metrics_to_score=None, thresholds=None,
                    weight_overrides=None):
    """Score a slice of historical data using SummaryManager.

    Returns a DataFrame with company index and 'points' + per-metric scores.

    weight_overrides: optional {metric: weight}. When None (default) the
    production HIGHEST/HIGH/LOW weight tiers apply exactly as before — every
    existing caller passes nothing, so behavior is unchanged. The new panel
    pipeline passes an equal-weight dict here to score without those tiers.
    """
    sm = SummaryManager()
    if weight_overrides is not None:
        sm._weight_overrides = weight_overrides
    sm.process_historical(df_slice, metrics_to_score or _all_scored_metrics(),
                          thresholds=thresholds)
    calculate_score(sm, metrics_to_score=metrics_to_score)

    out = sm.summary
    if out is None or (isinstance(out, pd.DataFrame) and out.empty):
        return pd.DataFrame()

    frames = []
    s = out if isinstance(out, pd.DataFrame) else pd.DataFrame(out).T
    # Drop columns that are entirely NA to avoid FutureWarning on concat
    s = s.dropna(axis=1, how="all")
    if not s.empty:
        frames.append(s)

    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames)


def equal_metric_weights():
    """{metric: 1.0} for every scored metric -- what live scoring and the
    panel both use. One definition so the two can never drift apart."""
    return {m: 1.0 for m in _all_scored_metrics()}
