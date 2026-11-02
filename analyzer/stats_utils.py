"""Pure statistical helpers for the panel-validation pipeline.

Deliberately decoupled from panel.py / correlation.py / SummaryManager / any
CSV: Fama-MacBeth (used by validation) is closed-form numpy.linalg +
scipy.stats, directly testable on hand-computed fixtures. No new dependency:
numpy + scipy only (both already pinned).
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

def per_period_ols(X, y, min_obs=None):
    """One cross-section's OLS via numpy.linalg.lstsq (adds its own intercept).

    Drops rows with any NaN in X or y first. Returns the coefficient vector
    ``[intercept, b1, ..., bk]`` as a 1-D numpy array, or ``None`` when fewer
    than ``min_obs`` valid rows remain (default ``X.shape[1] + 2``).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    k = X.shape[1]
    if min_obs is None:
        min_obs = k + 2

    valid = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    Xv, yv = X[valid], y[valid]
    if Xv.shape[0] < min_obs:
        return None

    design = np.column_stack([np.ones(Xv.shape[0]), Xv])
    coefs, *_ = np.linalg.lstsq(design, yv, rcond=None)
    return coefs


def _standardize_col(col):
    col = np.asarray(col, dtype=float)
    m = np.nanmean(col)
    s = np.nanstd(col, ddof=0)
    if s == 0 or np.isnan(s):
        return np.zeros_like(col)
    return (col - m) / s


def fama_macbeth(panel_df, x_cols, y_col, period_col, standardize=True):
    """Fama-MacBeth: per-period cross-sectional OLS, then a t-test on the
    time series of each coefficient.

    For each period (group by ``period_col``) run ``per_period_ols``; when
    ``standardize`` is True, z-score each x column *within* that period first
    (so coefficients are comparable across periods and across metrics of very
    different raw scale, e.g. composite score vs a 0-8 Piotroski). Periods with
    too few observations are skipped (counted, not silently dropped). Finally
    ``ttest_1samp(coefs, 0)`` per x column.

    Returns a dict::

        {
          "per_factor": {x_col: {mean, std, t_stat, p_value, n_periods}},
          "n_periods_used": int,
          "n_periods_skipped": int,
        }
    """
    x_cols = list(x_cols)
    coef_series = {c: [] for c in x_cols}
    n_used = 0
    n_skipped = 0

    for _, grp in panel_df.groupby(period_col):
        X = grp[x_cols].to_numpy(dtype=float)
        y = grp[y_col].to_numpy(dtype=float)
        if standardize:
            X = np.column_stack([_standardize_col(X[:, j]) for j in range(X.shape[1])])
        coefs = per_period_ols(X, y)
        if coefs is None:
            n_skipped += 1
            continue
        n_used += 1
        for j, c in enumerate(x_cols):
            coef_series[c].append(coefs[j + 1])  # skip intercept

    per_factor = {}
    for c in x_cols:
        arr = np.asarray(coef_series[c], dtype=float)
        if arr.size == 0:
            per_factor[c] = {
                "mean": np.nan, "std": np.nan, "t_stat": np.nan,
                "p_value": np.nan, "n_periods": 0,
            }
            continue
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        if arr.size > 1:
            t_stat, p_value = sp_stats.ttest_1samp(arr, popmean=0.0)
            t_stat, p_value = float(t_stat), float(p_value)
        else:
            t_stat, p_value = np.nan, np.nan
        per_factor[c] = {
            "mean": mean, "std": std, "t_stat": t_stat,
            "p_value": p_value, "n_periods": int(arr.size),
        }

    return {
        "per_factor": per_factor,
        "n_periods_used": n_used,
        "n_periods_skipped": n_skipped,
    }
