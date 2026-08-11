"""Pure statistical helpers for the panel-validation pipeline.

Deliberately decoupled from panel.py / correlation.py / SummaryManager / any
CSV: Fama-MacBeth (used by validation) and the Deflated Sharpe Ratio (used by
the optimizer gate) are the same *kind* of code — closed-form / numpy.linalg +
scipy.stats — so they live together and stay directly testable on hand-computed
fixtures. No new dependency: numpy + scipy only (both already pinned).
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# Euler-Mascheroni constant (Bailey & Lopez de Prado extreme-value approx.)
_EULER_MASCHERONI = 0.5772156649015329


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


def expected_max_sharpe_under_trials(sigma_sr, n_trials):
    """Expected maximum of ``n_trials`` independent noise Sharpe draws.

    Bailey & Lopez de Prado's Euler-Mascheroni extreme-value approximation::

        E[max] ~ sigma_sr * ((1-g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N*e)))

    with ``g`` the Euler-Mascheroni constant and ``Z^-1`` the standard-normal
    ppf. Returns 0.0 for ``n_trials < 2`` (no multiple-testing inflation with a
    single trial) and clamps a non-positive ``sigma_sr`` to 0.0.
    """
    n = int(n_trials)
    if n < 2 or sigma_sr is None or sigma_sr <= 0:
        return 0.0
    g = _EULER_MASCHERONI
    z1 = sp_stats.norm.ppf(1.0 - 1.0 / n)
    z2 = sp_stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(sigma_sr * ((1.0 - g) * z1 + g * z2))


def probabilistic_sharpe_ratio(sr_hat, sr_benchmark, t_periods, skew, kurtosis):
    """Probabilistic Sharpe Ratio: P(true SR > benchmark) given estimation noise.

        PSR = Phi( (SR_hat - SR_bench) * sqrt(T - 1)
                   / sqrt(1 - skew*SR_hat + ((kurtosis - 1)/4) * SR_hat^2) )

    ``kurtosis`` is the full (non-excess) kurtosis (3 for a normal). Returns NaN
    if fewer than 2 periods or a non-positive denominator.
    """
    if t_periods is None or t_periods < 2:
        return float("nan")
    denom_sq = 1.0 - skew * sr_hat + ((kurtosis - 1.0) / 4.0) * sr_hat ** 2
    if denom_sq <= 0:
        return float("nan")
    z = (sr_hat - sr_benchmark) * np.sqrt(t_periods - 1) / np.sqrt(denom_sq)
    return float(sp_stats.norm.cdf(z))


def deflated_sharpe_ratio(trial_objectives, selected_return_series):
    """Deflated Sharpe Ratio: PSR with a multiple-testing-corrected benchmark.

    ``trial_objectives`` is every distinct candidate value the grid sweep
    evaluated — a proxy Sharpe-like statistic, not literal per-trial return
    series (the optimizer's objective is already a cross-validated scalar). Its
    count is the "N trials" for the deflation and its dispersion estimates the
    noise scale; the expected-max-under-N-trials becomes the PSR benchmark.
    ``selected_return_series`` is the chosen strategy's realized per-period
    returns, from which SR_hat, skew and kurtosis are measured.

    Returns a dict including ``significant_at_95``.
    """
    trials = np.asarray(pd.Series(trial_objectives, dtype=float).dropna())
    n_trials = int(trials.size)
    sigma_sr = float(np.std(trials, ddof=1)) if n_trials > 1 else 0.0
    sr_benchmark = expected_max_sharpe_under_trials(sigma_sr, n_trials)

    ret = np.asarray(pd.Series(selected_return_series, dtype=float).dropna())
    t_periods = int(ret.size)
    sd = float(np.std(ret, ddof=1)) if t_periods > 1 else 0.0
    if t_periods < 2 or sd == 0:
        return {
            "dsr": float("nan"),
            "sr_hat": float("nan"),
            "sr_benchmark": sr_benchmark,
            "sigma_sr": sigma_sr,
            "n_trials": n_trials,
            "t_periods": t_periods,
            "significant_at_95": False,
        }

    sr_hat = float(np.mean(ret) / sd)
    skew = float(sp_stats.skew(ret))
    kurtosis = float(sp_stats.kurtosis(ret, fisher=False))
    dsr = probabilistic_sharpe_ratio(sr_hat, sr_benchmark, t_periods, skew, kurtosis)
    return {
        "dsr": dsr,
        "sr_hat": sr_hat,
        "sr_benchmark": sr_benchmark,
        "sigma_sr": sigma_sr,
        "n_trials": n_trials,
        "t_periods": t_periods,
        "significant_at_95": bool(dsr > 0.95) if not np.isnan(dsr) else False,
    }
