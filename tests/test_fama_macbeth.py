"""Tests for item 4: per_period_ols + fama_macbeth (pure math, hand-computed)."""
import math

import numpy as np
import pandas as pd

from analyzer.stats_utils import per_period_ols, fama_macbeth


def test_per_period_ols_recovers_intercept_and_slope():
    # y = 2 + 3x exactly -> intercept 2, slope 3.
    x = np.array([0, 1, 2, 3, 4], dtype=float)
    y = 2 + 3 * x
    coefs = per_period_ols(x, y)
    assert math.isclose(coefs[0], 2.0, abs_tol=1e-9)
    assert math.isclose(coefs[1], 3.0, abs_tol=1e-9)


def test_per_period_ols_too_few_obs_returns_none():
    # k=1 -> min_obs default = 3. Only 2 rows -> None.
    assert per_period_ols(np.array([[0.0], [1.0]]), np.array([0.0, 1.0])) is None


def test_per_period_ols_drops_nan_rows():
    x = np.array([0, 1, 2, 3, np.nan], dtype=float)
    y = np.array([0, 2, 4, 6, 100], dtype=float)  # last row has NaN x -> dropped
    coefs = per_period_ols(x, y)
    assert math.isclose(coefs[1], 2.0, abs_tol=1e-9)


def test_fama_macbeth_slope_ttest_hand_computed():
    # Period 1: y = 2x (slope 2). Period 2: y = 4x (slope 4).
    # coef series = [2, 4]: mean 3, sd(ddof=1)=sqrt(2), se=1, t=3.0.
    x = np.array([0, 1, 2, 3], dtype=float)
    rows = []
    for period, slope in [(1, 2.0), (2, 4.0)]:
        for xi in x:
            rows.append({"period": period, "x": xi, "y": slope * xi})
    df = pd.DataFrame(rows)

    res = fama_macbeth(df, ["x"], "y", "period", standardize=False)
    pf = res["per_factor"]["x"]
    assert math.isclose(pf["mean"], 3.0, abs_tol=1e-9)
    assert math.isclose(pf["std"], math.sqrt(2), abs_tol=1e-9)
    assert math.isclose(pf["t_stat"], 3.0, abs_tol=1e-9)
    assert pf["n_periods"] == 2
    assert res["n_periods_used"] == 2
    assert res["n_periods_skipped"] == 0


def test_fama_macbeth_skips_thin_periods():
    x = np.array([0, 1, 2, 3], dtype=float)
    rows = [{"period": 1, "x": xi, "y": 2 * xi} for xi in x]
    # period 2 has only 1 obs -> below min_obs (3) -> skipped.
    rows.append({"period": 2, "x": 5.0, "y": 10.0})
    df = pd.DataFrame(rows)

    res = fama_macbeth(df, ["x"], "y", "period", standardize=False)
    assert res["n_periods_used"] == 1
    assert res["n_periods_skipped"] == 1
    # single usable period -> t_stat undefined (needs >1).
    assert math.isnan(res["per_factor"]["x"]["t_stat"])
