"""Tests for item 4: PSR / expected-max-Sharpe / deflated Sharpe (pure math)."""
import math

import numpy as np

from analyzer.stats_utils import (
    expected_max_sharpe_under_trials,
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
)


def test_expected_max_sharpe_single_trial_is_zero():
    assert expected_max_sharpe_under_trials(1.0, 1) == 0.0
    assert expected_max_sharpe_under_trials(0.0, 100) == 0.0


def test_expected_max_sharpe_increases_with_trials():
    e10 = expected_max_sharpe_under_trials(1.0, 10)
    e100 = expected_max_sharpe_under_trials(1.0, 100)
    e1000 = expected_max_sharpe_under_trials(1.0, 1000)
    assert 0 < e10 < e100 < e1000


def test_expected_max_sharpe_scales_linearly_with_sigma():
    base = expected_max_sharpe_under_trials(1.0, 50)
    assert math.isclose(expected_max_sharpe_under_trials(2.5, 50), 2.5 * base, rel_tol=1e-12)


def test_psr_half_when_sr_equals_benchmark():
    # numerator 0 -> Phi(0) = 0.5 exactly, regardless of moments.
    assert math.isclose(
        probabilistic_sharpe_ratio(0.5, 0.5, 100, 0.0, 3.0), 0.5, abs_tol=1e-12
    )


def test_psr_hand_computed_z():
    # sr_hat=0.5, bench=0, T=101, skew=0, kurt=3.
    # denom = sqrt(1 + 0.5*0.25) = sqrt(1.125); z = 0.5*sqrt(100)/sqrt(1.125).
    from scipy import stats as sp_stats
    z = 0.5 * math.sqrt(100) / math.sqrt(1.125)
    expected = float(sp_stats.norm.cdf(z))
    assert math.isclose(
        probabilistic_sharpe_ratio(0.5, 0.0, 101, 0.0, 3.0), expected, abs_tol=1e-12
    )
    assert expected > 0.99


def test_dsr_significant_for_strong_consistent_series():
    rng = np.random.default_rng(0)
    # a strongly positive, low-noise return series -> high Sharpe.
    returns = 0.05 + 0.01 * rng.standard_normal(200)
    # modest trial dispersion / few trials -> low deflation benchmark.
    trials = 0.1 * rng.standard_normal(20)
    res = deflated_sharpe_ratio(trials, returns)
    assert res["n_trials"] == 20
    assert res["t_periods"] == 200
    assert res["significant_at_95"] is True
    assert res["dsr"] > 0.95


def test_dsr_not_significant_for_noise_series():
    rng = np.random.default_rng(1)
    returns = 0.0 + 1.0 * rng.standard_normal(60)  # ~zero mean, high noise
    trials = 1.0 * rng.standard_normal(500)  # many noisy trials -> high benchmark
    res = deflated_sharpe_ratio(trials, returns)
    assert res["significant_at_95"] is False


def test_dsr_degenerate_series_returns_not_significant():
    res = deflated_sharpe_ratio([0.1, 0.2, 0.3], [1.0])  # single return -> undefined
    assert res["significant_at_95"] is False
    assert math.isnan(res["dsr"])
