"""Tests for item 1: the quintile long-short spread objective.

_quintile_spread splits companies by score into top/bottom buckets and
returns the difference in mean forward return. Quintiles kick in at n>=25;
below that it falls back to terciles; below n=6 it returns NaN.
"""
import math

import numpy as np

from analyzer.correlation import _quintile_spread


def test_tercile_fallback_below_25():
    # n=6 -> terciles, q = 6//3 = 2. scores == returns == 1..6.
    # top 2 returns = {6,5} mean 5.5; bottom 2 = {2,1} mean 1.5; spread = 4.0
    scores = [1, 2, 3, 4, 5, 6]
    assert math.isclose(_quintile_spread(scores, scores), 4.0, abs_tol=1e-9)


def test_tercile_at_n24_boundary():
    # n=24 (< 25) still terciles, q = 24//3 = 8.
    # top 8 = 17..24 mean 20.5; bottom 8 = 1..8 mean 4.5; spread = 16.0
    scores = list(range(1, 25))
    assert math.isclose(_quintile_spread(scores, scores), 16.0, abs_tol=1e-9)


def test_quintile_split_at_25():
    # n=25 -> quintiles, q = 25//5 = 5.
    # top 5 = 21..25 mean 23; bottom 5 = 1..5 mean 3; spread = 20.0
    scores = list(range(1, 26))
    assert math.isclose(_quintile_spread(scores, scores), 20.0, abs_tol=1e-9)


def test_too_few_points_returns_nan():
    assert math.isnan(_quintile_spread([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]))


def test_length_mismatch_returns_nan():
    assert math.isnan(_quintile_spread([1, 2, 3], [1, 2]))


def test_negative_spread_when_score_anticorrelated():
    # Score ascending, return descending -> top scorers have low returns
    scores = [1, 2, 3, 4, 5, 6]
    returns = [6, 5, 4, 3, 2, 1]
    assert _quintile_spread(scores, returns) < 0


def test_nan_pairs_dropped():
    scores = [1, 2, 3, 4, 5, 6, np.nan]
    returns = [1, 2, 3, 4, 5, 6, 100.0]
    # the NaN-score row is dropped, leaving the clean n=6 tercile case
    assert math.isclose(_quintile_spread(scores, returns), 4.0, abs_tol=1e-9)
