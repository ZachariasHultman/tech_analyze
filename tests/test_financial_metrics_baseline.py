"""Baseline regression tests for _cagr (analyzer/financial_metrics.py).

Covers all four sign branches documented in the function's docstring.
"""
import math

from analyzer.financial_metrics import _cagr


def test_cagr_both_positive():
    result = _cagr(100.0, 121.0, 2)
    assert math.isclose(result, 0.1, abs_tol=1e-9)


def test_cagr_turnaround_negative_to_positive():
    result = _cagr(-50.0, 50.0, 1)
    assert math.isclose(result, 1.0, abs_tol=1e-9)


def test_cagr_deterioration_positive_to_nonpositive():
    assert _cagr(100.0, -10.0, 1) == -1.0
    assert _cagr(100.0, 0.0, 1) == -1.0


def test_cagr_both_negative_improving():
    # loss shrinking: -100 -> -50 is an improvement
    result = _cagr(-100.0, -50.0, 1)
    assert math.isclose(result, 1.0, abs_tol=1e-9)


def test_cagr_both_negative_worsening():
    # loss growing: -50 -> -100 is deterioration
    result = _cagr(-50.0, -100.0, 1)
    assert math.isclose(result, -1.0, abs_tol=1e-9)


def test_cagr_none_or_invalid_years_returns_none():
    assert _cagr(None, 100.0, 2) is None
    assert _cagr(100.0, None, 2) is None
    assert _cagr(100.0, 121.0, 0) is None
    assert _cagr(0.0, 100.0, 2) is None
