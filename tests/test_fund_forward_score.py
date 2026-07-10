"""Tests for item 4: the forward-fundamentals target _fund_forward_score."""
import math

from analyzer.historical_calc import _fund_forward_score


def test_all_components_pass_scores_4():
    score = _fund_forward_score(
        rev_start=100, rev_end=110,   # revenue grew
        nm_start=0.10, nm_end=0.10,   # margin stable
        nde_start=1.0, nde_end=1.5,   # leverage roughly flat
        dps_start=1.0, dps_end=1.0,   # dividend held
        years=1,
    )
    assert math.isclose(score, 4.0, abs_tol=1e-9)


def test_everything_deteriorates_scores_0():
    score = _fund_forward_score(
        rev_start=100, rev_end=90,    # revenue shrank
        nm_start=0.10, nm_end=0.05,   # margin collapsed
        nde_start=1.0, nde_end=3.0,   # leverage blew out
        dps_start=2.0, dps_end=1.0,   # dividend cut
        years=1,
    )
    assert math.isclose(score, 0.0, abs_tol=1e-9)


def test_missing_dividend_rescales_not_nan():
    # No dividend data at all -> that component is excluded; the remaining 3
    # (rev pass, margin fail, leverage pass) rescale to 2 * 4/3.
    score = _fund_forward_score(
        rev_start=100, rev_end=110,
        nm_start=0.10, nm_end=0.05,   # fails
        nde_start=1.0, nde_end=1.2,   # passes
        dps_start=None, dps_end=None,  # missing -> excluded
        years=1,
    )
    assert not math.isnan(score)
    assert math.isclose(score, 2 * 4.0 / 3, abs_tol=1e-9)


def test_never_paid_dividend_counts_as_pass_when_some_data():
    # dps both 0 (series exists, never paid) counts as available + pass.
    score = _fund_forward_score(
        rev_start=100, rev_end=110,
        nm_start=0.10, nm_end=0.10,
        nde_start=1.0, nde_end=1.0,
        dps_start=0.0, dps_end=0.0,
        years=1,
    )
    assert math.isclose(score, 4.0, abs_tol=1e-9)


def test_fewer_than_two_components_returns_nan():
    # Only revenue available; everything else missing -> NaN.
    score = _fund_forward_score(
        rev_start=100, rev_end=110,
        nm_start=None, nm_end=None,
        nde_start=None, nde_end=None,
        dps_start=None, dps_end=None,
        years=1,
    )
    assert math.isnan(score)
