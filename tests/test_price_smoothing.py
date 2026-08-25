"""FCF yield divides by a 60-day mean close, not by spot.

Spot price made FCF yield's market cap a fast-moving input in an otherwise
annual-report-driven score. With watchlist ranks 19-25 packed inside 0.65
points, that noise alone reshuffled the bottom of the list weekly.

(`dividend yield status` was the other smoothed metric; it has since been
removed from scoring outright -- measured IC -0.026 over 7 fiscal years.)
"""

import pandas as pd
import pytest

from analyzer.config import PRICE_SMOOTHING_WINDOW
from analyzer.financial_metrics import smoothed_close


def _closes(values, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame({"close": values}, index=idx)


# ------------------------------------------------------------ smoothed_close

def test_smoothed_close_is_the_mean_of_the_last_window():
    # 100 days: 40 at 10.0 then 60 at 20.0. Only the last 60 count.
    df = _closes([10.0] * 40 + [20.0] * 60)
    assert smoothed_close(df, window=60) == pytest.approx(20.0)


def test_smoothed_close_ignores_everything_before_the_window():
    df = _closes([1000.0] * 40 + [20.0] * 60)
    assert smoothed_close(df, window=60) == pytest.approx(20.0)


def test_smoothed_close_returns_none_below_the_window():
    """A 60-day average computed from 8 days is a different quantity, not a
    noisier estimate of the same one -- silently substituting it would make a
    thinly-listed company look differently priced for a non-price reason."""
    assert smoothed_close(_closes([10.0] * 8), window=60) is None


def test_smoothed_close_handles_missing_and_empty_input():
    assert smoothed_close(None) is None
    assert smoothed_close(pd.DataFrame()) is None
    assert smoothed_close(pd.DataFrame({"open": [1.0] * 100})) is None


def test_smoothed_close_default_window_is_the_configured_one():
    df = _closes([5.0] * (PRICE_SMOOTHING_WINDOW - 1))
    assert smoothed_close(df) is None
    df = _closes([5.0] * PRICE_SMOOTHING_WINDOW)
    assert smoothed_close(df) == pytest.approx(5.0)


def test_smoothing_damps_a_spike_that_spot_would_pass_through():
    """The point of the change: one week of price moves must not move the
    denominator much."""
    flat = [100.0] * 60
    spiked = [100.0] * 55 + [130.0] * 5
    assert smoothed_close(_closes(flat), window=60) == pytest.approx(100.0)
    # spot would read +30%; the 60-day mean moves 2.5%
    assert smoothed_close(_closes(spiked), window=60) == pytest.approx(102.5)
