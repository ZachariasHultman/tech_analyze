"""SMA200 metrics on a listing with barely 200 trading days."""

import numpy as np
import pandas as pd
import pytest

from analyzer.financial_metrics import calc_sma200_metrics


def _hist(n):
    return pd.DataFrame({"close": np.linspace(10.0, 12.0, n)})


def test_exactly_200_closes_gives_sma_without_slope():
    sma, week, slope = calc_sma200_metrics(_hist(200))
    assert sma == pytest.approx(11.0)
    assert week is not None
    assert slope is None


def test_201_closes_has_a_slope():
    assert calc_sma200_metrics(_hist(201))[2] is not None


def test_under_200_closes_gives_nothing():
    assert calc_sma200_metrics(_hist(199)) == (None, None, None)
