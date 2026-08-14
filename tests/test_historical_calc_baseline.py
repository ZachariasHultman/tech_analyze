"""Baseline regression tests for analyzer/historical_calc.py's
price_cagr_window.
"""
import pandas as pd

from analyzer.historical_calc import price_cagr_window


# ---------------------------------------------------------------- price_cagr_window

def test_price_cagr_window_total():
    close = pd.Series(
        [100.0, 150.0, 200.0],
        index=pd.to_datetime(["2021-01-01", "2022-06-01", "2024-01-01"]),
    ).sort_index()
    start = pd.Timestamp("2021-01-01")
    end = pd.Timestamp("2024-01-01")

    result = price_cagr_window(close, start, end, 3)

    years = (end - start).days / 365.25
    expected = (200.0 / 100.0) ** (1 / years) - 1
    assert result == expected


def test_price_cagr_window_yoy():
    close = pd.Series(
        [100.0, 150.0, 180.0, 200.0],
        index=pd.to_datetime(["2021-01-01", "2022-06-01", "2023-06-01", "2024-01-01"]),
    ).sort_index()
    start = pd.Timestamp("2023-01-01")
    end = pd.Timestamp("2024-01-01")

    result = price_cagr_window(close, start, end, 1)

    # prev = last close strictly before start (150 @ 2022-06-01)
    # last_val = last close in [start, end] (200 @ 2024-01-01)
    years = (end - start).days / 365.25
    expected = (200.0 / 150.0) ** (1 / years) - 1
    assert result == expected


def test_price_cagr_window_empty_returns_none():
    assert price_cagr_window(None, pd.Timestamp("2021-01-01"), pd.Timestamp("2022-01-01"), 1) is None
    assert price_cagr_window(pd.Series(dtype=float), pd.Timestamp("2021-01-01"), pd.Timestamp("2022-01-01"), 1) is None


def test_price_cagr_window_yoy_no_prev_returns_none():
    close = pd.Series([200.0], index=pd.to_datetime(["2024-01-01"]))
    result = price_cagr_window(close, pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01"), 1)
    assert result is None
