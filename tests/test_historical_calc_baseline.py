"""Baseline regression tests for analyzer/historical_calc.py helpers whose
behavior is NOT changed by the look-ahead fix (item 1): make_windows,
price_cagr_window, slice_df_between.
"""
import pandas as pd

from analyzer.historical_calc import make_windows, price_cagr_window, slice_df_between


# ---------------------------------------------------------------- make_windows

def test_make_windows_span_1_skips_redundant_yoy():
    max_date = pd.Timestamp("2024-01-01")
    out = make_windows(max_date, 1)
    assert len(out) == 1
    label, start, end, yrs = out[0]
    assert label == "1Y_TOTAL"
    assert start == max_date - pd.DateOffset(years=1)
    assert end == max_date
    assert yrs == 1


def test_make_windows_span_3_labels_and_spans():
    max_date = pd.Timestamp("2024-01-01")
    out = make_windows(max_date, 3)
    labels = [o[0] for o in out]
    assert labels == ["3Y_TOTAL", "3Y_YoY-1", "3Y_YoY-2", "3Y_YoY-3"]

    total = out[0]
    assert total[1] == max_date - pd.DateOffset(years=3)
    assert total[2] == max_date
    assert total[3] == 3

    yoy1 = out[1]
    assert yoy1[1] == max_date - pd.DateOffset(years=1)
    assert yoy1[2] == max_date
    assert yoy1[3] == 1

    yoy2 = out[2]
    assert yoy2[1] == max_date - pd.DateOffset(years=2)
    assert yoy2[2] == max_date - pd.DateOffset(years=1)
    assert yoy2[3] == 1

    yoy3 = out[3]
    assert yoy3[1] == max_date - pd.DateOffset(years=3)
    assert yoy3[2] == max_date - pd.DateOffset(years=2)
    assert yoy3[3] == 1


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


# ---------------------------------------------------------------- slice_df_between

def test_slice_df_between_date_column_inclusive_boundaries():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01"]),
        "value": [1.0, 2.0, 3.0],
    })
    out = slice_df_between(df, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-01"))
    assert list(out["value"]) == [1.0, 2.0]


def test_slice_df_between_date_index():
    df = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01"]),
    )
    out = slice_df_between(df, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-01"))
    assert list(out["value"]) == [1.0, 2.0]


def test_slice_df_between_non_dataframe_passthrough():
    assert slice_df_between(None, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-01")) is None
    assert slice_df_between(5.0, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-01")) == 5.0
