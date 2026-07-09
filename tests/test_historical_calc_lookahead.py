"""Regression tests for item 1: kill look-ahead bias in the backtest.

Guards two things:
  1. slice_df_upto's boundary behavior (<=end included, day-after excluded),
     for both the normal DataFrame shape and the stringified-JSON-dict shape
     used by free_cashflow/free_cashflow_yield (which convert_cell never
     turns into a DataFrame, so it silently leaked through the old
     slice_df_between no-op).
  2. That predictors computed as-of start_d (pe/fcfy via _safe_last,
     _build_ticker_dicts fields, cagr) are unaffected by data dated strictly
     after start_d, while total_return (the target variable) still reflects
     data inside the window — sanity check against a trivially-wrong test.
"""
import json
import math

import pandas as pd

from analyzer.historical_calc import (
    slice_df_upto,
    slice_df_between,
    price_cagr_window,
    _build_ticker_dicts,
    _safe_last,
)


# ---------------------------------------------------------------- slice_df_upto boundary

def test_slice_df_upto_boundary_dataframe_shape():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2021-06-01", "2021-06-02"]),
        "value": [2.0, 99.0],
    })
    end = pd.Timestamp("2021-06-01")
    out = slice_df_upto(df, end)
    assert list(out["value"]) == [2.0]


def test_slice_df_upto_boundary_stringified_dict_shape():
    raw = json.dumps({"2021-06-01": 2.0, "2021-06-02": 99.0})
    end = pd.Timestamp("2021-06-01")
    out = slice_df_upto(raw, end)
    assert isinstance(out, pd.Series)
    assert len(out) == 1
    assert out.iloc[0] == 2.0


def test_slice_df_upto_scalar_passthrough():
    assert slice_df_upto(5.0, pd.Timestamp("2021-06-01")) == 5.0
    assert slice_df_upto(None, pd.Timestamp("2021-06-01")) is None


def test_slice_df_upto_date_indexed_dataframe():
    df = pd.DataFrame(
        {"value": [2.0, 99.0]},
        index=pd.to_datetime(["2021-06-01", "2021-06-02"]),
    )
    out = slice_df_upto(df, pd.Timestamp("2021-06-01"))
    assert list(out["value"]) == [2.0]


# ---------------------------------------------------------------- no-look-ahead in predictors

def test_no_lookahead_bias_in_predictors():
    start_d = pd.Timestamp("2022-01-01")
    end_d = pd.Timestamp("2023-01-01")

    # pe/de/roe: normal history before start_d, outlier spike strictly after
    pe_df = pd.DataFrame({
        "date": pd.to_datetime(["2021-06-01", "2021-12-01", "2022-06-01"]),
        "value": [15.0, 16.0, 9999.0],  # spike after start_d
    })
    de_df = pd.DataFrame({
        "date": pd.to_datetime(["2021-06-01", "2021-12-01", "2022-06-01"]),
        "value": [0.5, 0.6, 555.0],  # spike after start_d
    })
    roe_df = pd.DataFrame({
        "date": pd.to_datetime(["2021-06-01", "2021-12-01", "2022-06-01"]),
        "value": [0.10, 0.12, 5.0],  # spike after start_d
    })
    net_profit_df = pd.DataFrame({
        "date": pd.to_datetime(["2021-06-01", "2021-12-01", "2022-06-01"]),
        "value": [100.0, 110.0, 1e9],  # spike after start_d
    })
    total_assets_df = pd.DataFrame({
        "date": pd.to_datetime(["2021-06-01", "2021-12-01", "2022-06-01"]),
        "value": [1000.0, 1100.0, 1e12],  # spike after start_d
    })
    # free_cashflow_yield: stringified dict shape (the extra leak found in item 1)
    fcfy_raw = json.dumps({"2021-06-01": 0.03, "2022-06-01": 999.0})  # spike after start_d

    row = {
        "pe": pe_df,
        "de_ratio": de_df,
        "roe": roe_df,
        "net_profit": net_profit_df,
        "total_assets": total_assets_df,
        "free_cashflow_yield": fcfy_raw,
    }
    pre_metrics = list(row.keys())

    asof = {k: slice_df_upto(row[k], start_d) for k in pre_metrics}

    pe_val = _safe_last(asof.get("pe"))
    de_val = _safe_last(asof.get("de_ratio"))
    roe_val = _safe_last(asof.get("roe"))
    fcfy_val = _safe_last(asof.get("free_cashflow_yield"))

    assert pe_val == 16.0
    assert de_val == 0.6
    assert roe_val == 0.12
    assert fcfy_val == 0.03

    ticker_analysis, _ticker_info = _build_ticker_dicts(asof)
    net_profit_entries = ticker_analysis["companyFinancialsByYear"]["netProfit"]
    total_assets_entries = ticker_analysis["companyFinancialsByYear"]["totalAssets"]
    assert [e["value"] for e in net_profit_entries] == [100.0, 110.0]
    assert [e["value"] for e in total_assets_entries] == [1000.0, 1100.0]

    # ---- price / cagr: spike dated strictly after start_d must not leak ----
    close = pd.Series(
        [50.0, 100.0, 100000.0],
        index=pd.to_datetime(["2020-06-01", "2022-01-01", "2023-01-01"]),
    ).sort_index()

    cagr = price_cagr_window(close, start_d - pd.DateOffset(years=1), start_d, 1)
    years = (start_d - (start_d - pd.DateOffset(years=1))).days / 365.25
    expected_cagr = (100.0 / 50.0) ** (1 / years) - 1
    assert math.isclose(cagr, expected_cagr, abs_tol=1e-9)

    # ---- sanity check: total_return (the target var) DOES reflect the spike ----
    ohlc_df = pd.DataFrame({"close": close})
    ohlc_win = slice_df_between(ohlc_df, start_d, end_d)
    price_start = ohlc_win["close"].iloc[0]
    price_end = ohlc_win["close"].iloc[-1]
    total_return = (price_end / price_start) - 1
    assert total_return > 100  # guards against a trivially-wrong test
