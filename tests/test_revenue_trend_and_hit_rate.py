"""Regression tests for two data-availability fixes in financial_metrics.py:

1. calculate_revenue_trend previously required BOTH quarterly and yearly
   data to return anything, discarding a perfectly usable yearly trend
   whenever quarterly history didn't reach back far enough (Avanza's
   quarterly financials are often only ~2 years deep, much shorter than
   yearly). The two are now computed independently.

2. calculate_revenue_yoy_hit_rate / calculate_eps_yoy_hit_rate required at
   least 8 raw quarters before computing anything, even though the
   underlying math (_rolling_yoy_from_quarterly, a 4-quarter lag) only
   needs 5 to produce one real data point. Real data often has exactly
   7-8 quarters available, just under the old floor. Lowered to 5.
"""
from analyzer.financial_metrics import (
    calculate_revenue_trend,
    calculate_revenue_yoy_hit_rate,
    calculate_eps_yoy_hit_rate,
)


def _yearly_entries(values, start_year=2019):
    return [
        {"date": f"{start_year + i}-02-01", "value": v, "reportType": "FULL_YEAR"}
        for i, v in enumerate(values)
    ]


def _quarterly_entries(values, start="2024-01-01"):
    import pandas as pd

    dates = pd.date_range(start, periods=len(values), freq="QS")
    return [{"date": d.strftime("%Y-%m-%d"), "value": v} for d, v in zip(dates, values)]


# ---------------------------------------------------------------- calculate_revenue_trend

def test_revenue_trend_computes_yearly_when_quarterly_missing():
    ticker_analysis = {
        "companyFinancialsByYear": {"sales": _yearly_entries([100, 110, 120, 130, 140])},
        "companyFinancialsByQuarter": {"sales": []},
    }
    slope_year, slope_quarter, yr, qtr = calculate_revenue_trend(ticker_analysis)
    assert slope_year is not None
    assert slope_year > 0  # growing revenue
    assert slope_quarter is None


def test_revenue_trend_computes_quarterly_when_yearly_missing():
    ticker_analysis = {
        "companyFinancialsByYear": {"sales": []},
        "companyFinancialsByQuarter": {"sales": _quarterly_entries([10, 11, 12, 13, 14, 15, 16])},
    }
    slope_year, slope_quarter, yr, qtr = calculate_revenue_trend(ticker_analysis)
    assert slope_year is None
    assert slope_quarter is not None
    assert slope_quarter > 0


def test_revenue_trend_returns_both_when_both_available():
    ticker_analysis = {
        "companyFinancialsByYear": {"sales": _yearly_entries([100, 110, 120])},
        "companyFinancialsByQuarter": {"sales": _quarterly_entries([10, 11, 12, 13, 14])},
    }
    slope_year, slope_quarter, yr, qtr = calculate_revenue_trend(ticker_analysis)
    assert slope_year is not None
    assert slope_quarter is not None


def test_revenue_trend_all_none_when_both_missing():
    ticker_analysis = {
        "companyFinancialsByYear": {"sales": []},
        "companyFinancialsByQuarter": {"sales": []},
    }
    assert calculate_revenue_trend(ticker_analysis) == (None, None, None, None)


# ---------------------------------------------------------------- hit-rate floor

def test_revenue_hit_rate_computes_with_seven_quarters():
    # 7 quarters, alternating up/down -> exactly matches real-world Avanza
    # quarterly retention depth observed in production data.
    ticker_analysis = {
        "companyFinancialsByQuarter": {
            "sales": _quarterly_entries([10, 9, 11, 12, 10, 13, 14])
        }
    }
    hit, info = calculate_revenue_yoy_hit_rate(ticker_analysis, lookback_quarters=12)
    assert hit is not None
    assert info["reason"] != "not_enough_quarters" if "reason" in info else True


def test_revenue_hit_rate_still_none_below_five_quarters():
    ticker_analysis = {
        "companyFinancialsByQuarter": {"sales": _quarterly_entries([10, 11, 12, 13])}
    }
    hit, info = calculate_revenue_yoy_hit_rate(ticker_analysis)
    assert hit is None
    assert info["reason"] == "not_enough_quarters"


def test_eps_hit_rate_computes_with_five_quarters():
    ticker_analysis = {
        "companyKeyRatiosByQuarterQuarter": {
            "earningsPerShare": _quarterly_entries([1.0, 1.1, 1.2, 1.3, 1.4])
        }
    }
    hit, info = calculate_eps_yoy_hit_rate(ticker_analysis)
    assert hit is not None
