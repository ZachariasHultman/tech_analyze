"""Regression test for a data-availability fix in financial_metrics.py.

calculate_revenue_trend previously required BOTH quarterly and yearly data to
return anything, discarding a perfectly usable yearly trend whenever quarterly
history didn't reach back far enough (Avanza's quarterly financials are often
only ~2 years deep, much shorter than yearly). The two are now computed
independently.

This file also covered calculate_revenue_yoy_hit_rate / calculate_eps_yoy_hit_rate
and their 5-quarter floor. Both metrics were removed from scoring -- they had 0%
coverage in the backtest panel while being ~100% populated live, so they moved
the live ranking without appearing in anything that validated it.
"""
from analyzer.financial_metrics import calculate_revenue_trend


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
