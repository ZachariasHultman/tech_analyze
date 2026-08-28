"""Baseline regression tests for _cagr (analyzer/financial_metrics.py).

Covers all four sign branches documented in the function's docstring.
"""
import math

import pytest

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


# ---------------------------------------------------------------------------
# calculate_PE / calculate_de / calculate_roe
#
# All three are plain dict-in, value-out readers over Avanza's yearly blocks --
# no API client, no network -- and all three were effectively untested (93%,
# 90% and 94% of their lines uncovered) while sitting directly under the live
# score. calculate_PE alone feeds three of the five VALUE_METRICS
# (`fcfy_pe`, `cagr_pe`, `roe_pe`), so a wrong value there moves most of the
# value sleeve at once, and the panel cannot catch it: the panel reads `pe`
# out of snapshots this function produced.
# ---------------------------------------------------------------------------
from analyzer.financial_metrics import calculate_PE, calculate_de, calculate_roe


def _year(date, value, report_type="FULL_YEAR"):
    return {"date": date, "value": value, "reportType": report_type}


def _key_ratios(entries):
    return {"stockKeyRatiosByYear": {"priceEarningsRatio": entries}}


def _financials(**series):
    return {"companyFinancialsByYear": series}


# --- calculate_PE ----------------------------------------------------------

def test_pe_keeps_only_full_year_entries():
    """Quarterly rows share the block and must not reach the score."""
    latest, hist = calculate_PE(_key_ratios([
        _year("2023-12-31", 10.0),
        _year("2024-03-31", 99.0, report_type="Q1"),
        _year("2024-12-31", 12.0),
    ]))
    assert latest == [10.0, 12.0]
    assert [h["date"] for h in hist] == ["2023-12-31", "2024-12-31"]


def test_pe_truncates_to_the_last_five_years():
    """`latest_vals` is capped at 5 while the history is returned in full."""
    entries = [_year(f"20{y:02d}-12-31", float(y)) for y in range(15, 22)]
    latest, hist = calculate_PE(_key_ratios(entries))
    assert latest == [17.0, 18.0, 19.0, 20.0, 21.0]
    assert len(hist) == 7


def test_pe_returns_a_list_not_a_scalar():
    """Deliberate asymmetry: PE hands back a list where de/roe hand back a
    float. Downstream unwrapping depends on it, so it is pinned here."""
    latest, _ = calculate_PE(_key_ratios([_year("2024-12-31", 12.0)]))
    assert latest == [12.0]


def test_pe_drops_entries_missing_date_or_value():
    latest, hist = calculate_PE(_key_ratios([
        {"date": "2023-12-31", "reportType": "FULL_YEAR"},        # no value
        {"value": 5.0, "reportType": "FULL_YEAR"},                 # no date
        _year("2024-12-31", 12.0),
    ]))
    assert latest == [12.0]
    assert len(hist) == 1


def test_pe_with_no_full_year_rows_returns_none_pair():
    assert calculate_PE(_key_ratios([_year("2024-03-31", 9.0, "Q1")])) == (None, None)


def test_pe_does_not_sort_by_date():
    """Order comes from Avanza and is never re-sorted, so `[-5:]` means "last
    five as delivered", not "five most recent". Pinned as the contract it is;
    if the feed ever stops being chronological this test is where it shows.
    """
    latest, _ = calculate_PE(_key_ratios([
        _year("2024-12-31", 12.0),
        _year("2023-12-31", 10.0),
    ]))
    assert latest == [12.0, 10.0]


# --- calculate_de ----------------------------------------------------------

def test_de_returns_latest_scalar_and_full_history():
    latest, hist = calculate_de(_financials(debtToEquityRatio=[
        _year("2023-12-31", 1.5),
        _year("2024-12-31", 0.8),
    ]))
    assert latest == 0.8
    assert isinstance(latest, float)
    assert len(hist) == 2


def test_de_ignores_non_full_year_rows():
    latest, hist = calculate_de(_financials(debtToEquityRatio=[
        _year("2024-06-30", 9.9, "Q2"),
        _year("2024-12-31", 0.8),
    ]))
    assert latest == 0.8
    assert len(hist) == 1


def test_de_empty_series_returns_none_pair():
    assert calculate_de(_financials(debtToEquityRatio=[])) == (None, None)


# --- calculate_roe ---------------------------------------------------------

def test_roe_computes_profit_over_equity_per_year():
    """equity = totalAssets - totalLiabilities, year by year."""
    latest, series = calculate_roe(_financials(
        netProfit=[_year("2023-12-31", 100.0), _year("2024-12-31", 240.0)],
        totalAssets=[_year("2023-12-31", 1000.0), _year("2024-12-31", 1200.0)],
        totalLiabilities=[_year("2023-12-31", 500.0), _year("2024-12-31", 400.0)],
    ))
    assert series[0]["value"] == pytest.approx(100.0 / 500.0)
    assert latest == pytest.approx(240.0 / 800.0)


def test_roe_zero_equity_yields_none_for_that_year():
    """A company whose liabilities exactly match its assets divides by zero;
    the year must come back None rather than raising or reading as 0."""
    latest, series = calculate_roe(_financials(
        netProfit=[_year("2024-12-31", 100.0)],
        totalAssets=[_year("2024-12-31", 500.0)],
        totalLiabilities=[_year("2024-12-31", 500.0)],
    ))
    assert series[-1]["value"] is None
    assert latest is None


def test_roe_missing_any_of_the_three_series_returns_none_pair():
    assert calculate_roe(_financials(
        netProfit=[_year("2024-12-31", 100.0)],
        totalAssets=[],
        totalLiabilities=[_year("2024-12-31", 400.0)],
    )) == (None, None)


def test_roe_pairs_the_three_series_by_date_not_by_index():
    """The three blocks are independent series with independent year coverage.

    Here totalAssets and totalLiabilities begin a year later than netProfit.
    Paired by list position -- the old behaviour -- 2023's profit was divided
    by 2024's equity and filed under the 2023 date, silently, while 2024 was
    dropped by an IndexError. Paired by date, 2023 simply has no ROE (there is
    no balance sheet for it) and 2024 is computed correctly.
    """
    latest, series = calculate_roe(_financials(
        netProfit=[_year("2023-12-31", 100.0), _year("2024-12-31", 200.0)],
        totalAssets=[_year("2024-12-31", 1000.0)],
        totalLiabilities=[_year("2024-12-31", 900.0)],
    ))
    assert [s["date"] for s in series] == ["2024-12-31"]
    assert series[0]["value"] == pytest.approx(200.0 / 100.0)
    assert latest == pytest.approx(2.0)


def test_roe_ignores_a_trailing_balance_sheet_year_with_no_profit():
    """The shape actually seen live: totalAssets/totalLiabilities carry one
    report more than netProfit. Every profit year still pairs with its own
    equity, and the extra balance-sheet year contributes nothing."""
    latest, series = calculate_roe(_financials(
        netProfit=[_year("2023-12-31", 100.0), _year("2024-12-31", 240.0)],
        totalAssets=[_year("2023-12-31", 1000.0), _year("2024-12-31", 1200.0),
                     _year("2025-12-31", 1400.0)],
        totalLiabilities=[_year("2023-12-31", 500.0), _year("2024-12-31", 400.0),
                          _year("2025-12-31", 300.0)],
    ))
    assert [s["date"] for s in series] == ["2023-12-31", "2024-12-31"]
    assert series[0]["value"] == pytest.approx(100.0 / 500.0)
    assert latest == pytest.approx(240.0 / 800.0)


def test_roe_skips_a_year_whose_balance_sheet_is_missing_mid_series():
    """A gap in the middle is the case index pairing corrupted worst: every
    year after the gap shifted by one. Now the gap year is simply absent."""
    latest, series = calculate_roe(_financials(
        netProfit=[_year("2022-12-31", 50.0), _year("2023-12-31", 100.0),
                   _year("2024-12-31", 240.0)],
        totalAssets=[_year("2022-12-31", 600.0), _year("2024-12-31", 1200.0)],
        totalLiabilities=[_year("2022-12-31", 100.0), _year("2024-12-31", 400.0)],
    ))
    assert [s["date"] for s in series] == ["2022-12-31", "2024-12-31"]
    assert series[0]["value"] == pytest.approx(50.0 / 500.0)
    assert latest == pytest.approx(240.0 / 800.0)


def test_roe_returns_none_pair_when_no_year_has_both_sides():
    assert calculate_roe(_financials(
        netProfit=[_year("2023-12-31", 100.0)],
        totalAssets=[_year("2024-12-31", 1000.0)],
        totalLiabilities=[_year("2024-12-31", 900.0)],
    )) == (None, None)
