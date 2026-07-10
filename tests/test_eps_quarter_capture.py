"""Regression tests: quarterly EPS is now captured into --save snapshots and
wired through the historical backtest adapter, so eps_yoy_hit_rate_status can
actually be computed instead of always reading empty (companyKeyRatiosByQuarter
Quarter.earningsPerShare was previously hardcoded to [] in _build_ticker_dicts
because nothing in the snapshot pipeline stored it).
"""
import pandas as pd

from analyzer.data_processing import _extract_quarterly_series
from analyzer.historical_calc import _build_ticker_dicts


def test_extract_quarterly_series_keeps_only_complete_entries():
    ticker_analysis = {
        "companyKeyRatiosByQuarterQuarter": {
            "earningsPerShare": [
                {"date": "2025-04-17", "reportType": "Q1", "value": 5.72},
                {"date": "2025-07-17", "reportType": "Q2", "value": 6.03},
                # not-yet-reported placeholder: no date/value, must be dropped
                {"reportType": "Q3", "financialYear": 2025},
            ]
        }
    }
    out = _extract_quarterly_series(
        ticker_analysis, "companyKeyRatiosByQuarterQuarter", "earningsPerShare"
    )
    assert out == [
        {"date": "2025-04-17", "value": 5.72},
        {"date": "2025-07-17", "value": 6.03},
    ]


def test_extract_quarterly_series_missing_key_returns_none():
    assert _extract_quarterly_series({}, "companyKeyRatiosByQuarterQuarter", "earningsPerShare") is None


def test_build_ticker_dicts_wires_eps_quarter_into_ticker_analysis():
    eps_q_df = pd.DataFrame({
        "date": pd.to_datetime(["2024-04-18", "2024-07-18", "2024-10-17"]),
        "value": [5.36, 6.22, 5.15],
    })
    asof = {"eps_quarter": eps_q_df}

    ticker_analysis, _ticker_info = _build_ticker_dicts(asof)

    eps_quarterly = ticker_analysis["companyKeyRatiosByQuarterQuarter"]["earningsPerShare"]
    assert len(eps_quarterly) == 3
    assert eps_quarterly[0]["value"] == 5.36


def test_build_ticker_dicts_empty_when_eps_quarter_absent():
    ticker_analysis, _ticker_info = _build_ticker_dicts({})
    assert ticker_analysis["companyKeyRatiosByQuarterQuarter"]["earningsPerShare"] == []
