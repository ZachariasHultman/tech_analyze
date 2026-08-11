"""Tests for item 2: build_fundamentals_panel (fiscal-year panel construction).

Uses a synthetic get_hist_data (monkeypatched) so the test never depends on the
real gitignored data/*.csv snapshots or on secret metrics.py numeric values —
only on the structural contract (one row per (company, fiscal_year), the key
columns, and the as-of report-date anchoring).
"""
import numpy as np
import pandas as pd

import analyzer.panel as panel


def _ohlc(dates, prices):
    idx = pd.to_datetime(dates)
    return pd.DataFrame({"close": prices}, index=idx)


def _year_series(dates, values):
    return pd.DataFrame({"date": pd.to_datetime(dates), "value": values})


def _synthetic_df():
    # Company X reports 3 fiscal years, company Y reports 2.
    x_dates = ["2021-02-10", "2022-02-10", "2023-02-10"]
    y_dates = ["2022-05-01", "2023-05-01"]
    price_dates = pd.date_range("2019-01-01", "2023-06-01", freq="D")
    x_prices = np.linspace(100, 200, len(price_dates))
    y_prices = np.linspace(50, 80, len(price_dates))
    rows = {
        "X": {
            "company": "X",
            "sector": "Industrials",
            "ohlc": _ohlc(price_dates, x_prices),
            "revenue_year": _year_series(x_dates, [1000, 1100, 1200]),
            "roe": _year_series(x_dates, [0.1, 0.12, 0.14]),
            "pe": _year_series(x_dates, [15, 16, 17]),
        },
        "Y": {
            "company": "Y",
            "sector": "Utilities",
            "ohlc": _ohlc(price_dates, y_prices),
            "revenue_year": _year_series(y_dates, [500, 550]),
            "roe": _year_series(y_dates, [0.08, 0.09]),
            "pe": _year_series(y_dates, [12, 13]),
        },
    }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "company"
    return df


def test_iter_fiscal_years_sorted_unique():
    row = {"revenue_year": _year_series(
        ["2023-02-10", "2021-02-10", "2022-02-10", "2021-02-10"], [3, 1, 2, 1]
    )}
    years = panel._iter_fiscal_years(row)
    assert [pd.Timestamp(d).year for d in years] == [2021, 2022, 2023]


def test_iter_fiscal_years_missing_column():
    assert panel._iter_fiscal_years({}) == []


def test_one_row_per_company_fiscal_year(monkeypatch):
    monkeypatch.setattr(panel, "get_hist_data", lambda data_dir="data": _synthetic_df())
    out = panel.build_fundamentals_panel("ignored")

    # 3 fiscal years for X + 2 for Y = 5 rows.
    assert len(out) == 5
    assert set(out["company_id"]) == {"X", "Y"}
    assert sorted(out[out["company_id"] == "X"]["fiscal_year"]) == [2021, 2022, 2023]
    assert sorted(out[out["company_id"] == "Y"]["fiscal_year"]) == [2022, 2023]

    # Key/schema columns are present.
    for col in ("company_id", "fiscal_year", "report_date", "sector",
                "pe", "roe", "earnings quality status"):
        assert col in out.columns

    # earnings quality status is permanently NaN historically (no OCF).
    assert out["earnings quality status"].isna().all()

    # fiscal_year == report_date.year.
    for _, r in out.iterrows():
        assert pd.Timestamp(r["report_date"]).year == r["fiscal_year"]
