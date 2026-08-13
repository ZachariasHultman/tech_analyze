"""The panel's forward return is a SEK return — and still works without FX.

Companion to test_panel_scores.py (same monkeypatched-scoring fixtures): that
file pins the per-year scoring/demeaning logic, this one pins the numeraire.
"""
import math

import pandas as pd

import analyzer.fx as fx
import analyzer.panel as panel


def _ohlc(pairs):
    idx = pd.to_datetime([d for d, _ in pairs])
    return pd.DataFrame({"close": [v for _, v in pairs]}, index=idx)


def _series(pairs):
    return pd.Series(
        [v for _, v in pairs], index=pd.to_datetime([d for d, _ in pairs])
    )


# A is US-listed (USD), B is Stockholm-listed (SEK). Same +20% / +10% local
# returns as test_panel_scores, so any difference below is FX and only FX.
_YAHOO = {
    "A": _series([("2021-02-10", 100.0), ("2022-02-10", 120.0),
                  ("2022-06-01", 130.0)]),
    "B": _series([("2021-02-10", 50.0), ("2022-02-10", 55.0),
                  ("2022-06-01", 60.0)]),
}
_SYMBOLS = {"A": "AAPL", "B": "B.ST"}

# USD/SEK 8.00 -> 10.00 across the holding year: a +25% tailwind shared by
# every US name, which is exactly what the demeaning used to mistake for skill.
_RATES = pd.DataFrame(
    {"USD": [8.0, 10.0, 10.0]},
    index=pd.to_datetime(["2021-02-10", "2022-02-10", "2022-06-01"]),
)


def _hist_df():
    a = _ohlc([("2021-02-10", 100), ("2022-02-10", 120), ("2022-06-01", 130)])
    b = _ohlc([("2021-02-10", 50), ("2022-02-10", 55), ("2022-06-01", 60)])
    df = pd.DataFrame({"ohlc": [a, b]}, index=["A", "B"])
    df.index.name = "company"
    return df


def _fundamentals():
    return pd.DataFrame([
        {"company": c, "company_id": c, "fiscal_year": 2021,
         "report_date": "2021-02-10", "sector": "Industrials",
         "roe_pe ratio status": 1.0 + i}
        for i, c in enumerate(["A", "B"])
    ])


def _fake_score(df, metrics_to_score=None, thresholds=None,
                weight_overrides=None):
    companies = df["company"].tolist()
    return pd.DataFrame(
        {"points": [1.0] * len(companies),
         "quality_pct": [0.5] * len(companies),
         "value_pct": [0.5] * len(companies),
         "combined_score": [0.25] * len(companies),
         "roe_pe ratio status_score": [1.0] * len(companies)},
        index=companies,
    )


def _run(monkeypatch, rates):
    monkeypatch.setattr(panel, "get_hist_data", lambda data_dir="data": _hist_df())
    monkeypatch.setattr(panel, "_all_scored_metrics", lambda: ["roe_pe ratio status"])
    monkeypatch.setattr(panel, "_score_snapshot", _fake_score)
    monkeypatch.setattr(
        panel, "load_verified_yahoo_closes",
        lambda data_dir="data": ({k: v.copy() for k, v in _YAHOO.items()}, dict(_SYMBOLS)),
    )
    monkeypatch.setattr(fx, "load_sek_rates", lambda *a, **k: rates)
    return panel.build_scores_panel(_fundamentals(), "ignored").set_index("company_id")


def test_us_return_is_compounded_with_the_usdsek_move(monkeypatch):
    out = _run(monkeypatch, _RATES)
    # (120 * 10.00) / (100 * 8.00) - 1 = +50%, not the +20% USD return
    assert math.isclose(out.loc["A", "fwd_return_1y"], 0.50, abs_tol=1e-9)
    assert out.loc["A", "return_basis"] == "yahoo_adjusted:USD->SEK"


def test_sek_listed_company_is_not_touched(monkeypatch):
    out = _run(monkeypatch, _RATES)
    # multiplying an already-SEK series by anything is the bug, not the fix
    assert math.isclose(out.loc["B", "fwd_return_1y"], 0.10, abs_tol=1e-9)
    assert math.isclose(out.loc["B", "price_at_report"], 50.0, abs_tol=1e-9)
    assert out.loc["B", "return_basis"] == "yahoo_adjusted:SEK"


def test_demeaning_now_happens_in_one_currency(monkeypatch):
    out = _run(monkeypatch, _RATES)
    assert math.isclose(out.loc["A", "universe_mean_return_that_year"], 0.30,
                        abs_tol=1e-9)
    assert math.isclose(out.loc["A", "fwd_excess_return_1y"], 0.20, abs_tol=1e-9)
    assert math.isclose(out.loc["B", "fwd_excess_return_1y"], -0.20, abs_tol=1e-9)


def test_no_fx_cache_degrades_to_todays_behaviour(monkeypatch, capsys):
    out = _run(monkeypatch, None)
    # identical to the pre-FX pipeline: local-currency returns, nothing raised
    assert math.isclose(out.loc["A", "fwd_return_1y"], 0.20, abs_tol=1e-9)
    assert math.isclose(out.loc["B", "fwd_return_1y"], 0.10, abs_tol=1e-9)
    assert (out["return_basis"] == "yahoo_adjusted:unconverted").all()
    assert "no FX cache" in capsys.readouterr().out


def test_unknown_suffix_is_left_alone_and_flagged(monkeypatch, capsys):
    monkeypatch.setitem(_SYMBOLS, "A", "A.XYZ")
    try:
        out = _run(monkeypatch, _RATES)
    finally:
        _SYMBOLS["A"] = "AAPL"
    assert math.isclose(out.loc["A", "fwd_return_1y"], 0.20, abs_tol=1e-9)
    assert out.loc["A", "return_basis"] == "yahoo_adjusted:unconverted:unknown-currency"
    assert "left in listing currency" in capsys.readouterr().out


def test_currency_absent_from_cache_is_left_alone_and_flagged(monkeypatch):
    rates = _RATES.rename(columns={"USD": "EUR"})
    out = _run(monkeypatch, rates)
    assert math.isclose(out.loc["A", "fwd_return_1y"], 0.20, abs_tol=1e-9)
    assert out.loc["A", "return_basis"] == "yahoo_adjusted:unconverted:no-USD-rate"


def test_avanza_fallback_leg_records_that_it_was_not_converted(monkeypatch):
    monkeypatch.setattr(panel, "get_hist_data", lambda data_dir="data": _hist_df())
    monkeypatch.setattr(panel, "_all_scored_metrics", lambda: ["roe_pe ratio status"])
    monkeypatch.setattr(panel, "_score_snapshot", _fake_score)
    monkeypatch.setattr(panel, "load_verified_yahoo_closes",
                        lambda data_dir="data": ({}, {}))
    monkeypatch.setattr(fx, "load_sek_rates", lambda *a, **k: _RATES)
    out = panel.build_scores_panel(_fundamentals(), "ignored")
    assert (out["return_basis"] == "avanza_price_plus_dps:unconverted").all()
