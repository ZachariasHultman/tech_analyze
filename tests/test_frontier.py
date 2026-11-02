"""Portfolio weights over the pushed watchlist (analyzer/frontier.py)."""

import numpy as np
import pandas as pd
import pytest

import analyzer.main as main_mod
from analyzer import frontier
from analyzer.frontier import (
    avanza_close_to_sek, build_portfolios, display_name, format_portfolios,
    format_portfolios_html, ledoit_wolf, weekly_returns,
)


def _ohlc(rets, start="2021-01-04"):
    """Avanza-shaped OHLC frame (``date`` column) from daily returns."""
    dates = pd.bdate_range(start, periods=len(rets))
    close = 100 * np.cumprod(1 + np.asarray(rets))
    return pd.DataFrame({"date": dates.date, "close": close})


def _universe(n=6, days=900, seed=0, vols=None):
    rng = np.random.default_rng(seed)
    vols = vols or [0.01 + 0.004 * i for i in range(n)]
    market = rng.normal(0, 0.006, days)
    hist = {f"S{i}": _ohlc(market + rng.normal(0, v, days)) for i, v in enumerate(vols)}
    return hist, {k: "SEK" for k in hist}


def test_ledoit_wolf_positive_definite_when_fewer_obs_than_assets():
    X = np.random.default_rng(1).normal(size=(20, 40))
    S, delta = ledoit_wolf(X)
    assert 0 < delta <= 1
    assert np.linalg.eigvalsh(S).min() > 0


def test_weights_long_only_sum_to_one_and_respect_cap():
    hist, ccy = _universe(n=10)
    blend = pd.Series(np.linspace(0.1, 0.9, 10), index=list(hist))
    p = build_portfolios(hist, ccy, blend, list(hist), None, cap=0.15)
    for key in ("minvar", "maxsharpe"):
        w = p[key]["weights"]
        assert w.sum() == pytest.approx(1.0)
        assert (w >= 0).all() and (w <= 0.15 + 1e-9).all()


def test_cap_below_one_over_n_is_lifted_to_feasible():
    hist, ccy = _universe(n=4)
    p = build_portfolios(hist, ccy, None, list(hist), None, cap=0.15)
    assert p["cap"] == pytest.approx(0.25)
    assert p["minvar"]["weights"].sum() == pytest.approx(1.0)


def test_minvar_favours_low_vol_and_beats_equal_weight():
    hist, ccy = _universe(n=5, vols=[0.005, 0.02, 0.02, 0.02, 0.02])
    p = build_portfolios(hist, ccy, None, list(hist), None, cap=1.0)
    w = p["minvar"]["weights"]
    assert w.idxmax() == "S0"
    assert p["minvar"]["vol"] < p["equal_vol"]
    assert p["maxsharpe"] is None  # no rank -> B is not invented


def test_maxsharpe_tilts_toward_higher_rank():
    # Identical risk; only the rank differs.
    hist, ccy = _universe(n=4, vols=[0.015] * 4, seed=3)
    blend = pd.Series({"S0": 0.9, "S1": 0.5, "S2": 0.5, "S3": 0.5,
                       "X1": 0.1, "X2": 0.2, "X3": 0.3})
    p = build_portfolios(hist, ccy, blend, list(hist), None, cap=1.0)
    ms, mv = p["maxsharpe"]["weights"], p["minvar"]["weights"]
    assert ms["S0"] > mv["S0"]
    assert p["maxsharpe"]["exp"] > p["minvar"]["exp"]


def test_gbp_avanza_series_is_not_divided_by_100():
    close = pd.Series([5.0, 5.1], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    rates = pd.DataFrame({"GBP": [13.0]}, index=pd.to_datetime(["2024-01-01"]))
    out = avanza_close_to_sek(close, "GBP", rates)
    assert list(out) == pytest.approx([65.0, 66.3])


def test_fx_forward_fills_and_drops_dates_before_rates():
    close = pd.Series([1.0, 2.0, 3.0],
                      index=pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05"]))
    rates = pd.DataFrame({"USD": [10.0, 11.0]},
                         index=pd.to_datetime(["2024-01-02", "2024-01-04"]))
    out = avanza_close_to_sek(close, "USD", rates)
    assert list(out.index) == list(pd.to_datetime(["2024-01-03", "2024-01-05"]))
    assert list(out) == pytest.approx([20.0, 33.0])


def test_missing_rate_leaves_name_unconverted_and_flags_it():
    hist, _ = _universe(n=3)
    ccy = {"S0": "SEK", "S1": "USD", "S2": "SEK"}
    _, unconverted, _ = weekly_returns(hist, ccy, None, list(hist))
    assert unconverted == ["S1"]


def test_short_history_is_dropped_not_allowed_to_shrink_window():
    hist, ccy = _universe(n=4, days=900)
    hist["NEW"] = _ohlc(np.zeros(200) + 0.001, start="2024-06-03")
    ccy["NEW"] = "SEK"
    rets, _, dropped = weekly_returns(hist, ccy, None, list(hist))
    assert dropped == ["NEW"]
    assert len(rets) >= frontier.MIN_WEEKS


def test_too_little_history_returns_none_and_formats():
    hist = {f"S{i}": _ohlc(np.full(100, 0.001)) for i in range(3)}
    p = build_portfolios(hist, {k: "SEK" for k in hist}, None, list(hist), None)
    assert p is None
    assert any("Not computed" in line for line in format_portfolios(None))


def test_format_names_both_portfolios_explicitly():
    hist, ccy = _universe(n=6)
    blend = pd.Series(np.linspace(0.2, 0.9, 6), index=list(hist))
    text = "\n".join(format_portfolios(
        build_portfolios(hist, ccy, blend, list(hist), None)))
    assert "A = MIN-VARIANCE (robust)" in text
    assert "B = MAX-SHARPE (uses rank)" in text
    assert "A min-var" in text and "B max-Sharpe" in text


def test_portfolio_lines_never_raise(monkeypatch):
    class _Broken:
        price_hist = None  # .get on None raises inside
        price_ccy = None

    lines, html_part = main_mod._portfolio_section(_Broken(), {"qualified_names": ["X"]})
    assert any("Not computed" in line for line in lines)
    assert "Not computed" in html_part


def test_listing_currency_prefers_avanza_then_suffix():
    assert main_mod._listing_currency({"listing": {"currency": "nok"}}, "X.ST") == "NOK"
    assert main_mod._listing_currency({"listing": {}}, "GLEN.L") == "GBP"
    assert main_mod._listing_currency({}, None) is None


def test_display_name_strips_avanza_id_only():
    assert display_name("Veidekke 52602") == "Veidekke"
    assert display_name("Münchener Rückversicherungs-Gesellschaft 123") == \
        "Münchener Rückversicherungs-Gesellschaft"
    assert display_name("3M") == "3M"


def test_html_is_a_table_with_both_portfolios_and_clean_names():
    hist, ccy = _universe(n=6)
    hist = {f"{k} 1234": v for k, v in hist.items()}
    ccy = {f"{k} 1234": v for k, v in ccy.items()}
    blend = pd.Series(np.linspace(0.2, 0.9, 6), index=list(hist))
    p = build_portfolios(hist, ccy, blend, list(hist), None)
    out = format_portfolios_html(p)
    assert "<table" in out and out.count("<tr") == 1 + 3 + 1 + len(
        [n for n in hist if p["minvar"]["weights"][n] >= 0.005
         or p["maxsharpe"]["weights"][n] >= 0.005])
    assert "Min-variance" in out and "Max-Sharpe" in out
    assert "robust" in out and "uses rank" in out
    assert "1234" not in out
    assert "<style" not in out  # Gmail strips style blocks


def test_html_escapes_names():
    hist, ccy = _universe(n=3)
    hist = {"A&B <x>" if k == "S0" else k: v for k, v in hist.items()}
    ccy = {k: "SEK" for k in hist}
    out = format_portfolios_html(build_portfolios(hist, ccy, None, list(hist), None))
    assert "A&amp;B &lt;x&gt;" in out


def test_weight_below_half_percent_shows_dash():
    hist, ccy = _universe(n=5, vols=[0.005, 0.03, 0.03, 0.03, 0.03])
    p = build_portfolios(hist, ccy, None, list(hist), None, cap=1.0)
    p["minvar"]["weights"][:] = [0.996, 0.004, 0.0, 0.0, 0.0]
    text = "\n".join(format_portfolios(p))
    assert "0% in both (4)" in text
