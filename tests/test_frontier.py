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
    assert "B = MAX-SHARPE (uses score)" in text
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
    assert "robust" in out and "uses score" in out
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


class _WatchlistAvanza:
    """Records add/remove calls per watchlist id."""

    def __init__(self, lists):
        # lists: {name: (id, [orderbook ids])}
        self.lists = lists
        self.calls = []

    def get_watchlists(self):
        return [{"name": n, "watchListId": i, "orderbookIds": ids}
                for n, (i, ids) in self.lists.items()]

    def add_to_watchlist(self, oid, wl_id):
        self.calls.append(("add", wl_id, oid))

    def remove_from_watchlist(self, oid, wl_id):
        self.calls.append(("remove", wl_id, oid))


def _portfolio(minvar, maxsharpe):
    return {"minvar": {"weights": pd.Series(minvar)},
            "maxsharpe": {"weights": pd.Series(maxsharpe)} if maxsharpe else None}


def test_watchlists_mirror_held_names_and_drop_the_rest():
    av = _WatchlistAvanza({"min-var": ("10", ["111", "999"]),
                           "max-sharpe": ("20", [])})
    p = _portfolio({"A 111": 0.6, "B 222": 0.4, "C 333": 0.004},
                   {"A 111": 0.1, "C 333": 0.9})
    known = ["A 111", "B 222", "C 333", "Old 999"]
    notes = main_mod._sync_portfolio_watchlists(av, p, known)

    assert set(av.calls) == {
        ("add", "10", "222"),        # B held, not yet on list
        ("remove", "10", "999"),     # Old no longer held
        ("add", "20", "111"), ("add", "20", "333"),
    }
    # C at 0.4% is below the display bar: not pushed to min-var
    assert ("add", "10", "333") not in av.calls
    assert notes[0].startswith("Avanza 'min-var': 2 stocks (+1 added, -1 removed)")
    assert notes[1].startswith("Avanza 'max-sharpe': 2 stocks (+2 added, -0 removed)")


def test_missing_watchlist_warns_and_leaves_the_other_alone():
    av = _WatchlistAvanza({"min-var": ("10", [])})
    p = _portfolio({"A 111": 1.0}, {"A 111": 1.0})
    notes = main_mod._sync_portfolio_watchlists(av, p, ["A 111"])
    assert av.calls == [("add", "10", "111")]
    assert "max-sharpe': list not found" in notes[1]


def test_uncomputed_portfolio_never_empties_a_list():
    av = _WatchlistAvanza({"min-var": ("10", ["111"]), "max-sharpe": ("20", ["111"])})
    assert main_mod._sync_portfolio_watchlists(av, None, ["A 111"]) == []
    # B missing (no usable score): max-sharpe untouched, min-var still synced
    main_mod._sync_portfolio_watchlists(av, _portfolio({"A 111": 1.0}, None), ["A 111"])
    assert av.calls == []


def test_sync_notes_reach_text_and_html():
    hist, ccy = _universe(n=4)
    p = build_portfolios(hist, ccy, None, list(hist), None)
    note = "Avanza 'min-var': 4 stocks (+4 added, -0 removed)"
    assert note in "\n".join(format_portfolios(p, [note]))
    assert note.replace("'", "&#x27;") in format_portfolios_html(p, [note])


def test_network_error_retries_then_skips_and_cleans_up(monkeypatch):
    import requests
    from analyzer.summary_manager import SummaryManager
    calls = {"n": 0}

    def flaky(ticker_id, ticker_info, manager, *a, **k):
        calls["n"] += 1
        manager.summary["X AB 1"] = {"points": [0]}  # half-initialised row
        manager.price_hist["X AB 1"] = "partial"
        raise requests.exceptions.ChunkedEncodingError("broken")

    monkeypatch.setattr(main_mod, "get_data", flaky)
    m, skipped = SummaryManager(), []
    out = main_mod._get_data_or_skip("1", {"name": "X AB", "orderbookId": 1},
                                     m, None, None, False, skipped)
    assert out is None and calls["n"] == 2
    assert "X AB 1" not in m.summary and "X AB 1" not in m.price_hist
    assert skipped[0][0] == "1" and "ChunkedEncodingError" in skipped[0][1]


def test_network_error_recovers_on_retry(monkeypatch):
    import requests
    calls = {"n": 0}

    def once(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.ConnectionError("blip")
        return "X AB 1", None

    monkeypatch.setattr(main_mod, "get_data", once)
    skipped = []
    assert main_mod._get_data_or_skip("1", {"name": "X AB", "orderbookId": 1},
                                      object(), None, None, False, skipped) == ("X AB 1", None)
    assert skipped == []


# --- core-satellite -------------------------------------------------------

from analyzer.benchmarks import core_close
from analyzer.frontier import _solve


def _core_world(seed=4, days=1000):
    """Core = a market factor; S_CORE tracks it, S_IND is independent."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=days)
    mkt = rng.normal(0, 0.008, days)
    core = pd.Series(100 * np.cumprod(1 + mkt), index=dates)
    hist = {
        "S_CORE": _ohlc(mkt + rng.normal(0, 0.002, days)),
        "S_IND": _ohlc(rng.normal(0, 0.009, days)),
        "S_MIX": _ohlc(0.5 * mkt + rng.normal(0, 0.007, days)),
    }
    return hist, {k: "SEK" for k in hist}, core


def test_solver_without_core_is_the_original():
    S = np.array([[0.04, 0.01], [0.01, 0.09]])
    a = _solve(S, 1.0)
    b = _solve(S, 1.0, core=(np.zeros(2), 0.0, 0.0, 1.0))
    assert a == pytest.approx(b)


def test_core_aware_minvar_avoids_what_the_core_already_holds():
    hist, ccy, core = _core_world()
    alone = build_portfolios(hist, ccy, None, list(hist), None, cap=1.0)
    total = build_portfolios(hist, ccy, None, list(hist), None, cap=1.0,
                             core_close=core, satellite_share=0.2)
    w = total["minvar"]["weights"]
    assert w["S_IND"] > w["S_CORE"]
    assert w["S_IND"] > alone["minvar"]["weights"]["S_IND"] - 1e-9
    assert total["minvar"]["corr_core"] < alone["minvar"].get("corr_core", 1.0)
    # the satellite lowers total risk below the core on its own
    assert total["minvar"]["total_vol"] < total["core"]["vol"]
    assert total["core"]["share"] == pytest.approx(0.8)


def test_short_core_falls_back_with_a_note():
    hist, ccy, core = _core_world()
    p = build_portfolios(hist, ccy, None, list(hist), None,
                         core_close=core.iloc[-60:], satellite_share=0.2)
    assert p["core"] is None and "core history too short" in p["core_note"]
    assert "__core__" not in p["minvar"]["weights"].index


def test_core_rows_and_note_render():
    hist, ccy, core = _core_world()
    blend = pd.Series({"S_CORE": 0.9, "S_IND": 0.8, "S_MIX": 0.7, "X": 0.1, "Y": 0.2})
    p = build_portfolios(hist, ccy, blend, list(hist), None, core_close=core,
                         satellite_share=0.2)
    p["core"]["label"] = "70% A + 30% B"
    text = "\n".join(format_portfolios(p))
    assert "Total vol / yr (80% core + 20% these)" in text
    assert "Correlation with core" in text
    assert "Core = 70% A + 30% B, 80% of the total" in text
    assert "WHOLE" in text
    assert "WHOLE portfolio" in format_portfolios_html(p)


def test_core_close_mixes_daily_returns():
    d = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    a = pd.Series([100.0, 110.0, 110.0], index=d)   # +10%, 0%
    b = pd.Series([50.0, 50.0, 55.0], index=d)      # 0%, +10%
    out = core_close({"a": a, "b": b}, {"a": 0.7, "b": 0.3})
    assert list(out) == pytest.approx([107.0, 107.0 * 1.03])
    assert core_close({"a": a}, {"a": 0.7, "b": 0.3}) is None


def test_fetch_core_degrades_without_raising():
    assert main_mod._fetch_core(None) == (None, [])

    class _Down:
        def get_chart_data(self, **_):
            raise RuntimeError("down")

    core, notes = main_mod._fetch_core(_Down())
    assert core is None and "satellite optimised on its own" in notes[0]


# --- withholding tax (ISK) --------------------------------------------------

from analyzer.frontier import withholding_drag


def test_withholding_drag_is_yield_times_loss_and_zero_at_home():
    d = withholding_drag({"SE": 0.05, "US": 0.03, "DK": 0.03, "X": None},
                         {"SE": "SE", "US": "US", "DK": "DK", "X": "DK"},
                         {"DK": 0.12})
    assert d["SE"] == 0 and d["US"] == 0
    assert d["DK"] == pytest.approx(0.0036)
    assert d["X"] == 0  # no yield known -> no drag, never NaN


def test_drag_moves_max_sharpe_but_never_min_variance():
    hist, ccy = _universe(n=4, vols=[0.015] * 4, seed=3)
    blend = pd.Series({"S0": 0.9, "S1": 0.9, "S2": 0.5, "S3": 0.5,
                       "X1": 0.1, "X2": 0.2, "X3": 0.3})
    base = build_portfolios(hist, ccy, blend, list(hist), None, cap=1.0)
    taxed = build_portfolios(hist, ccy, blend, list(hist), None, cap=1.0,
                             tax_drag=pd.Series({"S0": 0.02}))
    assert taxed["maxsharpe"]["weights"]["S0"] < base["maxsharpe"]["weights"]["S0"]
    assert taxed["minvar"]["weights"].to_dict() == pytest.approx(
        base["minvar"]["weights"].to_dict())


# --- per-sector cap -------------------------------------------------------

from analyzer.frontier import sector_group


def test_sector_group_reads_middle_level_from_list_or_json():
    lvls = [{"sectorId": "99", "sectorName": "Bioteknik"},
            {"sectorId": "17", "sectorName": "Bioteknik & Läkemedel"},
            {"sectorId": "15", "sectorName": "Hälsovård"}]
    assert sector_group(lvls) == "Bioteknik & Läkemedel"
    import json
    assert sector_group(json.dumps(lvls)) == "Bioteknik & Läkemedel"
    assert sector_group([{"sectorName": "Bank"}]) == "Bank"
    assert sector_group(None) is None
    assert sector_group("not json") is None
    assert sector_group([]) is None


def _calm_pharma():
    # The four calmest names share a group: unconstrained, min-variance piles in.
    # 8 others at the 15% cap hold 120%, so 30% on pharma stays feasible.
    hist, ccy = _universe(n=12, vols=[0.004] * 4 + [0.02] * 8, seed=11)
    sectors = {f"S{i}": ("Pharma" if i < 4 else f"Other{i}") for i in range(12)}
    blend = pd.Series({f"S{i}": 0.9 - 0.05 * i for i in range(12)}
                      | {"X1": 0.1, "X2": 0.2})
    return hist, ccy, sectors, blend


def test_sector_cap_binds_in_both_portfolios():
    hist, ccy, sectors, blend = _calm_pharma()
    pharma = [f"S{i}" for i in range(4)]
    free = build_portfolios(hist, ccy, blend, list(hist), None, cap=0.15)
    assert free["minvar"]["weights"][pharma].sum() > 0.30 + 1e-3
    capped = build_portfolios(hist, ccy, blend, list(hist), None, cap=0.15,
                              sectors=sectors, sector_cap=0.30)
    for key in ("minvar", "maxsharpe"):
        w = capped[key]["weights"]
        assert w[pharma].sum() <= 0.30 + 1e-6
        assert w.sum() == pytest.approx(1.0)
        assert (w <= 0.15 + 1e-9).all()


def test_no_sector_data_leaves_weights_unchanged():
    hist, ccy, _, blend = _calm_pharma()
    base = build_portfolios(hist, ccy, blend, list(hist), None, cap=0.15)
    for sectors in (None, {}, {k: None for k in hist}):
        p = build_portfolios(hist, ccy, blend, list(hist), None, cap=0.15,
                             sectors=sectors)
        for key in ("minvar", "maxsharpe"):
            assert p[key]["weights"].to_dict() == pytest.approx(
                base[key]["weights"].to_dict())


def test_sector_cap_relaxes_when_too_few_groups():
    hist, ccy, _, blend = _calm_pharma()
    two = {f"S{i}": ("A" if i < 4 else "B") for i in range(12)}
    p = build_portfolios(hist, ccy, blend, list(hist), None, cap=1.0,
                         sectors=two, sector_cap=0.30)
    w = p["minvar"]["weights"]
    assert w.sum() == pytest.approx(1.0)
    assert w[[f"S{i}" for i in range(4)]].sum() <= 0.50 + 1e-6


def test_live_loop_can_reach_frontier():
    # The live loop calls frontier.sector_group per ticker; it once only
    # existed as a local import inside _portfolio_section and crashed the Pi.
    assert main_mod.frontier is frontier
