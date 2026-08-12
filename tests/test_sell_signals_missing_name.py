"""Regression test: _compute_sell_signals' "not analyzed" warning should
print the stock's name alongside its orderbookId, not just the bare ID --
previously printed only e.g. "[WARN] 1041067 not analyzed", which isn't
identifiable without a manual lookup. Falls back to the bare ID if the
Avanza lookup itself fails (e.g. delisted/API error), so one bad lookup
can't crash the whole warning loop.
"""
import pandas as pd

import analyzer.main as main
from analyzer.summary_manager import SummaryManager


class _FakeAvanza:
    def __init__(self, watchlist_orderbook_ids, names_by_id, raise_for=()):
        self._wl_ids = watchlist_orderbook_ids
        self._names = names_by_id
        self._raise_for = set(raise_for)

    def get_watchlists(self):
        return [{"name": "Äger", "orderbookIds": self._wl_ids}]

    def get_stock_info(self, ticker_id):
        if ticker_id in self._raise_for:
            raise RuntimeError("boom")
        return {"name": self._names.get(ticker_id, ticker_id)}


def _manager_with_no_scored_stocks():
    m = SummaryManager()
    m.summary = pd.DataFrame({"points": []})
    m.summary_investment = pd.DataFrame()
    return m


def test_missing_stock_warning_includes_name(monkeypatch, capsys):
    avanza = _FakeAvanza(
        watchlist_orderbook_ids=["1041067"],
        names_by_id={"1041067": "Some Company AB"},
    )
    manager = _manager_with_no_scored_stocks()
    manager.summary = pd.DataFrame({"points": [1.0]}, index=["Other Company 999"])

    main._compute_sell_signals(avanza, manager, "Äger")

    out = capsys.readouterr().out
    assert "Some Company AB" in out
    assert "1041067" in out


def test_missing_stock_warning_falls_back_to_id_on_lookup_failure(monkeypatch, capsys):
    avanza = _FakeAvanza(
        watchlist_orderbook_ids=["693833"],
        names_by_id={},
        raise_for={"693833"},
    )
    manager = _manager_with_no_scored_stocks()
    manager.summary = pd.DataFrame({"points": [1.0]}, index=["Other Company 999"])

    # Must not raise even though the name lookup fails.
    main._compute_sell_signals(avanza, manager, "Äger")

    out = capsys.readouterr().out
    assert "693833" in out
