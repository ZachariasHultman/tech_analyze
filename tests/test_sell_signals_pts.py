"""_compute_sell_signals: single, reliability-free trigger -- flags a stock
when pts < 0, full stop. The old reliability-gated conditions ("only trust
a negative score when reliability is established", "flag when reliability
is inverse") were removed along with the reliability mechanism itself.
"""
import pandas as pd

import analyzer.main as main
from analyzer.summary_manager import SummaryManager


class _FakeAvanza:
    def __init__(self, watchlist_orderbook_ids):
        self._wl_ids = watchlist_orderbook_ids

    def get_watchlists(self):
        return [{"name": "Äger", "orderbookIds": self._wl_ids}]

    def get_stock_info(self, ticker_id):
        return {"name": ticker_id}


def _manager(points_by_company):
    m = SummaryManager()
    m.summary = pd.DataFrame(
        {"points": list(points_by_company.values())},
        index=list(points_by_company.keys()),
    )
    return m


def test_negative_pts_triggers_sell_signal():
    avanza = _FakeAvanza(watchlist_orderbook_ids=["111"])
    manager = _manager({"Alpha AB 111": -2.5})

    signals = main._compute_sell_signals(avanza, manager, "Äger")

    assert len(signals) == 1
    assert signals[0]["name"] == "Alpha AB 111"
    assert signals[0]["pts"] == -2.5
    assert signals[0]["reasons"] == "fundamentals have deteriorated"


def test_nonnegative_pts_does_not_trigger():
    avanza = _FakeAvanza(watchlist_orderbook_ids=["111"])
    manager = _manager({"Alpha AB 111": 0.0})

    signals = main._compute_sell_signals(avanza, manager, "Äger")

    assert signals == []


def test_sorted_worst_first_by_pts():
    avanza = _FakeAvanza(watchlist_orderbook_ids=["111", "222"])
    manager = _manager({"Alpha AB 111": -1.0, "Beta AB 222": -5.0})

    signals = main._compute_sell_signals(avanza, manager, "Äger")

    assert [s["name"] for s in signals] == ["Beta AB 222", "Alpha AB 111"]


def test_missing_points_column_skips_instead_of_raising(capsys):
    """`points` is the only key the deterioration check has, but its absence is
    a skipped check, not a crash.

    `_compute_sell_signals` concats through `dropna(axis=1, how="all")`, so an
    all-NaN `points` column is dropped before the direct index reads it -- the
    same unguarded-index shape that made `_update_watchlist` push an empty
    watchlist, here landing as a KeyError that would take down the whole
    weekly run mid-report.
    """
    avanza = _FakeAvanza(watchlist_orderbook_ids=["111"])
    manager = SummaryManager()
    manager.summary = pd.DataFrame(
        {"quality_pct": [0.9], "value_pct": [0.9]},
        index=["Alpha AB 111"],
    )
    assert main._compute_sell_signals(avanza, manager, "Äger") == []
    assert "skipping sell check" in capsys.readouterr().out
