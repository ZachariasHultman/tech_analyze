"""Regression tests for the q/v/combined/r display fix:

1. _fmt_reliability formats a shrunk-reliability + sample-size pair
   consistently ("+0.60 (n=5)"), used everywhere "r" is shown.
2. _update_watchlist's row tuples must carry the SHRUNK reliability figure
   (spearman_shrunk), not raw spearman -- previously the printed "r" was
   raw spearman while the actual ranking (_combined) already used the
   shrunk value, so the displayed number didn't match what drove the list.
3. LEGEND_LINES gets printed once in the watchlist report.
"""
import math

import pandas as pd

import analyzer.main as main
from analyzer.summary_manager import SummaryManager


def test_fmt_reliability_formats_value_and_n():
    assert main._fmt_reliability(0.6, 5) == "+0.60 (n=5)"
    assert main._fmt_reliability(-0.4, 5) == "-0.40 (n=5)"


def test_fmt_reliability_na_value_returns_na():
    assert main._fmt_reliability(float("nan"), 5) == "N/A"
    assert main._fmt_reliability(None, 5) == "N/A"


def test_fmt_reliability_missing_n_shows_question_mark():
    assert main._fmt_reliability(0.5, float("nan")) == "+0.50 (n=?)"


class _FakeAvanza:
    def __init__(self, watchlist_orderbook_ids):
        self._wl_ids = watchlist_orderbook_ids

    def get_watchlists(self):
        return [{"name": "Bör köpa", "watchListId": "1", "orderbookIds": self._wl_ids}]

    def add_to_watchlist(self, orderbook_id, watchlist_id):
        pass

    def remove_from_watchlist(self, orderbook_id, watchlist_id):
        pass


def test_update_watchlist_uses_shrunk_reliability_not_raw(monkeypatch, capsys, tmp_path):
    # raw spearman and spearman_shrunk deliberately differ so the test fails
    # if the old (raw) field is read instead of the shrunk one.
    rel_csv = tmp_path / "company_reliability.csv"
    rel_csv.write_text(
        "company,spearman,spearman_shrunk,n_windows,reliable\n"
        "Alpha AB 111,0.90,0.30,5,True\n"
    )
    monkeypatch.setattr(main, "project_root", str(tmp_path))

    manager = SummaryManager()
    manager.summary = pd.DataFrame(
        {
            "points": [3.0],
            "quality_pct": [0.8],
            "value_pct": [0.7],
            "combined_score": [0.56],
        },
        index=["Alpha AB 111"],
    )
    manager.summary_investment = pd.DataFrame()

    avanza = _FakeAvanza(watchlist_orderbook_ids=[])
    main._update_watchlist(avanza, manager, top_n=10, target_name="Bör köpa")

    out = capsys.readouterr().out
    assert "r=+0.30 (n=5)" in out
    assert "r=+0.90" not in out


def test_update_watchlist_prints_legend(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(main, "project_root", str(tmp_path))

    manager = SummaryManager()
    manager.summary = pd.DataFrame(
        {
            "points": [3.0],
            "quality_pct": [0.8],
            "value_pct": [0.7],
            "combined_score": [0.56],
        },
        index=["Alpha AB 111"],
    )
    manager.summary_investment = pd.DataFrame()

    avanza = _FakeAvanza(watchlist_orderbook_ids=[])
    main._update_watchlist(avanza, manager, top_n=10, target_name="Bör köpa")

    out = capsys.readouterr().out
    for line in main.LEGEND_LINES:
        assert line in out
