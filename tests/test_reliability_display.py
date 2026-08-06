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


def test_removed_stock_shows_full_metrics_not_just_name(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(main, "project_root", str(tmp_path))

    manager = SummaryManager()
    manager.summary = pd.DataFrame(
        {
            "points": [3.0, -1.0],
            # Beta is scored but ranks below the sleeve gate -> removed,
            # while still being present in `combined` with real metrics.
            "quality_pct": [0.8, 0.2],
            "value_pct": [0.7, 0.1],
            "combined_score": [0.56, 0.02],
        },
        index=["Alpha AB 111", "Beta AB 222"],
    )
    manager.summary_investment = pd.DataFrame()

    # Beta is already on the watchlist (existing_ids) but its low sleeve
    # scores mean it won't qualify this run -> should be removed with its
    # real q/v/combined shown, not just its bare name.
    avanza = _FakeAvanza(watchlist_orderbook_ids=["222"])
    main._update_watchlist(avanza, manager, top_n=10, target_name="Bör köpa")

    out = capsys.readouterr().out
    assert "Removed 1 stock(s)" in out
    assert "Beta AB 222" in out
    assert "q=0.20, v=0.10, combined=0.02" in out
    assert "not scored this run" not in out


def test_removed_stock_not_in_this_runs_universe_falls_back_gracefully(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(main, "project_root", str(tmp_path))

    manager = SummaryManager()
    # Only Alpha is scored this run -- Gamma is on the watchlist (e.g. from
    # a broader --preset/--watchlists scope on a prior run) but wasn't part
    # of this run's universe at all, so it has no row anywhere in `combined`.
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

    avanza = _FakeAvanza(watchlist_orderbook_ids=["999"])
    main._update_watchlist(avanza, manager, top_n=10, target_name="Bör köpa")

    out = capsys.readouterr().out
    assert "Unknown (999)  (not scored this run)" in out
