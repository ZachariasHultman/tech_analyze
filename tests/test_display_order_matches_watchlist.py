"""The printed/exported table must be ordered by the same key the watchlist is.

`_update_watchlist` sorts on `blend_score` (falling back to `points`), but
`SummaryManager._display`'s own sort key was left on `points` when the ranking
key moved. The two then disagree, which is precisely the failure its docstring
says it exists to prevent: the top of the printed table stops being the set
that gets pushed.

These tests pin the ordering contract rather than the implementation, so the
fallback chain (blend_score -> points -> combined_score) is exercised end to
end and a future ranking-key change has to update both sides or fail here.
"""
import pandas as pd
import pytest

from analyzer.main import _update_watchlist
from analyzer.summary_manager import SummaryManager


def _frame():
    """points and blend_score deliberately rank the rows in opposite orders.

    Ordered by `points` the rows read Gamma, Beta, Alpha; ordered by
    `blend_score` they read Alpha, Beta, Gamma. Any sort that mixes the two is therefore unambiguously visible.
    """
    return pd.DataFrame(
        {
            "points": [1.0, 5.0, 9.0],
            "blend_score": [0.9, 0.5, 0.1],
            "combined_score": [0.4, 0.4, 0.4],
            "quality_pct": [0.9, 0.9, 0.9],
            "value_pct": [0.9, 0.9, 0.9],
        },
        index=["Alpha AB 111", "Beta AB 222", "Gamma AB 333"],
    )


def _display_order(df, monkeypatch, tmp_path):
    """Run _display for its side effect and read back the exported order."""
    monkeypatch.chdir(tmp_path)
    sm = SummaryManager()
    sm.summary = df
    sm.summary_investment = pd.DataFrame()
    sm._display(save_df=True)
    return list(pd.read_csv(tmp_path / "summary.csv", index_col=0).index)


def test_display_orders_by_blend_score_when_present(monkeypatch, tmp_path):
    assert _display_order(_frame(), monkeypatch, tmp_path) == ["Alpha AB 111", "Beta AB 222", "Gamma AB 333"]


def test_display_falls_back_to_points_without_blend(monkeypatch, tmp_path):
    df = _frame().drop(columns=["blend_score"])
    assert _display_order(df, monkeypatch, tmp_path) == ["Gamma AB 333", "Beta AB 222", "Alpha AB 111"]


def test_display_falls_back_to_combined_score_without_points(monkeypatch, tmp_path):
    """combined_score is flat here, so this only asserts it does not raise and
    still emits every row -- the ordering among ties is not a contract."""
    df = _frame().drop(columns=["blend_score", "points"])
    assert sorted(_display_order(df, monkeypatch, tmp_path)) == ["Alpha AB 111", "Beta AB 222", "Gamma AB 333"]


def test_display_and_watchlist_agree_on_the_top_slot(monkeypatch, tmp_path):
    """The end-to-end property: whatever _update_watchlist ranks first is what
    the printed table shows first. This is the invariant the two sort keys
    exist to hold; the tests above only pin each side of it separately."""

    class _FakeAvanza:
        def get_watchlists(self):
            return [{"id": "wl", "name": "Bör köpa", "orderbooks": []}]

        def add_to_watchlist(self, orderbook_id, watchlist_id):
            pass

        def remove_from_watchlist(self, orderbook_id, watchlist_id):
            pass

    manager = SummaryManager()
    manager.summary = _frame()
    manager.summary_investment = pd.DataFrame()

    result = _update_watchlist(_FakeAvanza(), manager, top_n=3,
                               target_name="Bör köpa")
    pushed = [r[0] for r in result["added"]]
    assert pushed, "nothing was pushed — fixture no longer clears the sleeve gate"

    order = _display_order(_frame(), monkeypatch, tmp_path)
    assert order[0] == pushed[0]
