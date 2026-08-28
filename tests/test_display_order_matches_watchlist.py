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
    `blend_score` they read Alpha, Beta, Gamma; ordered by `combined_score`
    they read Beta, Gamma, Alpha. Any sort that mixes them is therefore
    unambiguously visible, at every rung of the fallback chain.
    """
    return pd.DataFrame(
        {
            "points": [1.0, 5.0, 9.0],
            "blend_score": [0.9, 0.5, 0.1],
            "combined_score": [0.1, 0.9, 0.5],
            "quality_pct": [0.9, 0.9, 0.9],
            "value_pct": [0.9, 0.9, 0.9],
        },
        index=["Alpha AB 111", "Beta AB 222", "Gamma AB 333"],
    )


# The three ranking columns each impose a different order, so any sort that
# reaches the wrong rung of the fallback chain is unambiguously visible.
_ORDER_BY_BLEND = ["Alpha AB 111", "Beta AB 222", "Gamma AB 333"]
_ORDER_BY_POINTS = ["Gamma AB 333", "Beta AB 222", "Alpha AB 111"]
_ORDER_BY_COMBINED = ["Beta AB 222", "Gamma AB 333", "Alpha AB 111"]


def _display_order(df, monkeypatch, tmp_path):
    """Run _display for its side effect and read back the exported order."""
    monkeypatch.chdir(tmp_path)
    sm = SummaryManager()
    sm.summary = df
    sm._display(save_df=True)
    return list(pd.read_csv(tmp_path / "summary.csv", index_col=0).index)


def test_display_orders_by_blend_score_when_present(monkeypatch, tmp_path):
    assert _display_order(_frame(), monkeypatch, tmp_path) == ["Alpha AB 111", "Beta AB 222", "Gamma AB 333"]


def test_display_falls_back_to_points_without_blend(monkeypatch, tmp_path):
    df = _frame().drop(columns=["blend_score"])
    assert _display_order(df, monkeypatch, tmp_path) == ["Gamma AB 333", "Beta AB 222", "Alpha AB 111"]


def test_display_falls_back_to_combined_score_without_points(monkeypatch, tmp_path):
    df = _frame().drop(columns=["blend_score", "points"])
    assert _display_order(df, monkeypatch, tmp_path) == _ORDER_BY_COMBINED


class _FakeAvanza:
    def get_watchlists(self):
        return [{"id": "wl", "name": "Bör köpa", "orderbooks": []}]

    def add_to_watchlist(self, orderbook_id, watchlist_id):
        pass

    def remove_from_watchlist(self, orderbook_id, watchlist_id):
        pass


def _pushed_order(df):
    """Run _update_watchlist and read back the order it pushed."""
    manager = SummaryManager()
    manager.summary = df
    result = _update_watchlist(_FakeAvanza(), manager, top_n=3,
                               target_name="Bör köpa")
    assert result is not None, "_update_watchlist bailed out instead of ranking"
    pushed = [r[0] for r in result["added"]]
    assert pushed, "nothing was pushed — fixture no longer clears the sleeve gate"
    return pushed


def test_display_and_watchlist_agree_on_the_top_slot(monkeypatch, tmp_path):
    """The end-to-end property: whatever _update_watchlist ranks first is what
    the printed table shows first. This is the invariant the two sort keys
    exist to hold; the tests above only pin each side of it separately."""
    pushed = _pushed_order(_frame())
    order = _display_order(_frame(), monkeypatch, tmp_path)
    assert order[0] == pushed[0]


@pytest.mark.parametrize(
    "dropped, expected",
    [
        ([], _ORDER_BY_BLEND),
        (["blend_score"], _ORDER_BY_POINTS),
        (["blend_score", "points"], _ORDER_BY_COMBINED),
    ],
    ids=["blend_score", "points", "combined_score"],
)
def test_both_sides_walk_the_same_fallback_chain(dropped, expected, monkeypatch, tmp_path):
    """Every rung of blend_score -> points -> combined_score, both sides.

    `_display` walked all three rungs while `_update_watchlist` only walked
    two: its intake filter dropped any summary without a `points` column
    outright, so a combined_score-only cross-section printed a full table and
    pushed an empty watchlist -- the two-ranking-regimes defect again, in the
    one shape the previous tests could not see, because they only exercised
    the third rung on the display side.
    """
    df = _frame().drop(columns=dropped)
    assert _display_order(df, monkeypatch, tmp_path) == expected
    assert _pushed_order(df.copy()) == expected


def test_watchlist_bails_out_when_no_ranking_column_survives():
    """No blend_score, no points, no combined_score -- there is nothing to rank
    on, and pushing an arbitrary order would be worse than pushing nothing."""
    df = _frame().drop(columns=["blend_score", "points", "combined_score"])
    manager = SummaryManager()
    manager.summary = df
    assert _update_watchlist(_FakeAvanza(), manager, top_n=3,
                             target_name="Bör köpa") is None
