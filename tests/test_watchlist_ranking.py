"""_update_watchlist ranks by pts, with no reliability involved.

Ranking moved from combined_score to pts: over 7 fiscal years pts carried the
only significant out-of-sample signal (IC +0.041, t=+3.29, p=0.017, positive
in 6 of 7 years), while combined_score reached only p=0.18 despite a higher
point estimate -- multiplying two noisy percentiles amplifies the noise. The
sleeve gate still filters on quality_pct/value_pct; only the ordering within
the qualifying set changed.

Reliability (the old company_reliability.csv tilt and its short-lived
panel-based replacement) was removed entirely: neither could produce a
differentiated signal given the live OHLC depth ceiling, so ranking reads no
reliability file or state at all.
"""
import pandas as pd

import analyzer.main as main
from analyzer.summary_manager import SummaryManager


class _FakeAvanza:
    def __init__(self, watchlist_orderbook_ids):
        self._wl_ids = watchlist_orderbook_ids

    def get_watchlists(self):
        return [{"name": "Bör köpa", "watchListId": "1", "orderbookIds": self._wl_ids}]

    def add_to_watchlist(self, orderbook_id, watchlist_id):
        pass

    def remove_from_watchlist(self, orderbook_id, watchlist_id):
        pass


def test_ranks_by_points_not_combined_score(capsys):
    manager = SummaryManager()
    manager.summary = pd.DataFrame(
        {
            "points": [1.0, 3.0],
            "quality_pct": [0.8, 0.9],
            "value_pct": [0.7, 0.7],
            # Beta has the lower combined_score despite higher points --
            # ranking must follow pts, so Beta comes first.
            "combined_score": [0.56, 0.50],
        },
        index=["Alpha AB 111", "Beta AB 222"],
    )
    manager.summary_investment = pd.DataFrame()

    avanza = _FakeAvanza(watchlist_orderbook_ids=[])
    result = main._update_watchlist(avanza, manager, top_n=10, target_name="Bör köpa")

    added_names = [r[0] for r in result["added"]]
    assert added_names == ["Beta AB 222", "Alpha AB 111"]

    out = capsys.readouterr().out
    assert "r=" not in out
    assert "reliability" not in out.lower()
    # combined is still reported, it just no longer decides the order
    assert "(pts=+1.00, q=0.80, v=0.70, combined=0.56)" in out


def test_removed_stock_still_shows_full_metrics(capsys):
    manager = SummaryManager()
    manager.summary = pd.DataFrame(
        {
            "points": [3.0, -1.0],
            "quality_pct": [0.8, 0.2],
            "value_pct": [0.7, 0.1],
            "combined_score": [0.56, 0.02],
        },
        index=["Alpha AB 111", "Beta AB 222"],
    )
    manager.summary_investment = pd.DataFrame()

    avanza = _FakeAvanza(watchlist_orderbook_ids=["222"])
    main._update_watchlist(avanza, manager, top_n=10, target_name="Bör köpa")

    out = capsys.readouterr().out
    assert "Removed 1 stock(s)" in out
    assert "q=0.20, v=0.10, combined=0.02" in out


def test_push_top_defaults_to_a_quintile_not_a_handful():
    """Breadth is what harvests a weak ranking signal: IR ~ IC x sqrt(breadth).

    At IC +0.041 a 10-stock pick discards most of the measured edge, and the
    top-10's year-to-year excess-return sd was 9.98% against 4.48% at N=25 --
    noise large enough that no edge of a plausible size is ever observable
    there. 25 is also ~the top quintile of the ~127-stock universe, which is
    the bucket the backtest actually validates.
    """
    import analyzer.main as main_mod

    # The argparse parser is built inside main(); assert on the function
    # signature that consumes the value instead.
    import inspect
    sig = inspect.signature(main_mod._update_watchlist)
    assert sig.parameters["top_n"].default == 25


def test_ranks_by_blend_when_present(capsys):
    """With blend_score present it, not pts, decides the order.

    Measured on the panel: the top-25 basket ranked on pts returned +2.77%/yr
    excess (p=0.171, positive 5 of 7 fiscal years); ranked on the 50/50
    momentum blend, +5.90%/yr (p=0.053, 6 of 7).
    """
    manager = SummaryManager()
    manager.summary = pd.DataFrame(
        {
            # Alpha wins on pts, Beta wins on the blend -- Beta must come first.
            "points": [9.0, 2.0],
            "quality_pct": [0.8, 0.6],
            "value_pct": [0.7, 0.6],
            "combined_score": [0.56, 0.36],
            "momentum_pct": [0.10, 0.95],
            "fundamental_pct": [0.95, 0.40],
            "blend_score": [0.525, 0.675],
        },
        index=["Alpha AB 111", "Beta AB 222"],
    )
    manager.summary_investment = pd.DataFrame()

    avanza = _FakeAvanza(watchlist_orderbook_ids=[])
    result = main._update_watchlist(avanza, manager, top_n=10, target_name="Bör köpa")

    assert [r[0] for r in result["added"]] == ["Beta AB 222", "Alpha AB 111"]
    out = capsys.readouterr().out
    assert "rank=0.68" in out and "mom=0.95" in out
    # pts is still reported -- it just no longer decides the order
    assert "pts=+2.00" in out


def test_falls_back_to_pts_when_no_blend_column():
    """A summary built before the blend existed (or by a caller that doesn't
    produce it) must still rank rather than collapse to an arbitrary order."""
    manager = SummaryManager()
    manager.summary = pd.DataFrame(
        {
            "points": [1.0, 3.0],
            "quality_pct": [0.8, 0.9],
            "value_pct": [0.7, 0.7],
            "combined_score": [0.56, 0.63],
        },
        index=["Alpha AB 111", "Beta AB 222"],
    )
    manager.summary_investment = pd.DataFrame()

    avanza = _FakeAvanza(watchlist_orderbook_ids=[])
    result = main._update_watchlist(avanza, manager, top_n=10, target_name="Bör köpa")
    assert [r[0] for r in result["added"]] == ["Beta AB 222", "Alpha AB 111"]
