"""_update_watchlist ranks purely by combined_score -- reliability (old
company_reliability.csv-based tilt, and its short-lived panel-based
replacement) was removed entirely: neither mechanism could produce a
differentiated signal given the live OHLC depth ceiling, so ranking no
longer reads any reliability file/state at all.
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


def test_ranks_purely_by_combined_score_no_reliability_involved(capsys):
    manager = SummaryManager()
    manager.summary = pd.DataFrame(
        {
            "points": [1.0, 3.0],
            "quality_pct": [0.8, 0.9],
            "value_pct": [0.7, 0.7],
            # Beta has the lower combined_score despite higher points --
            # ranking must follow combined_score, not points.
            "combined_score": [0.56, 0.50],
        },
        index=["Alpha AB 111", "Beta AB 222"],
    )
    manager.summary_investment = pd.DataFrame()

    avanza = _FakeAvanza(watchlist_orderbook_ids=[])
    result = main._update_watchlist(avanza, manager, top_n=10, target_name="Bör köpa")

    added_names = [r[0] for r in result["added"]]
    assert added_names == ["Alpha AB 111", "Beta AB 222"]

    out = capsys.readouterr().out
    assert "r=" not in out
    assert "reliability" not in out.lower()
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
