"""Every company must be ranked against the whole universe, not its template.

calculate_score used to score `summary` and `summary_investment` in two
separate passes, so percentiles were computed *within* each pool and then
compared across them in _update_watchlist. Being best-of-3 investment
companies beat being best-of-26 regular ones:

    Alfa Laval    q=1.00 v=0.73 combined=0.73    (best of 26)
    Investor B    q=1.00 v=1.00 combined=1.00    (best of 3)

Seen live: Investor B (pts=+3.81, combined=0.51) took the last watchlist slot
from Swedbank A (pts=+4.92, combined=0.50) -- 1.1 fewer points, in on a 0.01
percentile edge drawn from a pool of a handful.

The per-metric scores feeding `points` had the same problem, plus a second
regime: below 5 companies calculate_score skips cross-sectional ranking
entirely, so a small investment pool was scored on absolute thresholds while
everyone else was peer-ranked.
"""

import numpy as np
import pandas as pd
import pytest

from analyzer.data_processing import calculate_score
from analyzer.summary_manager import SummaryManager


def _manager(n_regular=26, n_investment=3):
    """Regular companies deliberately span a wide metric range; investment
    companies sit in the middle of it, so a correct pooled ranking must place
    them mid-pack rather than at the top of their own little pool."""
    sm = SummaryManager()
    sm._weight_overrides = {"roe_pe ratio status": 1.0,
                            "net debt - ebitda status": 1.0,
                            "dividend yield status": 1.0}
    reg_sector = [{"sectorId": "1", "sectorName": "Industri"}]
    inv_sector = [{"sectorId": "51", "sectorName": "Investmentbolag"}]

    for i in range(n_regular):
        name = f"Reg{i:02d}"
        sm._initialize_template(name, reg_sector)
        sm._update(name, reg_sector, "roe_pe ratio status", 0.1 + i * 0.1)
        sm._update(name, reg_sector, "net debt - ebitda status", 5.0 - i * 0.15)
        sm._update(name, reg_sector, "dividend yield status", 0.001 * i)
    for i in range(n_investment):
        name = f"Inv{i:02d}"
        sm._initialize_template(name, inv_sector)
        # mid-range values -- unremarkable against the full universe
        sm._update(name, inv_sector, "roe_pe ratio status", 1.0 + i * 0.05)
        sm._update(name, inv_sector, "net debt - ebitda status", 2.5 - i * 0.05)
        sm._update(name, inv_sector, "dividend yield status", 0.013 + i * 0.001)
    return sm


def _num(df, name, col):
    v = df.loc[name, col]
    return float(v[0] if isinstance(v, (list, tuple)) else v)


def test_best_of_a_small_pool_does_not_get_a_perfect_percentile():
    sm = _manager()
    calculate_score(sm)
    inv = sm.summary_investment
    for name in inv.index:
        assert _num(inv, name, "quality_pct") < 1.0 or _num(inv, name, "value_pct") < 1.0, (
            f"{name} scored a perfect percentile -- still pooled separately"
        )


def test_mid_range_investment_company_ranks_mid_pack():
    sm = _manager()
    calculate_score(sm)
    both = pd.concat([sm.summary, sm.summary_investment])
    combined = both["combined_score"].map(
        lambda v: float(v[0] if isinstance(v, (list, tuple)) else v))
    order = combined.sort_values(ascending=False).index.tolist()
    top5 = order[:5]
    assert not any(n.startswith("Inv") for n in top5), (
        f"a mid-range investment company reached the top 5: {top5}"
    )


def test_percentile_denominator_is_the_whole_universe():
    """The direct test: a percentile over n companies is a multiple of 1/n.

    Counting distinct values would not prove this -- the +/-1 step function
    ties many companies together regardless of pool size. Checking that every
    percentile is a whole number of 1/29 (and not of 1/3) pins the denominator.
    """
    sm = _manager(n_regular=26, n_investment=3)
    calculate_score(sm)
    both = pd.concat([sm.summary, sm.summary_investment])
    n = len(both)
    assert n == 29
    for col in ("quality_pct", "value_pct"):
        vals = both[col].map(
            lambda v: float(v[0] if isinstance(v, (list, tuple)) else v)).dropna()
        assert len(vals) == n
        # rank(pct=True) averages ties, so a tied pair lands on a half step.
        scaled = vals * n * 2
        assert np.allclose(scaled, np.round(scaled)), (
            f"{col} is not quantised to 1/{n} -- ranked against a sub-pool"
        )
    # And an investment company's percentile is not pinned to 1/3 steps.
    inv_q = sm.summary_investment["quality_pct"].map(
        lambda v: float(v[0] if isinstance(v, (list, tuple)) else v))
    assert not all(abs(v * 3 - round(v * 3)) < 1e-9 for v in inv_q)


def test_both_frames_survive_and_keep_their_own_columns():
    sm = _manager()
    calculate_score(sm)
    assert len(sm.summary) == 26
    assert len(sm.summary_investment) == 3
    # the regular table must not sprout investment-only columns
    assert "nav discount status" not in sm.summary.columns
    for col in ("points", "quality_pct", "value_pct", "combined_score"):
        assert col in sm.summary.columns
        assert col in sm.summary_investment.columns


def test_investment_only_universe_still_scores():
    sm = _manager(n_regular=0, n_investment=6)
    calculate_score(sm)
    assert len(sm.summary_investment) == 6
    assert sm.summary.empty


def test_regular_only_universe_is_unchanged():
    sm = _manager(n_regular=12, n_investment=0)
    calculate_score(sm)
    assert len(sm.summary) == 12
    assert sm.summary_investment.empty
    assert sm.summary["combined_score"].notna().all()
