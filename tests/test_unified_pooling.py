"""Every company is ranked against the whole universe, in one cross-section.

History, because the failure was subtle and the fix is easy to undo.
`calculate_score` used to score `summary` and `summary_investment` in two
separate passes, so percentiles were computed *within* each pool and then
compared across them in `_update_watchlist`. Being best-of-3 investment
companies beat being best-of-26 regular ones:

    Alfa Laval    q=1.00 v=0.73 combined=0.73    (best of 26)
    Investor B    q=1.00 v=1.00 combined=1.00    (best of 3)

Seen live: Investor B (pts=+3.81, combined=0.51) took the last watchlist slot
from Swedbank A (pts=+4.92, combined=0.50) -- 1.1 fewer points, in on a 0.01
percentile edge drawn from a pool of a handful. The per-metric scores feeding
`points` had the same problem plus a second regime: below 5 companies
`calculate_score` skips cross-sectional ranking entirely, so a small pool was
scored on absolute thresholds while everyone else was peer-ranked.

Pooling fixed that. The second pool has since been removed outright -- see
tests/test_investment_companies_excluded.py -- so the specific bug can no
longer be constructed. What these tests still pin is the invariant that
outlived it: **the percentile denominator is the whole scored universe**, so no
subgroup can be graded on a curve of its own.
"""

import numpy as np
import pandas as pd
import pytest

from analyzer.data_processing import calculate_score
from analyzer.summary_manager import SummaryManager

_SECTOR = [{"sectorId": "1", "sectorName": "Industri"}]


def _manager(n=29):
    """A wide-spanning universe with three companies parked mid-range.

    The mid-range trio stands in for what used to be the separate pool: if
    percentiles were ever computed over a subgroup again, they would come out
    near 1.0 instead of mid-pack.
    """
    sm = SummaryManager()
    sm._weight_overrides = {"roe_pe ratio status": 1.0,
                            "net debt - ebitda status": 1.0,
                            "price momentum status": 1.0}
    for i in range(n - 3):
        name = f"Reg{i:02d}"
        sm._initialize_template(name, _SECTOR)
        sm._update(name, _SECTOR, "roe_pe ratio status", 0.1 + i * 0.1)
        sm._update(name, _SECTOR, "net debt - ebitda status", 5.0 - i * 0.15)
        sm._update(name, _SECTOR, "price momentum status", -0.2 + 0.02 * i)
    for i in range(3):
        name = f"Mid{i:02d}"
        sm._initialize_template(name, _SECTOR)
        sm._update(name, _SECTOR, "roe_pe ratio status", 1.0 + i * 0.05)
        sm._update(name, _SECTOR, "net debt - ebitda status", 2.5 - i * 0.05)
        sm._update(name, _SECTOR, "price momentum status", 0.05 + i * 0.01)
    return sm


def _num(df, name, col):
    v = df.loc[name, col]
    return float(v[0] if isinstance(v, (list, tuple)) else v)


def test_a_mid_range_company_does_not_get_a_perfect_percentile():
    sm = _manager()
    calculate_score(sm)
    for name in [i for i in sm.summary.index if str(i).startswith("Mid")]:
        assert (_num(sm.summary, name, "quality_pct") < 1.0
                or _num(sm.summary, name, "value_pct") < 1.0), (
            f"{name} scored a perfect percentile -- graded on its own curve"
        )


def test_mid_range_company_ranks_mid_pack():
    sm = _manager()
    calculate_score(sm)
    combined = sm.summary["combined_score"].map(
        lambda v: float(v[0] if isinstance(v, (list, tuple)) else v))
    order = combined.rank(pct=True)
    for name in [i for i in sm.summary.index if str(i).startswith("Mid")]:
        assert 0.05 < order[name] < 0.95, (
            f"{name} landed at percentile {order[name]:.2f}, not mid-pack")


def test_percentile_denominator_is_the_whole_universe():
    """Ranks must span [0, 1] across every scored row, with no ties at 1.0
    beyond the single genuine winner."""
    sm = _manager()
    calculate_score(sm)
    q = sm.summary["quality_pct"].map(
        lambda v: float(v[0] if isinstance(v, (list, tuple)) else v))
    assert q.notna().all()
    assert q.min() >= 0.0 and q.max() <= 1.0
    assert (q >= 1.0).sum() <= 1, "several rows tied at the top of their own pool"


def test_every_scored_row_survives_with_its_columns():
    sm = _manager()
    calculate_score(sm)
    assert len(sm.summary) == 29
    for col in ("points", "quality_pct", "value_pct", "combined_score",
                "blend_score"):
        assert col in sm.summary.columns


def test_an_empty_universe_scores_without_raising():
    sm = SummaryManager()
    calculate_score(sm)
    assert sm.summary.empty
