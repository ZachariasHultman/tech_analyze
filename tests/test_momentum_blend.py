"""The watchlist ranks on a 50/50 blend of momentum and fundamentals.

Measured on the fiscal-year panel (7 years, 811 company-years), ranking the
top-25 basket on `pts` alone returns +2.59%/yr excess (p=0.495, positive 4 of
7 years); the blend returns +6.36%/yr (p=0.146, 4 of 7). Neither is close to
significant -- the blend's case rests on the ordering, where it measures
IC +0.118 (p=0.052, positive 7 of 7) against pts' +0.081.

Momentum enters as the *raw* SMA200 distance rather than its banded metric
score: ranking on the score measured IC +0.092 (p=0.117, 5 of 7), ranking on
the underlying ratio +0.112 (p=0.045, 7 of 7).

No test here depends on real (gitignored) metrics.py weights -- they are pinned
via `_weight_overrides`.
"""

import numpy as np
import pandas as pd
import pytest

from analyzer.config import MOMENTUM_BLEND_WEIGHT
from analyzer.data_processing import calculate_score
from analyzer.summary_manager import SummaryManager

_SECTOR = [{"sectorId": "1", "sectorName": "Industri"}]


def _manager(rows):
    """rows: {name: {metric: value}}. Weights pinned so real metrics.py values
    (gitignored) cannot influence the outcome."""
    sm = SummaryManager()
    sm._weight_overrides = {
        "roe_pe ratio status": 1.0,
        "net debt - ebitda status": 1.0,
        "price momentum status": 1.0,
    }
    for name, metrics in rows.items():
        sm._initialize_template(name, _SECTOR)
        for metric, value in metrics.items():
            sm._update(name, _SECTOR, metric, value)
    return sm


def _val(df, name, col):
    v = df.loc[name, col]
    return float(v[0] if isinstance(v, (list, tuple)) else v)


def _universe(n=30):
    """A spread-out universe so percentile ranks are well defined."""
    rows = {}
    for i in range(n):
        rows[f"C{i:02d}"] = {
            "roe_pe ratio status": 0.2 + i * 0.1,
            "net debt - ebitda status": 5.0 - i * 0.15,
            "price momentum status": -0.20 + i * 0.02,
        }
    return rows


# ---------------------------------------------------------------- the columns

def test_blend_columns_are_produced():
    sm = _manager(_universe())
    calculate_score(sm)
    for col in ("momentum_pct", "fundamental_pct", "blend_score"):
        assert col in sm.summary.columns, f"{col} missing"


def test_blend_is_the_weighted_average_of_the_two_percentiles():
    sm = _manager(_universe())
    calculate_score(sm)
    w = MOMENTUM_BLEND_WEIGHT
    for name in sm.summary.index:
        expected = (
            w * _val(sm.summary, name, "momentum_pct")
            + (1 - w) * _val(sm.summary, name, "fundamental_pct")
        )
        assert _val(sm.summary, name, "blend_score") == pytest.approx(expected)


def test_percentiles_are_bounded():
    sm = _manager(_universe())
    calculate_score(sm)
    for col in ("momentum_pct", "fundamental_pct", "blend_score"):
        vals = pd.to_numeric(sm.summary[col], errors="coerce").dropna()
        assert vals.between(0.0, 1.0).all(), f"{col} outside 0..1"


# --------------------------------------------------- ranks on raw, not banded

def test_momentum_percentile_follows_the_raw_ratio_not_the_banded_score():
    """The banding into ok/nok bands discards signal; the rank must come from
    the underlying ratio, so two companies inside the same band still separate.
    """
    rows = _universe()
    # Both sit above any plausible "ok" band, but one is far stronger.
    rows["Hot"] = {"roe_pe ratio status": 1.0, "net debt - ebitda status": 2.0,
                   "price momentum status": 0.40}
    rows["Warm"] = {"roe_pe ratio status": 1.0, "net debt - ebitda status": 2.0,
                    "price momentum status": 0.12}
    sm = _manager(rows)
    calculate_score(sm)
    assert (_val(sm.summary, "Hot", "momentum_pct")
            > _val(sm.summary, "Warm", "momentum_pct"))


def test_momentum_ordering_is_monotone_in_the_raw_value():
    sm = _manager(_universe())
    calculate_score(sm)
    raw = pd.Series({n: _val(sm.summary, n, "price momentum status")
                     for n in sm.summary.index})
    pct = pd.Series({n: _val(sm.summary, n, "momentum_pct")
                     for n in sm.summary.index})
    assert raw.rank().equals(pct.rank())


# ----------------------------------------------------------- missing momentum

def test_missing_momentum_falls_back_to_the_fundamental_rank():
    """Momentum is a tilt on the composite, not a filter -- a company with no
    usable price history must keep its fundamental standing rather than drop
    out of the ranking."""
    rows = _universe()
    rows["NoPrice"] = {"roe_pe ratio status": 3.0,
                       "net debt - ebitda status": 0.5,
                       "price momentum status": None}
    sm = _manager(rows)
    calculate_score(sm)
    assert pd.isna(_val(sm.summary, "NoPrice", "momentum_pct"))
    assert (_val(sm.summary, "NoPrice", "blend_score")
            == pytest.approx(_val(sm.summary, "NoPrice", "fundamental_pct")))


# ------------------------------------------------------- it changes the order

def test_momentum_can_outrank_a_stronger_fundamental_score():
    """The whole point of the blend: strong price trend lifts a merely decent
    business above a better one that is trending down. If this fails the blend
    is not actually influencing the ranking.

    Momentum's weight inside `points` is pinned low here so the two halves can
    be told apart -- momentum deliberately stays in the fundamental composite
    as well (effective weight ~0.52, see config.MOMENTUM_BLEND_WEIGHT), and at
    equal weight a company with bottom-ranked momentum cannot also hold a top
    fundamental rank, which is what this test needs to set up.
    """
    rows = _universe()
    rows["Trending"] = {"roe_pe ratio status": 1.2,
                        "net debt - ebitda status": 2.5,
                        "price momentum status": 0.60}
    rows["Falling"] = {"roe_pe ratio status": 4.5,
                       "net debt - ebitda status": 0.2,
                       "price momentum status": -0.45}
    sm = _manager(rows)
    sm._weight_overrides["price momentum status"] = 0.05
    calculate_score(sm)

    assert (_val(sm.summary, "Falling", "fundamental_pct")
            > _val(sm.summary, "Trending", "fundamental_pct")), "fixture broken"
    assert (_val(sm.summary, "Trending", "blend_score")
            > _val(sm.summary, "Falling", "blend_score"))


def test_pts_is_left_untouched_by_the_blend():
    """`points` still drives the sell-signal deterioration check, so the blend
    must sit alongside it rather than replace it."""
    sm = _manager(_universe())
    calculate_score(sm)
    assert "points" in sm.summary.columns
    pts = pd.to_numeric(sm.summary["points"], errors="coerce")
    assert pts.notna().all()
    # not a percentile -- it is the raw summed score
    assert not pts.between(0.0, 1.0).all()


# ------------------------------------------------- survives the pooling split

def test_blend_columns_land_on_every_scored_row():
    """calculate_score scores one pooled frame; the blend columns are outputs
    of that pass and must be populated for every company in it. This used to
    also assert they survived the slice back into a second investment frame --
    that frame no longer exists (config.EXCLUDED_SECTORS)."""
    sm = SummaryManager()
    sm._weight_overrides = {"roe_pe ratio status": 1.0,
                            "net debt - ebitda status": 1.0,
                            "price momentum status": 1.0}
    for i in range(26):
        n = f"Reg{i:02d}"
        sm._initialize_template(n, _SECTOR)
        sm._update(n, _SECTOR, "roe_pe ratio status", 0.2 + i * 0.1)
        sm._update(n, _SECTOR, "net debt - ebitda status", 5.0 - i * 0.15)
        sm._update(n, _SECTOR, "price momentum status", -0.2 + i * 0.02)

    calculate_score(sm)
    frame = sm.summary
    assert "blend_score" in frame.columns
    assert pd.to_numeric(frame["blend_score"], errors="coerce").notna().all()
