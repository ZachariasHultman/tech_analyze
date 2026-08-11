"""Tests for item 3: build_scores_panel (per-year scoring, demeaning, fwd return).

Everything that touches the real gitignored metrics.py (_score_snapshot,
_all_scored_metrics) and the real snapshots (get_hist_data) is monkeypatched, so
the test exercises only the panel logic: per-fiscal-year-independent scoring, the
1-year forward-return anchor, and within-year demeaning.
"""
import math

import numpy as np
import pandas as pd

import analyzer.panel as panel


def _ohlc(pairs):
    idx = pd.to_datetime([d for d, _ in pairs])
    return pd.DataFrame({"close": [v for _, v in pairs]}, index=idx)


def _hist_df():
    # OHLC ends 2022-06-01: 2021 fiscal year has a +1y anchor, 2022 does not.
    a = _ohlc([("2020-01-01", 90), ("2021-02-10", 100),
               ("2022-02-10", 120), ("2022-06-01", 130)])
    b = _ohlc([("2020-01-01", 45), ("2021-02-10", 50),
               ("2022-02-10", 55), ("2022-06-01", 60)])
    df = pd.DataFrame({"ohlc": [a, b]}, index=["A", "B"])
    df.index.name = "company"
    return df


def _fundamentals():
    rows = []
    for fy, rd in [(2021, "2021-02-10"), (2022, "2022-02-10")]:
        for i, c in enumerate(["A", "B"]):
            rows.append({
                "company": c, "company_id": c, "fiscal_year": fy,
                "report_date": rd, "sector": "Industrials",
                "roe_pe ratio status": 0.1 * (i + 1) + fy,  # distinct per company
            })
    return pd.DataFrame(rows)


def _make_fake_score(calls):
    def _fake(df, metrics_to_score=None, thresholds=None, weight_overrides=None):
        companies = df["company"].tolist()
        calls.append((frozenset(companies), weight_overrides is not None))
        pts = 1.0 if weight_overrides is not None else 2.0  # equal vs tiered
        return pd.DataFrame(
            {
                "points": [pts] * len(companies),
                "quality_pct": [0.5] * len(companies),
                "value_pct": [0.5] * len(companies),
                "combined_score": [0.25] * len(companies),
                "roe_pe ratio status_score": [pts] * len(companies),
            },
            index=companies,
        )
    return _fake


def _run(monkeypatch):
    calls = []
    monkeypatch.setattr(panel, "get_hist_data", lambda data_dir="data": _hist_df())
    monkeypatch.setattr(panel, "_all_scored_metrics", lambda: ["roe_pe ratio status"])
    monkeypatch.setattr(panel, "_score_snapshot", _make_fake_score(calls))
    out = panel.build_scores_panel(_fundamentals(), "ignored")
    return out, calls


def test_forward_return_and_demeaning(monkeypatch):
    out, _ = _run(monkeypatch)
    y21 = out[out["fiscal_year"] == 2021].set_index("company_id")

    # A: 120/100-1 = 0.20 ; B: 55/50-1 = 0.10
    assert math.isclose(y21.loc["A", "fwd_return_1y"], 0.20, abs_tol=1e-9)
    assert math.isclose(y21.loc["B", "fwd_return_1y"], 0.10, abs_tol=1e-9)

    # universe mean = 0.15 ; excess demeaned within the year
    assert math.isclose(y21.loc["A", "universe_mean_return_that_year"], 0.15, abs_tol=1e-9)
    assert math.isclose(y21.loc["A", "fwd_excess_return_1y"], 0.05, abs_tol=1e-9)
    assert math.isclose(y21.loc["B", "fwd_excess_return_1y"], -0.05, abs_tol=1e-9)


def test_missing_forward_anchor_is_nan_but_still_scored(monkeypatch):
    out, _ = _run(monkeypatch)
    y22 = out[out["fiscal_year"] == 2022].set_index("company_id")

    # 2022 anchor (2023-02-10) is past the OHLC ceiling (2022-06-01) -> NaN.
    assert y22["fwd_return_1y"].isna().all()
    assert y22["fwd_excess_return_1y"].isna().all()
    assert y22["universe_mean_return_that_year"].isna().all()
    # but the rows still carry composite scores (ranked on the full cross-section)
    assert (y22["composite_score_equal"] == 1.0).all()


def test_equal_and_tiered_both_wired(monkeypatch):
    out, _ = _run(monkeypatch)
    assert (out["composite_score_equal"] == 1.0).all()   # weight_overrides given
    assert (out["composite_score_tiered"] == 2.0).all()  # no overrides


def test_scoring_is_per_fiscal_year_never_pooled(monkeypatch):
    out, calls = _run(monkeypatch)
    # 2 fiscal years x 2 weightings (equal + tiered) = 4 scoring calls.
    assert len(calls) == 4
    # every scoring call sees exactly one fiscal year's companies ({A, B}),
    # never a pooled cross-section spanning years (which would still be {A,B}
    # here, so also assert the call count encodes the per-year split).
    for companies, _ in calls:
        assert companies == frozenset({"A", "B"})
    # per-year percentile column spans the year independently
    y21 = out[out["fiscal_year"] == 2021]["roe_pe ratio status_pct"].dropna()
    assert set(y21) == {0.5, 1.0}
