"""Tests for item 5: validation battery check logic (synthetic fixtures).

run_sanity_checks / quintile_sorts_by_fiscal_year / ic_time_series consume
precomputed panels, so these fixtures are built directly as DataFrames — never
depending on the real gitignored metrics.py values (the same guarantee
test_reliability_gate.py gets by monkeypatching _score_snapshot; here the
functions never call it, so direct construction is cleaner).
"""
import math

import numpy as np
import pandas as pd

import analyzer.validation as validation


def _one_year_panel(neg_bottom=True, pct_span=True):
    """20 companies, fiscal year 2020. Bottom-decile ROE = c0,c1.

    raw fwd return = i*0.01 (mean 0.095) -> excess = i*0.01 - 0.095, negative
    for i in 0..9 (exactly half of 20).
    """
    n = 20
    i = np.arange(n)
    comp = np.where(i <= 1, -1.0, 1.0) if neg_bottom else np.where(i <= 1, 1.0, 1.0)
    mean_ret = float((i * 0.01).mean())
    pct = (i + 1) / n if pct_span else np.full(n, 0.5)
    panel = pd.DataFrame({
        "company_id": [f"c{j}" for j in i],
        "fiscal_year": 2020,
        "composite_score_equal": comp,
        "fwd_return_1y": i * 0.01,
        "universe_mean_return_that_year": mean_ret,
        "fwd_excess_return_1y": i * 0.01 - mean_ret,
        "roe_pe ratio status_pct": pct,
    })
    fundamentals = pd.DataFrame({
        "company_id": [f"c{j}" for j in i],
        "fiscal_year": 2020,
        "roe": i.astype(float),
    })
    return panel, fundamentals


def test_sanity_bottom_roe_scores_negative():
    panel, fund = _one_year_panel(neg_bottom=True)
    out = validation.run_sanity_checks(panel, fund)
    assert out["bottom_roe_negative_score_fraction"] == 1.0


def test_sanity_misaligned_scoring_flagged():
    panel, fund = _one_year_panel(neg_bottom=False)
    out = validation.run_sanity_checks(panel, fund)
    # bottom-ROE names score positive -> fraction 0 -> below the 0.5 warn line.
    assert out["bottom_roe_negative_score_fraction"] == 0.0


def test_sanity_excess_return_direction_roughly_half():
    panel, fund = _one_year_panel()
    out = validation.run_sanity_checks(panel, fund)
    assert out["best_rally_year"] == 2020
    assert math.isclose(out["best_rally_negative_excess_fraction"], 0.5, abs_tol=1e-9)


def test_sanity_span_true_when_ranks_spread():
    panel, fund = _one_year_panel(pct_span=True)
    out = validation.run_sanity_checks(panel, fund)
    assert out["within_year_ranks_span_full"] is True


def test_sanity_span_false_when_ranks_compressed():
    panel, fund = _one_year_panel(pct_span=False)
    out = validation.run_sanity_checks(panel, fund)
    assert out["within_year_ranks_span_full"] is False


def _perfect_panel():
    """30 companies, score perfectly ordered with excess return."""
    n = 30
    i = np.arange(n, dtype=float)
    return pd.DataFrame({
        "company_id": [f"c{j}" for j in range(n)],
        "fiscal_year": 2020,
        "composite_score_equal": i,
        "fwd_excess_return_1y": i,
    })


def test_quintile_monotonic_and_positive_spread():
    res = validation.quintile_sorts_by_fiscal_year(_perfect_panel())
    yr = res["per_year"][0]
    assert yr["n_buckets"] == 5
    assert yr["monotonic"] is True
    assert yr["spread"] > 0


def test_ic_perfect_ordering_is_one():
    res = validation.ic_time_series(_perfect_panel())
    assert math.isclose(res["mean_ic"], 1.0, abs_tol=1e-9)


def test_run_validation_battery_returns_full_dict(tmp_path):
    # Two fiscal years, dense enough for the single-factor Fama-MacBeth.
    rows = []
    rng = np.random.default_rng(0)
    for fy in (2020, 2021):
        for j in range(30):
            score = float(j)
            rows.append({
                "company_id": f"c{j}",
                "fiscal_year": fy,
                "composite_score_equal": score,
                "composite_score_tiered": score,
                "combined_score": 0.25,
                "roe_pe ratio status_pct": (j + 1) / 30.0,
                "roe_pe ratio status_score": score + rng.normal(),
                "fwd_return_1y": 0.01 * j,
                "universe_mean_return_that_year": 0.145,
                "fwd_excess_return_1y": 0.01 * j - 0.145,
            })
    panel = pd.DataFrame(rows)
    fund = pd.DataFrame({
        "company_id": panel["company_id"],
        "fiscal_year": panel["fiscal_year"],
        "roe": panel["composite_score_equal"],
    })
    p_scores = tmp_path / "panel_scores.csv"
    p_fund = tmp_path / "panel_fundamentals.csv"
    panel.to_csv(p_scores, index=False)
    fund.to_csv(p_fund, index=False)

    res = validation.run_validation_battery(str(p_scores), str(p_fund))
    for key in ("header", "sanity", "quintiles", "ic", "per_metric_ic",
                "fama_macbeth_single", "fama_macbeth_multi"):
        assert key in res
    assert res["header"]["n_periods"] == 2
    assert res["fama_macbeth_single"]["n_periods_used"] == 2
