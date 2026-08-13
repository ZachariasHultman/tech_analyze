"""Tests for item 6: panel challenger gate (correlation.py Phase D).

Anything touching real scoring (_panel_year_scores) or the gitignored
metrics.py (_get_default_thresholds) is monkeypatched, so these exercise only
the gate/walk-forward/objective logic — never secret weight/threshold values.
"""
import math

import numpy as np
import pandas as pd

import analyzer.correlation as correlation


# ---- weight scaling (pure, uses correlation-module constants only) ----
def test_scale_weights_from_corrs_floors_and_momentum_cap():
    metrics = ["roe_pe ratio status", "price momentum status", "piotroski f-score status"]
    avg = {
        "roe_pe ratio status": 0.20,       # strongest fundamental -> 2.0
        "price momentum status": 0.20,     # positive but momentum-capped -> 1.0
        "piotroski f-score status": -0.10,  # non-positive -> weight floor 0.5
    }
    w = correlation._scale_weights_from_corrs(avg, metrics)
    assert math.isclose(w["roe_pe ratio status"], 2.0)
    assert math.isclose(w["price momentum status"], correlation.MOMENTUM_WEIGHT_CAP)
    assert math.isclose(w["piotroski f-score status"], 0.5)


# ---- panel objective groups by year and honors return_col ----
def _fake_panel_year_scores(g, fiscal_year, metrics, weights, thresholds,
                            return_col):
    """Stand-in for the vectorised per-year scorer.

    The panel objective now scores through _panel_year_scores (cached score
    matrix) rather than _score_with_weights. These tests exercise the
    gate/walk-forward/averaging logic, so they patch that seam and keep using
    a precomputed "s" column as the score.
    """
    return g["s"], g[return_col]


def _panel(two_years=True):
    rows = []
    # Year A: score and excess return perfectly aligned -> tercile spread +4.0
    for s, r in zip(range(1, 7), range(1, 7)):
        rows.append({"fiscal_year": 2020, "s": s, "fwd_excess_return_1y": r})
    if two_years:
        # Year B: anti-aligned -> spread -4.0 ; mean over two years = 0.0
        for s, r in zip(range(1, 7), range(6, 0, -1)):
            rows.append({"fiscal_year": 2021, "s": s, "fwd_excess_return_1y": r})
    return pd.DataFrame(rows)


def test_panel_avg_quintile_spread_means_over_years(monkeypatch):
    monkeypatch.setattr(correlation, "_panel_year_scores", _fake_panel_year_scores)
    assert math.isclose(
        correlation._panel_avg_quintile_spread({}, _panel(True), ["m"]), 0.0, abs_tol=1e-9
    )
    assert math.isclose(
        correlation._panel_avg_quintile_spread({}, _panel(False), ["m"]), 4.0, abs_tol=1e-9
    )


# ---- walk-forward holds out the target year ----
def test_leave_one_fiscal_year_out_excludes_held_out(monkeypatch):
    monkeypatch.setattr(correlation, "_panel_year_scores", _fake_panel_year_scores)
    monkeypatch.setattr(correlation, "_get_default_thresholds", lambda: {})
    seen_train_years = []

    def fake_opt(train, metrics):
        seen_train_years.append(set(train["fiscal_year"].unique()))
        return {"optimized_weights": {}, "optimized_thresholds": {}, "trial_objectives": []}

    folds = correlation.leave_one_fiscal_year_out(_panel(True), ["m"], optimizer_fn=fake_opt)
    assert {f["fiscal_year"] for f in folds} == {2020, 2021}
    # each refit trained on exactly the *other* year
    assert {2021} in seen_train_years
    assert {2020} in seen_train_years
    # folds and refits are produced in the same sorted-year order, so each
    # fold's held-out year is absent from the training set used to fit it.
    assert all(f["fiscal_year"] not in s for f, s in zip(folds, seen_train_years))


# ---- gate decision truth table ----
def _install_gate_stubs(monkeypatch, opt_spread, eq_spread, significant):
    monkeypatch.setattr(correlation, "_get_default_thresholds", lambda: {})
    monkeypatch.setattr(
        correlation, "leave_one_fiscal_year_out",
        lambda panel, metrics, optimizer_fn: [
            {"fiscal_year": 2020, "optimized_spread": opt_spread, "optimized_ic": 0.1,
             "equal_spread": eq_spread, "equal_ic": 0.1},
        ],
    )
    monkeypatch.setattr(
        correlation, "deflated_sharpe_ratio",
        lambda trials, series: {
            "significant_at_95": significant, "dsr": 0.99 if significant else 0.5,
            "n_trials": len(list(trials)), "sr_benchmark": 0.0,
        },
    )

    def fake_opt(panel, metrics):
        return {"optimized_weights": {"m1": 2.0}, "optimized_thresholds": {},
                "trial_objectives": [0.1, 0.2]}

    return fake_opt


def test_gate_accepts_when_beats_and_significant(monkeypatch):
    fake_opt = _install_gate_stubs(monkeypatch, opt_spread=0.30, eq_spread=0.10, significant=True)
    res = correlation.gate_optimized_weights(pd.DataFrame(), ["m1"], optimizer_fn=fake_opt)
    assert res["accept"] is True
    assert res["chosen_weights"] == {"m1": 2.0}


def test_gate_rejects_when_not_significant(monkeypatch):
    fake_opt = _install_gate_stubs(monkeypatch, opt_spread=0.30, eq_spread=0.10, significant=False)
    res = correlation.gate_optimized_weights(pd.DataFrame(), ["m1"], optimizer_fn=fake_opt)
    assert res["accept"] is False
    assert res["dsr_significant"] is False
    assert res["chosen_weights"] == {"m1": 1.0}  # equal-weight fallback


def test_gate_rejects_when_does_not_beat_equal(monkeypatch):
    fake_opt = _install_gate_stubs(monkeypatch, opt_spread=0.05, eq_spread=0.10, significant=True)
    res = correlation.gate_optimized_weights(pd.DataFrame(), ["m1"], optimizer_fn=fake_opt)
    assert res["accept"] is False
    assert res["beats_equal_weight"] is False
    assert res["chosen_weights"] == {"m1": 1.0}


# ---- confidence bar is configurable, not hardcoded at 0.95 ----
def test_gate_confidence_is_configurable(monkeypatch):
    # dsr=0.90 -- fails the default 0.95 bar, clears a lowered 0.85 bar.
    monkeypatch.setattr(correlation, "_get_default_thresholds", lambda: {})
    monkeypatch.setattr(
        correlation, "leave_one_fiscal_year_out",
        lambda panel, metrics, optimizer_fn: [
            {"fiscal_year": 2020, "optimized_spread": 0.30, "optimized_ic": 0.1,
             "equal_spread": 0.10, "equal_ic": 0.1},
        ],
    )
    monkeypatch.setattr(
        correlation, "deflated_sharpe_ratio",
        lambda trials, series: {
            "significant_at_95": False, "dsr": 0.90,
            "n_trials": len(list(trials)), "sr_benchmark": 0.0,
        },
    )

    def fake_opt(panel, metrics):
        return {"optimized_weights": {"m1": 2.0}, "optimized_thresholds": {},
                "trial_objectives": [0.1, 0.2]}

    default_res = correlation.gate_optimized_weights(pd.DataFrame(), ["m1"], optimizer_fn=fake_opt)
    assert default_res["accept"] is False
    assert default_res["confidence"] == 0.95

    lowered_res = correlation.gate_optimized_weights(
        pd.DataFrame(), ["m1"], optimizer_fn=fake_opt, confidence=0.85
    )
    assert lowered_res["accept"] is True
    assert lowered_res["confidence"] == 0.85
    assert lowered_res["chosen_weights"] == {"m1": 2.0}


def test_duplicate_company_in_a_year_does_not_crash_scoring(monkeypatch):
    """Regression: duplicate labels made the returns series longer than the
    scores, so the combined validity mask matched neither and .loc raised
    IndexError. Real trigger was two 2019 report dates for the same company."""
    import analyzer.fast_score as fast_score

    idx = ["a", "b", "c", "d", "e", "f"]
    G = pd.DataFrame({"m": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=idx)
    monkeypatch.setattr(correlation, "cached_score_matrix",
                        lambda *a, **k: (G, []))

    year = pd.DataFrame({
        "company": idx + ["a"],                       # 'a' appears twice
        "fwd_excess_return_1y": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9],
        "fiscal_year": 2019,
    })
    sv, rv = correlation._panel_year_scores(
        year, 2019, ["m"], {"m": 1.0}, None, "fwd_excess_return_1y")
    assert sv is not None
    assert len(sv) == len(rv) == 6
    assert not sv.index.has_duplicates
