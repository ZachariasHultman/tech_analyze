"""Regression test for _refine_threshold_2d's bound enforcement.

optimize_combo's threshold step used to be a fixed grid
(_threshold_grid_for_metric) with no notion of "realistic" values --
exactly the mechanism that caused the threshold-drift bug in
optimize_weights_and_thresholds (fixed separately: extreme thresholds can
look spuriously good on a small, noisy sample). The replacement,
_refine_threshold_2d, bounds its search to the metric's own observed data
range (2nd-98th percentile, +50% margin) specifically to prevent that
failure mode from recurring here too. This test uses a deliberately
pathological fake objective that always rewards moving further from
center, to prove the bound holds even when the objective actively wants
to escape it -- not just that it happens to stay put on a well-behaved
objective.
"""
import pandas as pd

import analyzer.correlation as corr


def test_refine_threshold_2d_respects_bounds_even_with_pathological_objective(monkeypatch):
    def fake_cv_score(weights_dict, df_total, target_timespans, metrics, thresholds_dict):
        t = thresholds_dict["test_metric"]
        # Unbounded reward for going further from zero in either direction --
        # simulates "extreme threshold looks spuriously better" pathology.
        return abs(t["nok"]) + abs(t["ok"])

    monkeypatch.setattr(corr, "_cv_score", fake_cv_score)

    df_total = pd.DataFrame({
        "company": ["A", "B", "C", "D", "E"],
        "test_metric": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    full_thresholds = {"test_metric": {"nok": 2.0, "ok": 4.0}}

    best_thr, best_cv = corr._refine_threshold_2d(
        "test_metric", {"test_metric": 1.0}, full_thresholds, df_total,
        ["3Y_TOTAL"], ["test_metric"],
    )

    vals = df_total["test_metric"]
    lo, hi = float(vals.quantile(0.02)), float(vals.quantile(0.98))
    margin = max((hi - lo) * 0.5, 1e-6)

    assert best_cv is not None
    assert (lo - margin) - 1e-6 <= best_thr["nok"] <= (hi + margin) + 1e-6
    assert (lo - margin) - 1e-6 <= best_thr["ok"] <= (hi + margin) + 1e-6


def test_refine_threshold_2d_returns_none_cv_when_no_data():
    df_total = pd.DataFrame({"company": ["A", "B"], "other_metric": [1.0, 2.0]})
    full_thresholds = {"test_metric": {"nok": 0.0, "ok": 1.0}}

    best_thr, best_cv = corr._refine_threshold_2d(
        "test_metric", {"test_metric": 1.0}, full_thresholds, df_total,
        ["3Y_TOTAL"], ["test_metric"],
    )

    assert best_cv is None
    assert best_thr == full_thresholds["test_metric"]
