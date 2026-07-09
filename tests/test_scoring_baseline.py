"""Baseline regression tests for SummaryManager scoring helpers.

These lock in behavior of `_score_scalar` (a closure inside `_assign_points`)
and `_assign_points_rank`. Neither test depends on real (gitignored)
metrics.py threshold/weight values: weight is pinned via `_weight_overrides`
and, where direction matters, via monkeypatching the module-level
DIRECTION_OVERRIDES dict imported into summary_manager.
"""
import math

import analyzer.summary_manager as summary_manager
from analyzer.summary_manager import SummaryManager


def _manager_with_weight(weight=1.0):
    m = SummaryManager()
    m._weight_overrides = {"test_metric": weight}
    return m


# ---------------------------------------------------------------- _score_scalar (via _assign_points)

def test_score_scalar_direction_plus1_extremes():
    m = _manager_with_weight(1.0)
    row = {"test_metric": [0.0]}
    override = {"nok": 0.0, "ok": 10.0}

    row["test_metric"] = [10.0]
    assert m._assign_points(row, "test_metric", threshold_override=override) == 1.0

    row["test_metric"] = [0.0]
    assert m._assign_points(row, "test_metric", threshold_override=override) == -1.0

    row["test_metric"] = [20.0]  # beyond ok, still clamps to +weight
    assert m._assign_points(row, "test_metric", threshold_override=override) == 1.0

    row["test_metric"] = [-5.0]  # beyond nok, still clamps to -weight
    assert m._assign_points(row, "test_metric", threshold_override=override) == -1.0


def test_score_scalar_direction_plus1_interpolation_midpoint():
    m = _manager_with_weight(1.0)
    override = {"nok": 0.0, "ok": 10.0}
    row = {"test_metric": [5.0]}
    result = m._assign_points(row, "test_metric", threshold_override=override)
    assert math.isclose(result, 0.0, abs_tol=1e-9)


def test_score_scalar_direction_minus1(monkeypatch):
    monkeypatch.setitem(summary_manager.DIRECTION_OVERRIDES, "test_metric_neg", -1)
    m = SummaryManager()
    m._weight_overrides = {"test_metric_neg": 1.0}
    override = {"nok": 10.0, "ok": 0.0}

    row = {"test_metric_neg": [0.0]}
    assert m._assign_points(row, "test_metric_neg", threshold_override=override) == 1.0

    row = {"test_metric_neg": [10.0]}
    assert m._assign_points(row, "test_metric_neg", threshold_override=override) == -1.0

    row = {"test_metric_neg": [5.0]}
    result = m._assign_points(row, "test_metric_neg", threshold_override=override)
    assert math.isclose(result, 0.0, abs_tol=1e-9)


def test_score_scalar_zero_weight_short_circuits():
    m = SummaryManager()
    m._weight_overrides = {"test_metric": 0}
    row = {"test_metric": [999.0]}
    override = {"nok": 0.0, "ok": 10.0}
    assert m._assign_points(row, "test_metric", threshold_override=override) == 0


# ---------------------------------------------------------------- _assign_points_rank

def test_assign_points_rank_boundaries():
    m = _manager_with_weight(1.0)
    assert m._assign_points_rank({}, "test_metric", 0.65) == 1.0
    assert m._assign_points_rank({}, "test_metric", 0.35) == -1.0


def test_assign_points_rank_extremes_clamp():
    m = _manager_with_weight(1.0)
    assert m._assign_points_rank({}, "test_metric", 1.0) == 1.0
    assert m._assign_points_rank({}, "test_metric", 0.0) == -1.0


def test_assign_points_rank_midpoint_interpolation():
    m = _manager_with_weight(1.0)
    result = m._assign_points_rank({}, "test_metric", 0.5)
    assert math.isclose(result, 0.0, abs_tol=1e-9)


def test_assign_points_rank_nan_is_zero():
    m = _manager_with_weight(1.0)
    assert m._assign_points_rank({}, "test_metric", float("nan")) == 0.0


def test_assign_points_rank_zero_weight():
    m = SummaryManager()
    m._weight_overrides = {"test_metric": 0}
    assert m._assign_points_rank({}, "test_metric", 0.9) == 0.0
