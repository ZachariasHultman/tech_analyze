"""Regression test: earnings quality status must have a nonzero WEIGHT_FLOORS
entry in correlation.py (a committed file -- unlike the tier assignment
itself, which lives in the gitignored analyzer/metrics.py and is
deliberately not asserted here per this project's testing conventions).

operatingCashFlow is populated live for Nordic stocks but Avanza exposes no
historical OCF series, so the backtest/optimizer can never compute a real
correlation for this metric -- its weight always resolves to whatever's in
WEIGHT_FLOORS (see optimize_weights_and_thresholds's Step 2: metrics absent
from positive_metrics get exactly WEIGHT_FLOORS.get(m, 0.0), nothing else).
Without a floor entry, this metric silently contributes zero to scoring
regardless of tier -- exactly the bug this floor exists to prevent.
"""
from analyzer.correlation import WEIGHT_FLOORS
from analyzer.summary_manager import SummaryManager


def test_earnings_quality_has_a_positive_weight_floor():
    assert WEIGHT_FLOORS.get("earnings quality status", 0) > 0


def test_earnings_quality_scores_nonzero_when_floor_weight_applied():
    m = SummaryManager()
    m._weight_overrides = {"earnings quality status": WEIGHT_FLOORS["earnings quality status"]}
    override = {"nok": 0.2, "ok": 0.7}
    row = {"earnings quality status": [1.25]}

    score = m._assign_points(row, "earnings quality status", threshold_override=override)

    assert score == WEIGHT_FLOORS["earnings quality status"]
