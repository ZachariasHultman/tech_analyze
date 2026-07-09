"""Regression test for item 3: _compute_reliability's n<8 gate.

_score_snapshot (which depends on real, gitignored metrics.py weights) is
monkeypatched to a deterministic passthrough (points == total_return) so
this test only exercises the n-windows gating logic, not real scoring.
"""
import math

import pandas as pd

import analyzer.correlation as correlation


def _fake_score_snapshot(df_ts, metrics_to_score=None, thresholds=None):
    return df_ts.set_index("company")[["total_return"]].rename(columns={"total_return": "points"})


def test_compute_reliability_nulls_rho_below_n8(monkeypatch):
    monkeypatch.setattr(correlation, "_score_snapshot", _fake_score_snapshot)

    padding = ["D", "E", "F", "G", "H"]  # keeps each per-timespan slice >= 5 rows
    rows = []
    for i in range(10):
        timespan = f"W{i}"
        for name in padding:
            rows.append({"company": name, "timespan": timespan, "total_return": float(i)})
        rows.append({"company": "A", "timespan": timespan, "total_return": float(i)})  # n=10
        if i < 5:
            rows.append({"company": "B", "timespan": timespan, "total_return": float(i)})  # n=5
        if i < 2:
            rows.append({"company": "C", "timespan": timespan, "total_return": float(i)})  # n=2

    df = pd.DataFrame(rows)
    result = correlation._compute_reliability(df, target_timespans=None).set_index("company")

    # n=10 (>=8): normal gate, perfect monotonic points==returns -> rho=1.0, reliable
    assert result.loc["A", "n_windows"] == 10
    assert result.loc["A", "reliable"] == True
    assert result.loc["A", "spearman"] == 1.0

    # n=5 (in [3,8)): rho nulled to NaN, not just reliable=False
    assert result.loc["B", "n_windows"] == 5
    assert result.loc["B", "reliable"] == False
    assert math.isnan(result.loc["B", "spearman"])

    # n=2 (<3): pre-existing behavior unchanged
    assert result.loc["C", "n_windows"] == 2
    assert result.loc["C", "reliable"] == False
    assert math.isnan(result.loc["C", "spearman"])
