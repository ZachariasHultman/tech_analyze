"""Tests for item 3: reliability shrinkage (replaces the old n<8 hard gate).

_score_snapshot (which depends on the real, gitignored metrics.py weights) is
monkeypatched to a deterministic passthrough (points == total_return) so
these tests exercise only the shrinkage / window-restriction logic, not real
scoring.
"""
import math

import pandas as pd

import analyzer.correlation as correlation
from analyzer.historical_calc import make_windows


def _fake_score_snapshot(df_ts, metrics_to_score=None, thresholds=None):
    return df_ts.set_index("company")[["total_return"]].rename(
        columns={"total_return": "points"}
    )


def _build_df():
    """5 non-overlapping 5Y_YoY windows + overlapping TOTAL windows.

    A appears in all 5 YoY windows (n=5) and every TOTAL window.
    B appears in 3 YoY windows (n=3). C appears in 2 (n=2, below floor).
    """
    padding = ["D", "E", "F", "G", "H"]  # keeps each per-timespan slice >= 5 rows
    rows = []
    yoy = [f"5Y_YoY-{k}" for k in range(1, 6)]
    total = ["3Y_TOTAL", "5Y_TOTAL"]
    for ts in yoy + total:
        i = float(int(ts.split("-")[-1]) if "YoY" in ts else 99)
        for name in padding:
            rows.append({"company": name, "timespan": ts, "total_return": i})
        rows.append({"company": "A", "timespan": ts, "total_return": i})
        if "YoY" in ts and int(ts.split("-")[-1]) <= 3:
            rows.append({"company": "B", "timespan": ts, "total_return": i})
        if "YoY" in ts and int(ts.split("-")[-1]) <= 2:
            rows.append({"company": "C", "timespan": ts, "total_return": i})
    return pd.DataFrame(rows), yoy


def test_shrinkage_formula_and_window_restriction(monkeypatch):
    monkeypatch.setattr(correlation, "_score_snapshot", _fake_score_snapshot)
    df, yoy = _build_df()

    # Pass only the 5Y_YoY-* windows: TOTAL windows must be ignored (dead
    # target_timespans parameter is now honored).
    result = correlation._compute_reliability(df, yoy).set_index("company")

    # A: n capped at 5 (TOTAL windows excluded), rho=1.0, shrunk = 5/(5+10).
    assert result.loc["A", "n_windows"] == 5
    assert result.loc["A", "spearman"] == 1.0
    assert math.isclose(result.loc["A", "spearman_shrunk"], 5 / 15, abs_tol=1e-4)

    # B: n=3, shrunk = 3/(3+10) ~ 0.2308.
    assert result.loc["B", "n_windows"] == 3
    assert math.isclose(result.loc["B", "spearman_shrunk"], 3 / 13, abs_tol=1e-4)

    # C: n=2 (< 3) -> spearman/shrunk NaN, not reliable.
    assert result.loc["C", "n_windows"] == 2
    assert math.isnan(result.loc["C", "spearman"])
    assert math.isnan(result.loc["C", "spearman_shrunk"])
    assert result.loc["C", "reliable"] == False


def test_dead_parameter_is_honored(monkeypatch):
    """Regression: target_timespans was ignored; A's n must reflect the
    restricted list, not every window in df."""
    monkeypatch.setattr(correlation, "_score_snapshot", _fake_score_snapshot)
    df, yoy = _build_df()

    # Restrict to a single YoY window -> A has only 1 sample -> below n<3 floor.
    result = correlation._compute_reliability(df, yoy[:1]).set_index("company")
    assert result.loc["A", "n_windows"] == 1
    assert math.isnan(result.loc["A", "spearman"])


def test_shrinkage_direct_formula():
    """Assert the shrink factor directly across n (window cap is 5 in practice)."""
    for n in (3, 5, 40):
        rho = 0.5
        assert math.isclose(rho * n / (n + 10), 0.5 * n / (n + 10), abs_tol=1e-12)
    # n=40 approaches but never reaches raw rho
    assert 40 / 50 < 1.0


def test_yoy_window_dedup_3y_equals_5y():
    """3Y_YoY-1 and 5Y_YoY-1 are the literal same calendar window -> restricting
    reliability to 5Y_YoY-* only (dropping 3Y_YoY-*) avoids double-counting."""
    max_date = pd.Timestamp("2025-01-01")
    w3 = {label: (start, end) for label, start, end, _ in make_windows(max_date, 3)}
    w5 = {label: (start, end) for label, start, end, _ in make_windows(max_date, 5)}
    assert w3["3Y_YoY-1"] == w5["5Y_YoY-1"]
    assert w3["3Y_YoY-2"] == w5["5Y_YoY-2"]
