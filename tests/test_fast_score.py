"""The vectorised scorer must be exactly the production scorer, only faster.

This is a speed refactor, so the test asserts *equality* with the existing
path rather than closeness of any downstream statistic -- if the two ever
diverge, the optimizer is silently optimising something other than what live
scoring computes, which is the exact class of bug this whole change set exists
to remove.

The bonus/malus term is the trap: it is not linear in the weights (it depends
on which metrics are non-zero), so a naive `G @ w` that folds it into the
matrix passes casual inspection and produces plausible-but-wrong numbers.
test_zero_weight_on_a_highest_weight_metric_kills_the_bonus pins it.
"""

import numpy as np
import pandas as pd
import pytest

from analyzer.correlation import (_all_scored_metrics, _get_default_thresholds,
                                  _score_with_weights)
from analyzer.fast_score import (build_score_matrix, cached_score_matrix,
                                 clear_cache, points_from_matrix)
from analyzer.metrics import HIGHEST_WEIGHT_METRICS


def _panel_year():
    pytest.importorskip("pandas")
    import os
    path = "data/panel_fundamentals.csv"
    if not os.path.exists(path):
        pytest.skip("no local panel; run --backfill-panel")
    df = pd.read_csv(path)
    df["company"] = df["company_id"]
    year = df[df.fiscal_year == 2024].copy()
    if len(year) < 25:
        pytest.skip("2024 cross-section too small")
    # _score_with_weights needs a target column to align against.
    year["fwd_excess_return_1y"] = np.linspace(-0.3, 0.3, len(year))
    return year


def _weights(rng, metrics):
    return {m: float(rng.choice([0.0, 0.25, 0.5, 1.0, 1.5, 2.0])) for m in metrics}


def test_matches_production_scoring_exactly():
    year = _panel_year()
    metrics = _all_scored_metrics()
    thr = _get_default_thresholds()
    G, hw = build_score_matrix(year, metrics, thr)
    assert G is not None and not G.empty

    rng = np.random.default_rng(0)
    for _ in range(8):
        w = _weights(rng, metrics)
        fast = points_from_matrix(G, hw, w)
        slow, _ = _score_with_weights(year, metrics, dict(w), thr,
                                      return_col="fwd_excess_return_1y")
        assert slow is not None
        aligned = fast.reindex(slow.index)
        assert np.abs(aligned - slow).max() < 1e-9


def test_equal_weight_matches_production():
    year = _panel_year()
    metrics = _all_scored_metrics()
    thr = _get_default_thresholds()
    G, hw = build_score_matrix(year, metrics, thr)
    w = {m: 1.0 for m in metrics}
    slow, _ = _score_with_weights(year, metrics, dict(w), thr,
                                  return_col="fwd_excess_return_1y")
    fast = points_from_matrix(G, hw, w).reindex(slow.index)
    assert np.abs(fast - slow).max() < 1e-9


def test_zero_weight_on_a_highest_weight_metric_kills_the_bonus():
    # A metric at weight 0 scores exactly 0, so it is neither >0 nor <0 and
    # the all-positive / all-negative bonus can never fire. Folding the bonus
    # into the matrix would miss this.
    G = pd.DataFrame({"a": [1.0, -1.0], "b": [1.0, -1.0]}, index=["x", "y"])
    hw = ["a", "b"]
    with_bonus = points_from_matrix(G, hw, {"a": 1.0, "b": 1.0})
    assert with_bonus.tolist() == [3.0, -3.0]  # 2 + bonus, -2 - malus
    no_bonus = points_from_matrix(G, hw, {"a": 0.0, "b": 1.0})
    assert no_bonus.tolist() == [1.0, -1.0]  # bonus suppressed


def test_linearity_in_weights_holds():
    G = pd.DataFrame({"a": [2.0, -1.0], "b": [0.5, 3.0]}, index=["x", "y"])
    doubled = points_from_matrix(G, [], {"a": 2.0, "b": 2.0})
    single = points_from_matrix(G, [], {"a": 1.0, "b": 1.0})
    assert doubled.tolist() == pytest.approx((2 * single).tolist())


def _count_scoring_calls(monkeypatch):
    """Record which metric sets actually get re-scored."""
    from analyzer import correlation, fast_score

    calls = []
    real = correlation._score_snapshot

    def spy(df, metrics_to_score=None, **kw):
        calls.append(list(metrics_to_score or []))
        return real(df, metrics_to_score=metrics_to_score, **kw)

    monkeypatch.setattr(correlation, "_score_snapshot", spy)
    return calls


# A cold build costs one real scoring pass plus the two threshold-probe
# passes (shift the band up, shift it down) that classify each metric.
_COLD_PASSES = 3


def test_repeated_query_does_not_rescore(monkeypatch):
    year = _panel_year()
    metrics = _all_scored_metrics()
    thr = _get_default_thresholds()
    clear_cache()
    calls = _count_scoring_calls(monkeypatch)

    first = cached_score_matrix(year, metrics, thr, group_key=2024)
    assert len(calls) == _COLD_PASSES
    second = cached_score_matrix(year, metrics, thr, group_key=2024)
    assert len(calls) == _COLD_PASSES, "identical query should be a pure cache hit"
    assert np.abs(first[0].to_numpy() - second[0].to_numpy()).max() == 0.0
    assert first[1] == second[1]


def test_threshold_sweep_is_free_for_rank_path_metrics(monkeypatch):
    """The point of the probe: metrics scored by cross-sectional rank ignore
    thresholds, so sweeping their thresholds must cost nothing."""
    year = _panel_year()
    metrics = _all_scored_metrics()
    thr = _get_default_thresholds()

    clear_cache()
    base, _ = cached_score_matrix(year, metrics, thr, group_key=2024)
    calls = _count_scoring_calls(monkeypatch)

    bumped = {m: {"nok": v["nok"] - 0.137, "ok": v["ok"] + 0.191}
              for m, v in thr.items()}
    after, _ = cached_score_matrix(year, metrics, bumped, group_key=2024)

    rescored = {m for c in calls for m in c}
    assert rescored != set(metrics), (
        "sweeping thresholds rebuilt every column -- the independence probe "
        "is not classifying anything"
    )
    # Whatever was NOT rescored must be byte-identical to the baseline.
    for m in set(metrics) - rescored:
        assert np.abs(base[m].to_numpy() - after[m].to_numpy()).max() == 0.0


def test_threshold_sensitive_metric_is_not_marked_independent():
    """A metric that genuinely reads thresholds must keep them in its key,
    otherwise the optimizer would score it against a stale band."""
    from analyzer import fast_score

    year = _panel_year()
    metrics = _all_scored_metrics()
    thr = _get_default_thresholds()
    clear_cache()
    cached_score_matrix(year, metrics, thr, group_key=2024)

    independent = {m for gk, m in fast_score._THRESHOLD_INDEPENDENT if gk[0] == 2024}
    # For every metric the probe declared independent, a wildly different
    # threshold must genuinely produce the same column.
    wild = {m: {"nok": v["nok"] * 0.1 - 5, "ok": v["ok"] * 10 + 5}
            for m, v in thr.items()}
    base, _ = build_score_matrix(year, metrics, thr)
    alt, _ = build_score_matrix(year, metrics, wild)
    for m in independent:
        assert np.abs(base[m].to_numpy() - alt[m].to_numpy()).max() == 0.0, (
            f"{m} was cached as threshold-independent but is not"
        )


def test_cache_clear_forces_a_rebuild(monkeypatch):
    year = _panel_year()
    metrics = _all_scored_metrics()
    thr = _get_default_thresholds()
    clear_cache()
    before = cached_score_matrix(year, metrics, thr, group_key=2024)
    clear_cache()
    calls = _count_scoring_calls(monkeypatch)
    after = cached_score_matrix(year, metrics, thr, group_key=2024)
    assert len(calls) == _COLD_PASSES
    assert np.abs(before[0].to_numpy() - after[0].to_numpy()).max() == 0.0


def test_per_column_cache_matches_a_single_full_pass():
    """Columns assembled from the memo must equal one monolithic build."""
    year = _panel_year()
    metrics = _all_scored_metrics()
    thr = _get_default_thresholds()
    clear_cache()
    assembled, hw_a = cached_score_matrix(year, metrics, thr, group_key=2024)
    monolithic, hw_b = build_score_matrix(year, metrics, thr)
    assert hw_a == hw_b
    aligned = monolithic.reindex(index=assembled.index, columns=assembled.columns)
    assert np.abs(assembled.to_numpy() - aligned.to_numpy()).max() == 0.0


def test_highest_weight_columns_are_detected():
    year = _panel_year()
    metrics = _all_scored_metrics()
    _, hw = build_score_matrix(year, metrics, _get_default_thresholds())
    assert hw, "expected at least one highest-weight metric to be scored"
    assert set(hw) <= set(HIGHEST_WEIGHT_METRICS)
