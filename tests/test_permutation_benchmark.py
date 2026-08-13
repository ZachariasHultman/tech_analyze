"""The challenger gate must measure its null, not approximate it.

deflated_sharpe_ratio's `sigma_sr` was the standard deviation of the objective
across grid *candidates* -- not the sampling noise of a Sharpe estimate, which
is what the formula needs. That made the bar move with how the search was
configured: the real panel search evaluated 1221 candidates but produced only
90 distinct objective values, so duplicates shrank sigma while inflating the
trial count, both in ways unrelated to the evidence.

permutation_benchmark replaces it with the real thing: shuffle the target
within each fiscal year, refit, and see how good a result this exact search
reaches when there is provably nothing to find.
"""

import numpy as np
import pandas as pd
import pytest

import analyzer.correlation as correlation
from analyzer.correlation import permutation_benchmark, permutation_p_value
from analyzer.stats_utils import deflated_sharpe_ratio


def _panel(n_years=4, n_per_year=30, signal=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for y in range(2020, 2020 + n_years):
        for i in range(n_per_year):
            s = float(rng.normal())
            rows.append({
                "company": f"c{i}", "company_id": f"c{i}", "fiscal_year": y,
                "s": s,
                "fwd_excess_return_1y": signal * s + float(rng.normal()),
            })
    return pd.DataFrame(rows)


def _fake_year_scores(g, fiscal_year, metrics, weights, thresholds, return_col):
    # Score = weight-scaled precomputed column, so the "optimizer" has a real
    # knob to turn without dragging in SummaryManager.
    w = float(weights.get("m", 1.0))
    return g["s"] * w, g[return_col]


def _greedy_optimizer(panel, metrics):
    """Miniature stand-in for optimize_panel_combo: sweeps one weight and
    reports every objective it evaluated."""
    trials = []
    best_w, best = 1.0, -np.inf
    for w in np.linspace(-2.0, 2.0, 21):
        obj = correlation._panel_avg_quintile_spread({"m": w}, panel, metrics)
        trials.append(obj)
        if obj > best:
            best, best_w = obj, w
    return {"optimized_weights": {"m": best_w}, "optimized_thresholds": {},
            "trial_objectives": trials}


@pytest.fixture(autouse=True)
def _patch_scoring(monkeypatch):
    monkeypatch.setattr(correlation, "_panel_year_scores", _fake_year_scores)
    monkeypatch.setattr(correlation, "_get_default_thresholds", lambda: {})


def test_shuffle_is_within_year_not_across():
    """Shuffling across years would leak one year's return level into another
    and quietly change each cross-section's own distribution."""
    panel = _panel()
    seen = []

    def capture(p, metrics):
        seen.append(p.groupby("fiscal_year")["fwd_excess_return_1y"]
                     .apply(lambda s: tuple(sorted(np.round(s, 12)))))
        return {"optimized_weights": {}, "optimized_thresholds": {},
                "trial_objectives": [0.0]}

    original = panel.groupby("fiscal_year")["fwd_excess_return_1y"] \
                    .apply(lambda s: tuple(sorted(np.round(s, 12))))
    permutation_benchmark(panel, ["m"], capture, n_permutations=3,
                          progress_every=0)
    for got in seen:
        pd.testing.assert_series_equal(got, original)


def test_target_actually_gets_permuted():
    panel = _panel()
    orders = []

    def capture(p, metrics):
        orders.append(tuple(np.round(p["fwd_excess_return_1y"], 12)))
        return {"optimized_weights": {}, "optimized_thresholds": {},
                "trial_objectives": [0.0]}

    permutation_benchmark(panel, ["m"], capture, n_permutations=5,
                          progress_every=0)
    assert len(set(orders)) > 1, "every permutation produced the same ordering"


def test_no_signal_does_not_clear_the_permutation_bar():
    """A panel with zero true signal must not look significant."""
    panel = _panel(signal=0.0, seed=1)
    observed = max(_greedy_optimizer(panel, ["m"])["trial_objectives"])
    null = permutation_benchmark(panel, ["m"], _greedy_optimizer,
                                 n_permutations=40, seed=7, progress_every=0)
    p = permutation_p_value(null["null_best"], observed)
    assert p > 0.05, f"pure noise looked significant (p={p:.3f})"
    assert observed <= null["p95"] * 1.5


def test_real_signal_clears_the_bar():
    panel = _panel(signal=1.5, seed=2)
    observed = max(_greedy_optimizer(panel, ["m"])["trial_objectives"])
    null = permutation_benchmark(panel, ["m"], _greedy_optimizer,
                                 n_permutations=40, seed=7, progress_every=0)
    p = permutation_p_value(null["null_best"], observed)
    assert p < 0.05, f"strong signal was missed (p={p:.3f})"


def test_p_value_never_reports_exactly_zero():
    # +1 correction: 200 draws cannot justify claiming p=0.
    assert permutation_p_value([0.0] * 200, 99.0) == pytest.approx(1 / 201)
    assert permutation_p_value([], 1.0) != 0.0


def test_zero_permutations_is_a_clean_skip():
    out = permutation_benchmark(_panel(), ["m"], _greedy_optimizer,
                                n_permutations=0)
    assert out["n_permutations"] == 0
    assert out["null_best"] == []
    assert out["sigma"] is None


def test_overrides_reach_the_dsr():
    base = deflated_sharpe_ratio([0.1, 0.2, 0.3], [0.05, 0.06, 0.07, 0.08])
    override = deflated_sharpe_ratio(
        [0.1, 0.2, 0.3], [0.05, 0.06, 0.07, 0.08],
        sigma_sr_override=0.5, sr_benchmark_override=0.9, n_trials_override=7,
    )
    assert override["sigma_sr"] == 0.5
    assert override["sr_benchmark"] == 0.9
    assert override["n_trials"] == 7
    assert override["dsr"] != base["dsr"]


def test_gate_reports_fold_wins_and_permutation(monkeypatch, capsys):
    monkeypatch.setattr(
        correlation, "leave_one_fiscal_year_out",
        lambda panel, metrics, optimizer_fn: [
            {"fiscal_year": 2022, "optimized_spread": 0.20, "optimized_ic": 0.1,
             "equal_spread": 0.10, "equal_ic": 0.05},
            {"fiscal_year": 2023, "optimized_spread": 0.05, "optimized_ic": 0.0,
             "equal_spread": 0.30, "equal_ic": 0.2},
        ],
    )
    res = correlation.gate_optimized_weights(
        _panel(), ["m"], optimizer_fn=_greedy_optimizer, n_permutations=5,
    )
    assert res["n_folds"] == 2
    assert res["n_folds_beating_equal"] == 1
    assert res["permutation"]["n_permutations"] == 5
    assert not np.isnan(res["permutation_p_value"])
    assert "beat equal weight in 1 of 2" in capsys.readouterr().out
