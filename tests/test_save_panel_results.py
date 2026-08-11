"""Tests for save_panel_optimization_results: persists gate_optimized_weights'
verdict to a JSON file that main.py's _load_optimized_params("panel") (and
the new default-preference chain) can load. Round-trip shape must exactly
match what _load_optimized_params expects: top-level "optimized_weights" /
"optimized_thresholds" keys.
"""
import json

import analyzer.correlation as correlation


def _fake_gate_result(accept=True):
    return {
        "accept": accept,
        "beats_equal_weight": accept,
        "dsr_significant": accept,
        "confidence": 0.925,
        "mean_optimized_spread": 0.1284,
        "mean_equal_spread": 0.0936,
        "dsr": {"dsr": 0.9387, "n_trials": 264, "sr_benchmark": 0.0007},
        "folds": [],
        "chosen_weights": {"roe_pe ratio status": 2.0} if accept else {"roe_pe ratio status": 1.0},
        "chosen_thresholds": {"roe_pe ratio status": {"nok": 0.2, "ok": 0.7}},
    }


def test_save_panel_results_round_trips_expected_shape(tmp_path):
    out_path = tmp_path / "optimization_results_panel.json"
    correlation.save_panel_optimization_results(_fake_gate_result(accept=True), out_path=str(out_path))

    with open(out_path) as f:
        data = json.load(f)

    # Exact shape _load_optimized_params (main.py) reads.
    assert data["optimized_weights"] == {"roe_pe ratio status": 2.0}
    assert data["optimized_thresholds"] == {"roe_pe ratio status": {"nok": 0.2, "ok": 0.7}}
    assert data["accepted"] is True
    assert data["confidence"] == 0.925
    assert data["dsr"] == 0.9387


def test_save_panel_results_writes_on_reject_too(tmp_path):
    # A reject's chosen_weights is already the gate's own equal-weight
    # fallback -- the file must still be written, not skipped, so
    # --use-panel after a rejected run means "use the honest equal-weight
    # recommendation," not "no file exists."
    out_path = tmp_path / "optimization_results_panel.json"
    correlation.save_panel_optimization_results(_fake_gate_result(accept=False), out_path=str(out_path))

    with open(out_path) as f:
        data = json.load(f)

    assert data["accepted"] is False
    assert data["optimized_weights"] == {"roe_pe ratio status": 1.0}
