"""The email must carry the evidence for the weights it is recommending.

Before this, the email listed a tidy top-10 and said nothing about whether the
system was earning trust: not whether the challenger gate accepted or fell back
to equal weight, not how the score did out of sample, not that n=4 fiscal years
supports no significance claim at all. A clean list implies confidence the data
cannot back, so the status block leads the email on every run.

It must also survive a Pi holding an older optimization_results_panel.json --
the weekly cron run cannot die because provenance metadata is missing.
"""

import json

import numpy as np
import pytest

from analyzer.correlation import build_validation_summary
from analyzer.main import _format_optimizer_status, _load_optimizer_status


def _gate_result(accept=True):
    return {
        "accept": accept,
        "n_companies": 127,
        "n_folds_beating_equal": 4,
        "n_folds": 4,
        "permutation_p_value": 0.02,
        "permutation": {"n_permutations": 200},
        "folds": [
            {"fiscal_year": 2023, "optimized_ic": 0.081, "optimized_spread": 0.02,
             "equal_ic": 0.060, "equal_spread": 0.01},
            {"fiscal_year": 2022, "optimized_ic": -0.050, "optimized_spread": -0.05,
             "equal_ic": -0.021, "equal_spread": -0.07},
            {"fiscal_year": 2025, "optimized_ic": 0.136, "optimized_spread": 0.08,
             "equal_ic": 0.096, "equal_spread": 0.06},
            {"fiscal_year": 2024, "optimized_ic": 0.119, "optimized_spread": 0.10,
             "equal_ic": 0.085, "equal_spread": 0.09},
        ],
    }


# ---------------------------------------------------------- summary building
def test_summary_is_sorted_by_year_and_counts_periods():
    s = build_validation_summary(_gate_result())
    assert [r["fiscal_year"] for r in s["per_year"]] == [2022, 2023, 2024, 2025]
    assert s["n_periods"] == 4
    assert s["n_companies"] == 127


def test_summary_reports_the_weights_actually_chosen():
    """On reject the system runs equal weight, so the reported evidence must
    describe equal weight -- not the challenger that was turned down."""
    accepted = build_validation_summary(_gate_result(accept=True))
    rejected = build_validation_summary(_gate_result(accept=False))
    assert accepted["mean_ic"] == pytest.approx(np.mean([-0.050, 0.081, 0.119, 0.136]))
    assert rejected["mean_ic"] == pytest.approx(np.mean([-0.021, 0.060, 0.085, 0.096]))
    assert rejected["mean_ic"] != accepted["mean_ic"]


def test_summary_carries_significance_and_is_json_safe():
    s = build_validation_summary(_gate_result())
    assert s["t_stat"] is not None and s["p_value"] is not None
    assert s["p_value"] > 0.05, "4 periods should not look significant"
    json.dumps(s)  # must round-trip into the weights file


def test_nan_fold_values_do_not_leak_into_the_summary():
    gate = _gate_result()
    gate["folds"][0]["optimized_ic"] = float("nan")
    s = build_validation_summary(gate)
    assert s["per_year"][1]["ic"] is None
    assert not np.isnan(s["mean_ic"])
    json.dumps(s)


# ---------------------------------------------------------------- rendering
def test_block_states_verdict_confidence_and_universe():
    block = "\n".join(_format_optimizer_status(
        {"accepted": True, **build_validation_summary(_gate_result())}))
    assert "ACCEPTED" in block
    assert "CONFIDENCE: LOW" in block
    assert "4 periods only" in block
    assert "beat equal weight in 4 of 4" in block
    assert "127 stocks" in block
    assert "Survivorship bias" in block
    assert "permutation test (200 refits" in block
    assert "total return" in block


def test_block_says_when_the_gate_rejected():
    block = "\n".join(_format_optimizer_status(
        {"accepted": False, **build_validation_summary(_gate_result(accept=False))}))
    assert "REJECTED" in block
    assert "equal weight" in block
    # On a reject the IC rows describe equal weight (what runs) while the
    # fold count describes the challenger, so the two must be distinguishable.
    assert "challenger beat equal weight in 4 of 4" in block
    assert "missed the significance bar" in block


def test_block_flags_stale_weights():
    s = build_validation_summary(_gate_result())
    s["fitted_at"] = "2020-01-01T00:00:00"
    block = "\n".join(_format_optimizer_status({"accepted": True, **s}))
    assert "STALE" in block


def test_missing_metadata_degrades_instead_of_crashing():
    block = "\n".join(_format_optimizer_status(None))
    assert "unknown" in block
    assert "optimization_results_panel.json" in block


def test_legacy_file_without_validation_block_is_treated_as_unknown(tmp_path,
                                                                    monkeypatch):
    from analyzer import main as main_mod

    legacy = {"optimized_weights": {"m": 1.0}, "optimized_thresholds": {},
              "accepted": True}
    (tmp_path / "optimization_results_panel.json").write_text(json.dumps(legacy))
    monkeypatch.setattr(main_mod, "project_root", str(tmp_path))
    assert _load_optimizer_status() is None
    # and the renderer copes with that
    assert "unknown" in "\n".join(_format_optimizer_status(None))


def test_status_round_trips_through_the_weights_file(tmp_path, monkeypatch):
    from analyzer import main as main_mod

    payload = {
        "optimized_weights": {"m": 1.0},
        "optimized_thresholds": {},
        "accepted": True,
        "dsr": 0.97,
        "confidence": 0.925,
        "validation": build_validation_summary(_gate_result()),
    }
    (tmp_path / "optimization_results_panel.json").write_text(json.dumps(payload))
    monkeypatch.setattr(main_mod, "project_root", str(tmp_path))

    status = _load_optimizer_status()
    assert status is not None
    assert status["accepted"] is True
    assert status["n_periods"] == 4
    block = "\n".join(_format_optimizer_status(status))
    assert "ACCEPTED" in block and "mean IC" in block


def test_confidence_verdict_is_derived_not_hardcoded():
    """It used to always print LOW, which contradicted itself the moment the
    IC test cleared 5% ("CONFIDENCE: LOW ... p=0.02")."""
    s = build_validation_summary(_gate_result())
    s.update({"t_stat": 3.05, "p_value": 0.02, "mean_spread": 0.06})
    block = "\n".join(_format_optimizer_status({"accepted": True, **s}))
    assert "CONFIDENCE: MODERATE" in block
    assert "CONFIDENCE: LOW" not in block

    s.update({"t_stat": 1.20, "p_value": 0.29})
    weak = "\n".join(_format_optimizer_status({"accepted": True, **s}))
    assert "CONFIDENCE: LOW" in weak


def test_flat_spread_is_called_out_even_when_ic_is_significant():
    """A significant IC and a usable top-N pick are different claims."""
    s = build_validation_summary(_gate_result())
    s.update({"t_stat": 3.05, "p_value": 0.02, "mean_spread": 0.001})
    block = "\n".join(_format_optimizer_status({"accepted": True, **s}))
    assert "CONFIDENCE: MODERATE" in block
    assert "CAVEAT" in block
    assert "extremes do not separate" in block


def test_survivorship_is_always_stated():
    for p in (0.02, 0.40):
        s = build_validation_summary(_gate_result())
        s.update({"t_stat": 2.0, "p_value": p, "mean_spread": 0.05})
        block = "\n".join(_format_optimizer_status({"accepted": True, **s}))
        assert "Survivorship bias" in block


# --------------------------------------------------------------------------
# Deflated Sharpe Ratio: the number the accept/reject actually turns on
# --------------------------------------------------------------------------

def _status(dsr=0.73, confidence=0.925, accepted=False):
    """Minimal status dict -- _format_optimizer_status reads a flat mapping
    of the JSON's top level merged with its `validation` block."""
    return {
        "accepted": accepted, "dsr": dsr, "confidence": confidence,
        "fitted_at": "2026-08-14", "n_periods": 7, "n_folds": 7,
        "n_folds_beating_equal": 4, "permutation_p_value": 0.31,
        "n_permutations": 200, "mean_ic": 0.044, "t_stat": 2.55,
        "p_value": 0.043, "per_year": [],
    }


def test_dsr_and_bar_are_both_shown():
    """'missed the significance bar' is an unquantified assertion on its own:
    0.91 against a 0.925 bar and 0.53 against it are very different rejects."""
    out = "\n".join(_format_optimizer_status(_status(dsr=0.712)))
    assert "0.712" in out and "0.925" in out


def test_reject_is_labelled_missed():
    out = "\n".join(_format_optimizer_status(_status(dsr=0.50, accepted=False)))
    assert "missed" in out


def test_accept_is_labelled_cleared():
    out = "\n".join(_format_optimizer_status(_status(dsr=0.97, accepted=True)))
    assert "cleared" in out


@pytest.mark.parametrize("dsr,confidence", [
    (None, 0.925), (0.73, None), (float("nan"), 0.925), ("n/a", 0.925),
])
def test_missing_or_junk_dsr_omits_the_line_without_crashing(dsr, confidence):
    """A Pi holding an older JSON must still render the rest of the block."""
    out = "\n".join(_format_optimizer_status(_status(dsr=dsr, confidence=confidence)))
    assert "deflated Sharpe" not in out
    assert "SYSTEM STATUS" in out
