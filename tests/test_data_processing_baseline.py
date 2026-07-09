"""Baseline regression test for enrich_ratios' den_floor clamp
(analyzer/data_processing.py).

RATIO_SPECS is monkeypatched with a synthetic spec so this test does not
depend on real (gitignored) metrics.py threshold/den_floor values.
"""
import math

import pandas as pd

import analyzer.data_processing as data_processing
from analyzer.data_processing import enrich_ratios


def _patch_specs(monkeypatch, spec):
    monkeypatch.setattr(data_processing, "RATIO_SPECS", {"test_ratio status": spec})


def test_enrich_ratios_den_floor_clamps_small_positive_denominator(monkeypatch):
    spec = {"num": "num_col", "den": "den_col", "dir": 1, "num_is_rate": False, "den_floor": 0.3}
    _patch_specs(monkeypatch, spec)

    df = pd.DataFrame({"num_col": [1.0], "den_col": [0.1]})
    out = enrich_ratios(df)

    expected = 1.0 / 0.3
    assert math.isclose(out["test_ratio status"].iloc[0], expected, abs_tol=1e-9)


def test_enrich_ratios_den_floor_clamps_small_negative_denominator(monkeypatch):
    spec = {"num": "num_col", "den": "den_col", "dir": 1, "num_is_rate": False, "den_floor": 0.3}
    _patch_specs(monkeypatch, spec)

    df = pd.DataFrame({"num_col": [1.0], "den_col": [-0.1]})
    out = enrich_ratios(df)

    expected = 1.0 / -0.3
    assert math.isclose(out["test_ratio status"].iloc[0], expected, abs_tol=1e-9)


def test_enrich_ratios_den_floor_leaves_large_denominator_unclamped(monkeypatch):
    spec = {"num": "num_col", "den": "den_col", "dir": 1, "num_is_rate": False, "den_floor": 0.3}
    _patch_specs(monkeypatch, spec)

    df = pd.DataFrame({"num_col": [1.0], "den_col": [1.0]})
    out = enrich_ratios(df)

    assert math.isclose(out["test_ratio status"].iloc[0], 1.0, abs_tol=1e-9)


def test_enrich_ratios_missing_columns_yields_none(monkeypatch):
    spec = {"num": "missing_num", "den": "missing_den", "dir": 1, "num_is_rate": False}
    _patch_specs(monkeypatch, spec)

    df = pd.DataFrame({"other": [1.0]})
    out = enrich_ratios(df)
    assert out["test_ratio status"].iloc[0] is None
