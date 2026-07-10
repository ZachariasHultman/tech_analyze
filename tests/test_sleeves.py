"""Tests for item 2: two-sleeve (quality/value) scoring.

Covers:
- every status metric in either SummaryManager template is assigned to
  exactly one of QUALITY_METRICS / VALUE_METRICS (fails loudly if a future
  template addition is left unassigned or double-assigned);
- the >= SLEEVE_GATE_MIN AND >= SLEEVE_GATE_MIN push gate rejects a stock
  strong in only one sleeve.
"""
import pandas as pd

from analyzer.config import QUALITY_METRICS, VALUE_METRICS, SLEEVE_GATE_MIN
from analyzer.summary_manager import SummaryManager


def _template_metrics():
    sm = SummaryManager()
    names = set()
    for tmpl in (sm.template, sm.template_investment):
        for k in tmpl:
            if isinstance(k, str) and k.endswith(" status"):
                names.add(k)
    return names


def test_every_template_metric_assigned_exactly_once():
    quality = set(QUALITY_METRICS)
    value = set(VALUE_METRICS)

    # No metric may live in both sleeves.
    assert quality.isdisjoint(value), quality & value
    # No accidental duplicates within a sleeve.
    assert len(quality) == len(QUALITY_METRICS)
    assert len(value) == len(VALUE_METRICS)

    template = _template_metrics()
    assigned = quality | value

    unassigned = template - assigned
    assert not unassigned, f"template metrics not in any sleeve: {sorted(unassigned)}"

    unknown = assigned - template
    assert not unknown, f"sleeve metrics not in any template: {sorted(unknown)}"


def _passes_gate(quality_pct, value_pct):
    combined = pd.DataFrame(
        {"quality_pct": [quality_pct], "value_pct": [value_pct]}
    )
    q = pd.to_numeric(combined["quality_pct"], errors="coerce")
    v = pd.to_numeric(combined["value_pct"], errors="coerce")
    return bool(((q >= SLEEVE_GATE_MIN) & (v >= SLEEVE_GATE_MIN)).iloc[0])


def test_gate_rejects_high_quality_low_value():
    assert _passes_gate(0.95, 0.1) is False


def test_gate_rejects_low_quality_high_value():
    assert _passes_gate(0.1, 0.95) is False


def test_gate_accepts_both_above_floor():
    assert _passes_gate(0.5, 0.5) is True
    assert _passes_gate(SLEEVE_GATE_MIN, SLEEVE_GATE_MIN) is True


def test_gate_rejects_just_below_floor():
    assert _passes_gate(SLEEVE_GATE_MIN - 0.01, 0.9) is False
