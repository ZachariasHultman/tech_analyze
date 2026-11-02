"""Share classes of one issuer count as one company in every pushed list."""

import pandas as pd
import pytest

import analyzer.main as main
from analyzer.summary_manager import SummaryManager, company_key, one_per_company


@pytest.mark.parametrize("name, key", [
    ("Handelsbanken A 5264", "handelsbanken"),
    ("Handelsbanken B 5265", "handelsbanken"),
    ("SEB C 123", "seb"),
    ("SSAB B 220", "ssab"),
    ("Autoliv SDB 5236", "autoliv"),
    ("Shell ADR 1339643", "shell"),
    ("Alphabet Class C 4", "alphabet"),
    ("Volvo ser. B 5", "volvo"),
    ("Investor Pref 6", "investor"),
    ("A.P. Møller - Mærsk B  52316", "a.p. møller - mærsk"),
    ("3M 7", "3m"),
    ("Coca-Cola 3803", "coca-cola"),
])
def test_company_key(name, key):
    assert company_key(name) == key


def test_keeps_the_best_ranked_class_only():
    ranked = pd.DataFrame({"s": [0.9, 0.8, 0.7]},
                          index=["Handelsbanken B 2", "Handelsbanken A 1", "Volvo B 3"])
    assert list(one_per_company(ranked).index) == ["Handelsbanken B 2", "Volvo B 3"]


def test_dropped_class_frees_its_slot_for_the_next_company():
    m = SummaryManager()
    m.summary = pd.DataFrame(
        {"blend_score": [0.9, 0.8, 0.7], "points": [1.0, 1.0, 1.0],
         "quality_pct": [0.5] * 3, "value_pct": [0.5] * 3},
        index=["SSAB A 1", "SSAB B 2", "Volvo B 3"])
    _, qualified = main._gate_and_rank(m, top_n=2)
    assert list(qualified.index) == ["SSAB A 1", "Volvo B 3"]


def test_backtest_basket_applies_the_same_rule():
    from analyzer.evaluation import basket
    g = pd.DataFrame({"company_id": ["SSAB A 1", "SSAB B 2", "Volvo B 3"],
                      "blend_score": [0.9, 0.8, 0.7],
                      "quality_pct": [0.5] * 3, "value_pct": [0.5] * 3})
    assert list(basket(g, "blend_score", top_n=2)["company_id"]) == ["SSAB A 1", "Volvo B 3"]
