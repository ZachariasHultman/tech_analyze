"""The too-few-samples coverage warning must print once per distinct message.

Coverage of a given cross-section cannot change between calls, but --optimize
re-scores the same cross-sections once per candidate (folds x permutations x
coordinate-descent steps), so the undeduped line printed the same fact
hundreds of times and pushed the gate's actual verdict off the screen.

Deduping on the message text, not on a "warned once" flag: a genuinely
different cross-section (different year, different metric set) is new
information and must still be reported.
"""
import pandas as pd
import pytest

from analyzer.data_processing import calculate_score, reset_coverage_warnings
from analyzer.summary_manager import SummaryManager


@pytest.fixture(autouse=True)
def _clean_warning_state():
    reset_coverage_warnings()
    yield
    reset_coverage_warnings()


def _manager(values, metric="test_metric"):
    m = SummaryManager()
    m._weight_overrides = {metric: 1.0}
    m.template = [metric]
    m.template_investment = []
    m.summary_investment = {}
    m.summary = pd.DataFrame(
        {metric: values}, index=[f"Company {i}" for i in range(len(values))]
    )
    return m


def _thin(n=10, n_real=2):
    """n companies but only n_real real values -- below max(3, 0.6n)."""
    return [float(i) if i < n_real else None for i in range(n)]


def test_identical_cross_section_warns_once(capsys):
    for _ in range(5):
        calculate_score(_manager(_thin()), metrics_to_score=["test_metric"])
    out = capsys.readouterr().out
    assert out.count("too few samples") == 1


def test_a_different_cross_section_still_warns(capsys):
    calculate_score(_manager(_thin(n=10)), metrics_to_score=["test_metric"])
    calculate_score(_manager(_thin(n=12)), metrics_to_score=["test_metric"])
    out = capsys.readouterr().out
    # different denominators -> different message -> both are real news
    assert out.count("too few samples") == 2


def test_reset_restores_reporting(capsys):
    calculate_score(_manager(_thin()), metrics_to_score=["test_metric"])
    reset_coverage_warnings()
    calculate_score(_manager(_thin()), metrics_to_score=["test_metric"])
    assert capsys.readouterr().out.count("too few samples") == 2


def test_dedupe_does_not_change_scores():
    """Suppressing a log line must not touch the fallback it describes."""
    first = _manager(_thin())
    calculate_score(first, metrics_to_score=["test_metric"])
    second = _manager(_thin())
    calculate_score(second, metrics_to_score=["test_metric"])
    pd.testing.assert_series_equal(
        first.summary["test_metric_score"], second.summary["test_metric_score"]
    )
