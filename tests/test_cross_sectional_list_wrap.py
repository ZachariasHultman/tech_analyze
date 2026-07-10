"""Regression test: the cross-sectional min-sample coverage check in
calculate_score must unwrap single-element list/tuple values before calling
pd.to_numeric, the same way _assign_points already does.

Historical re-scoring (process_historical) stores metric values as
single-element lists, e.g. [0.82]. Before this fix, pd.to_numeric couldn't
parse that shape, so the coverage check always saw 0 non-null values for
every historically-rescored metric -- always falling back to absolute
thresholds regardless of how much real data existed, and regardless of the
min-sample threshold chosen.
"""
import pandas as pd

from analyzer.data_processing import calculate_score
from analyzer.summary_manager import SummaryManager


def test_list_wrapped_values_count_toward_cross_sectional_coverage(capsys):
    m = SummaryManager()
    m._weight_overrides = {"test_metric": 1.0}
    m.template = ["test_metric"]
    m.template_investment = []
    m.summary_investment = {}

    # 10 companies, all list-wrapped -- matches process_historical's shape.
    # min_samples for n=10 is max(3, ceil(0.6*10)) = 6, well below 10.
    m.summary = pd.DataFrame(
        {"test_metric": [[float(i)] for i in range(10)]},
        index=[f"Company {i}" for i in range(10)],
    )

    calculate_score(m, metrics_to_score=["test_metric"])

    captured = capsys.readouterr()
    assert "test_metric" not in captured.out, (
        "list-wrapped values should count toward coverage and not trigger "
        "the too-few-samples fallback"
    )
    # Cross-sectional ranking actually ran: scores should vary by rank,
    # not all collapse to the same absolute-threshold value.
    assert m.summary["test_metric_score"].nunique() > 1
