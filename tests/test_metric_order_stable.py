"""_all_scored_metrics() must return a stable order across processes.

It used to be `list(set(...))`. Python randomizes string hashing per process
(PYTHONHASHSEED), so the returned order differed on every run. The coordinate
descent in optimize_panel_combo / optimize_combo iterates `for m in metrics`,
so a different order walks a different greedy path and lands on different
weights -- from identical input data. Observed in practice: two runs of
leave_one_fiscal_year_out on the same panel produced mean optimized spreads of
0.1315 and 0.0977, while the (order-independent) equal-weight baseline matched
to 15 decimals.

Those weights are what live scoring loads from optimization_results_panel.json,
so the nondeterminism reached production. This test pins the fix.
"""

import os
import subprocess
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_SNIPPET = (
    "from analyzer.correlation import _all_scored_metrics; "
    "print(chr(10).join(_all_scored_metrics()))"
)


def _metrics_with_hashseed(seed):
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    out = subprocess.run(
        [sys.executable, "-c", _SNIPPET],
        cwd=_ROOT, env=env, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip().splitlines()


def test_metric_order_identical_across_hash_seeds():
    runs = [_metrics_with_hashseed(seed) for seed in (0, 1, 12345)]
    assert runs[0], "expected a non-empty metric list"
    for other in runs[1:]:
        assert other == runs[0]


def test_metric_order_is_sorted():
    from analyzer.correlation import _all_scored_metrics

    metrics = _all_scored_metrics()
    assert metrics == sorted(metrics)
    assert len(metrics) == len(set(metrics)), "expected no duplicates"
