"""--evaluate: score variants, benchmarks and weighting (analyzer/evaluation.py)."""

import numpy as np
import pandas as pd
import pytest

from analyzer import benchmarks, evaluation
from analyzer.correlation import _all_scored_metrics, equal_metric_weights
from analyzer.evaluation import (
    basket, format_status, score_usefulness, weighting_backtest,
)


def _panel(n=40, years=(2021, 2022, 2023), seed=0, predictive=True):
    """Synthetic scored panel. With ``predictive`` the live key ranks the
    target perfectly; momentum is noise either way."""
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        blend = rng.permutation(np.linspace(0, 1, n))
        target = (blend - 0.5) * 0.2 if predictive else rng.normal(0, 0.1, n)
        for i in range(n):
            rows.append({
                "company_id": f"C{i}", "fiscal_year": y,
                "report_date": pd.Timestamp(f"{y + 1}-03-01"),
                "quality_pct": 0.5, "value_pct": 0.5,
                "blend_score": blend[i], "fundamental_pct": blend[i],
                "momentum_pct": rng.uniform(), "fwd_excess_return_1y": target[i],
                "fwd_total_return_1y": target[i] + 0.05,
            })
    return pd.DataFrame(rows)


def test_basket_applies_sleeve_gate_then_top_n():
    g = _panel(n=10, years=(2022,))
    g.loc[g["blend_score"].idxmax(), "quality_pct"] = 0.1  # best fails the gate
    b = basket(g, "blend_score", top_n=3)
    assert len(b) == 3
    assert g["blend_score"].max() not in set(b["blend_score"])


def test_predictive_score_is_recognised_and_noise_is_not():
    res = score_usefulness(_panel(), {})
    live = next(v for v in res["variants"] if v["key"] == "blend_score")
    mom = next(v for v in res["variants"] if v["key"] == "momentum_pct")
    assert live["mean_ic"] == pytest.approx(1.0)
    assert live["years_excess_positive"] == 3 and live["mean_excess"] > 0
    assert abs(mom["mean_ic"]) < 0.5


def test_index_comparison_matches_each_rows_own_window():
    df = _panel(n=30, years=(2022,))
    days = pd.bdate_range("2022-01-03", "2024-12-31")
    flat = pd.Series(100.0, index=days)  # index returns 0 -> vs index == total
    res = score_usefulness(df, {"IDX": flat})
    live = res["variants"][0]
    top = basket(df, "blend_score")
    assert live["vs"]["IDX"]["mean"] == pytest.approx(top["fwd_total_return_1y"].mean())
    uni = res["universe"]["vs"]["IDX"]
    assert uni["mean"] == pytest.approx(df["fwd_total_return_1y"].mean())


def test_partially_covered_index_reports_nothing_rather_than_a_subsample():
    df = _panel(n=30, years=(2022,))
    short = pd.Series(100.0, index=pd.bdate_range("2023-06-01", "2024-12-31"))
    res = score_usefulness(df, {"IDX": short})
    assert res["variants"][0]["vs"]["IDX"]["n_years"] == 0


def test_window_return():
    s = pd.Series([100.0, 110.0, 121.0],
                  index=pd.to_datetime(["2022-01-03", "2022-06-01", "2023-01-03"]))
    assert benchmarks.window_return(s, "2022-01-03", "2023-01-03") == pytest.approx(0.21)
    assert benchmarks.window_return(s, "2021-12-01", "2023-01-03") is None
    assert benchmarks.window_return(s, "2022-01-03", "2023-06-01") is None


def _closes(names, seed=1, start="2017-01-02", end="2024-12-31"):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    m = rng.normal(0, 0.006, len(days))
    return {n: pd.Series(100 * np.cumprod(1 + m + rng.normal(0, 0.008 + 0.002 * i, len(days))),
                         index=days)
            for i, n in enumerate(names)}


def test_weighting_runs_all_three_on_the_same_subset():
    df = _panel(n=30, years=(2022,))
    closes = _closes([f"C{i}" for i in range(30)])
    del closes["C29"]  # no verified prices -> excluded from all three
    out = weighting_backtest(df, closes, {})
    rec = out["per_year"][0]
    names = set(rec["equal"]["weights"])
    assert names == set(rec["minvar"]["weights"]) == set(rec["maxsharpe"]["weights"])
    assert "C29" not in names
    for key in ("equal", "minvar", "maxsharpe"):
        assert sum(rec[key]["weights"].values()) == pytest.approx(1.0)
    assert out["summary"]["minvar"]["years_beating_equal"] is not None


def test_covariance_never_sees_the_holding_window():
    """Prices on or after the basket's earliest report date must not move
    the weights -- they are the future the portfolio is being held into."""
    df = _panel(n=30, years=(2022,))
    names = [f"C{i}" for i in range(30)]
    base = _closes(names)
    decision = df["report_date"].min()
    shocked = {n: s.copy() for n, s in base.items()}
    rng = np.random.default_rng(9)
    for s in shocked.values():
        after = s.index >= decision
        s[after] = s[after] * np.cumprod(1 + rng.normal(0, 0.05, after.sum()))
    w0 = weighting_backtest(df, base, {})["per_year"][0]
    w1 = weighting_backtest(df, shocked, {})["per_year"][0]
    for key in ("minvar", "maxsharpe"):
        assert w0[key]["weights"] == pytest.approx(w1[key]["weights"])


def test_status_block_without_a_file_says_unknown():
    assert "unknown" in format_status(None)[0]


def test_live_and_panel_share_one_equal_weight_definition():
    w = equal_metric_weights()
    assert set(w) == set(_all_scored_metrics())
    assert set(w.values()) == {1.0}


def test_optimize_is_an_alias_for_evaluate():
    from analyzer.main import build_arg_parser
    assert build_arg_parser().parse_args(["--optimize"]).evaluate
    assert build_arg_parser().parse_args(["--evaluate"]).evaluate


def test_omxs30gi_cache_merges_and_fresh_values_win(tmp_path):
    def _ohlc(dates, closes):
        return {"ohlc": [{"timestamp": int(pd.Timestamp(d).value // 1_000_000),
                          "close": c} for d, c in zip(dates, closes)]}

    class _Av:
        def __init__(self, payload):
            self.payload = payload

        def get_chart_data(self, **_):
            return self.payload

    path = tmp_path / "OMXS30GI.csv"
    benchmarks.fetch_omxs30gi(_Av(_ohlc(["2021-11-01", "2021-11-02"], [500, 501])), path)
    merged = benchmarks.fetch_omxs30gi(
        _Av(_ohlc(["2021-11-02", "2021-11-03"], [502, 503])), path)
    assert list(merged.index.strftime("%Y-%m-%d")) == ["2021-11-01", "2021-11-02", "2021-11-03"]
    assert list(merged) == [500, 502, 503]


def test_core_aware_variants_need_core_history_and_keep_the_same_names():
    df = _panel(n=30, years=(2022,))
    names = [f"C{i}" for i in range(30)]
    closes = _closes(names)
    days = closes["C0"].index
    core = pd.Series(100 * np.cumprod(1 + np.random.default_rng(3).normal(0, 0.007, len(days))),
                     index=days)
    rec = weighting_backtest(df, closes, {}, core_close=core)["per_year"][0]
    assert set(rec["minvar_core"]["weights"]) == set(rec["minvar"]["weights"])
    # a core that starts after the decision date cannot be used
    late = core[core.index > df["report_date"].min()]
    rec2 = weighting_backtest(df, closes, {}, core_close=late)["per_year"][0]
    assert "minvar_core" not in rec2
    assert rec2["minvar"]["weights"] == pytest.approx(rec["minvar"]["weights"])
