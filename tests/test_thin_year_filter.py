"""Fiscal years with a tiny cross-section must not enter the panel evidence.

Real case that motivated this: fiscal year 2021 had 9 companies with a forward
return (the live OHLC window starts mid-2021, so only off-cycle reporters got a
price anchor). _quintile_spread falls back to terciles below n=25, so that year
was scored as 3 stocks vs 3 stocks -- and produced IC=+0.867, spread=+41.6%.
Averaged in as one of five equally-weighted years, it moved the reported mean
IC from +0.038 to +0.203 and the pooled spread from +1.1% to +9.2%.

It also distorted the challenger gate: 2021 was the single fold where equal
weight beat the optimized weights, turning a clean 4-of-4 into an ambiguous
3-of-5.
"""

import numpy as np
import pandas as pd
import pytest

from analyzer.config import MIN_CROSS_SECTION
from analyzer.panel import drop_thin_years


def _panel(year_sizes, return_col="fwd_excess_return_1y"):
    rows = []
    for fy, n in year_sizes.items():
        for i in range(n):
            rows.append({
                "company_id": f"co{i}",
                "company": f"co{i}",
                "fiscal_year": fy,
                return_col: 0.01 * i,
                "composite_score_equal": float(i),
            })
    return pd.DataFrame(rows)


def test_min_cross_section_matches_quintile_switchover():
    # _quintile_spread uses quintiles at n >= 25, terciles below. The floor
    # deliberately reuses that number rather than inventing a second one.
    from analyzer.correlation import _quintile_spread

    scores = pd.Series(range(MIN_CROSS_SECTION), dtype=float)
    returns = pd.Series(np.linspace(0, 1, MIN_CROSS_SECTION))
    assert _quintile_spread(scores, returns) is not None


def test_thin_year_dropped_dense_years_kept():
    df = _panel({2021: 9, 2022: 40, 2023: 40})
    out = drop_thin_years(df, "fwd_excess_return_1y")
    assert sorted(out["fiscal_year"].unique()) == [2022, 2023]
    assert len(out) == 80


def test_year_exactly_at_floor_is_kept():
    df = _panel({2021: MIN_CROSS_SECTION, 2022: MIN_CROSS_SECTION - 1})
    out = drop_thin_years(df, "fwd_excess_return_1y")
    assert sorted(out["fiscal_year"].unique()) == [2021]


def test_rows_without_a_return_do_not_count_toward_the_floor():
    # FY2026 rows exist (current cross-section) but have no forward return
    # yet. They must not prop a year up over the floor.
    df = _panel({2024: 40})
    thin = _panel({2026: 40})
    thin["fwd_excess_return_1y"] = np.nan
    out = drop_thin_years(pd.concat([df, thin], ignore_index=True),
                          "fwd_excess_return_1y")
    assert sorted(out["fiscal_year"].unique()) == [2024]


def test_empty_and_missing_column_are_safe():
    assert drop_thin_years(pd.DataFrame(), "fwd_excess_return_1y").empty
    df = _panel({2024: 40}).drop(columns=["fwd_excess_return_1y"])
    # No target column -> nothing to judge; pass the frame through untouched
    # rather than silently emptying the pipeline.
    assert len(drop_thin_years(df, "fwd_excess_return_1y")) == 40


def test_validation_battery_excludes_thin_years(tmp_path, capsys):
    from analyzer.validation import run_validation_battery

    rng = np.random.default_rng(0)
    rows = []
    for fy, n in {2021: 9, 2022: 40, 2023: 40, 2024: 40}.items():
        for i in range(n):
            rows.append({
                "company_id": f"co{i}", "fiscal_year": fy,
                "report_date": f"{fy}-02-01", "sector": "X",
                "composite_score_equal": float(i),
                "fwd_return_1y": float(rng.normal()),
                "universe_mean_return_that_year": 0.0,
                "fwd_excess_return_1y": float(rng.normal()),
            })
    scores = tmp_path / "panel_scores.csv"
    pd.DataFrame(rows).to_csv(scores, index=False)

    res = run_validation_battery(str(scores), str(tmp_path / "missing.csv"))

    assert res["header"]["n_periods"] == 3
    assert 2021 not in [r["fiscal_year"] for r in res["quintiles"]["per_year"]]
    assert 2021 not in [fy for fy, _, _ in res["ic"]["per_year"]]
    assert "2021" in capsys.readouterr().out, "dropped years should be reported"


def test_biased_subsample_year_is_dropped_despite_passing_size():
    """The partial-backfill trap: FY2019 arrived with 27 companies -- over the
    size floor -- but all 27 were large caps from a 30-symbol partial Yahoo
    backfill, against 87 companies with fundamentals that year."""
    rows = []
    for i in range(87):                      # FY2019: fundamentals for 87...
        rows.append({"company_id": f"co{i}", "company": f"co{i}",
                     "fiscal_year": 2019, "composite_score_equal": float(i),
                     "fwd_excess_return_1y": 0.01 * i if i < 27 else np.nan})
    for i in range(126):                     # FY2022: full coverage
        rows.append({"company_id": f"co{i}", "company": f"co{i}",
                     "fiscal_year": 2022, "composite_score_equal": float(i),
                     "fwd_excess_return_1y": 0.01 * i})
    out = drop_thin_years(pd.DataFrame(rows), "fwd_excess_return_1y")
    assert sorted(out["fiscal_year"].unique()) == [2022]


def test_full_coverage_year_above_the_floor_is_kept():
    rows = [{"company_id": f"co{i}", "company": f"co{i}", "fiscal_year": 2020,
             "composite_score_equal": float(i),
             "fwd_excess_return_1y": 0.01 * i if i < 80 else np.nan}
            for i in range(100)]
    out = drop_thin_years(pd.DataFrame(rows), "fwd_excess_return_1y")
    assert sorted(out["fiscal_year"].unique()) == [2020]  # 80% coverage


def test_coverage_reason_is_reported(capsys):
    rows = [{"company_id": f"co{i}", "company": f"co{i}", "fiscal_year": 2019,
             "composite_score_equal": float(i),
             "fwd_excess_return_1y": 0.01 * i if i < 30 else np.nan}
            for i in range(100)]
    drop_thin_years(pd.DataFrame(rows), "fwd_excess_return_1y")
    assert "biased subsample" in capsys.readouterr().out
