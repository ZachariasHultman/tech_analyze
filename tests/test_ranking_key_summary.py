"""The email must report the key the watchlist is ranked on, not just the gate's.

The gate's IC/spread describe its own objective, which is built from the
`points` composite. The watchlist is ordered by `blend_score`. Printing only
the gate's figures would put numbers in the email that measure a different
ordering than the list above them.
"""
import numpy as np
import pandas as pd
import pytest

from analyzer.correlation import blend_key_summary
from analyzer.main import _format_optimizer_status


def _panel(n_years=5, n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for y in range(2019, 2019 + n_years):
        blend = rng.uniform(0, 1, n)
        # target correlated with the key, so IC is positive by construction
        target = 0.3 * (blend - 0.5) + rng.normal(0, 0.1, n)
        for i in range(n):
            rows.append({"company_id": f"C{i:03d}", "fiscal_year": y,
                         "blend_score": blend[i],
                         "fwd_excess_return_1y": target[i]})
    return pd.DataFrame(rows)


def test_summary_reports_positive_ic_for_a_predictive_key():
    s = blend_key_summary(_panel())
    assert s is not None
    assert s["mean_ic"] > 0.2
    assert s["years_ic_positive"] == s["n_periods"] == 5
    assert s["mean_top_n_excess"] > 0


def test_summary_is_none_without_the_column(tmp_path):
    """`scores_path` points nowhere so the on-disk fallback can't rescue it --
    this asserts the no-blend case, not the fallback."""
    missing = str(tmp_path / "nope.csv")
    df = _panel().drop(columns=["blend_score"])
    assert blend_key_summary(df, scores_path=missing) is None


def test_summary_is_none_on_an_empty_panel(tmp_path):
    missing = str(tmp_path / "nope.csv")
    assert blend_key_summary(pd.DataFrame(), scores_path=missing) is None
    assert blend_key_summary(None, scores_path=missing) is None


def test_summary_skips_thin_years():
    """A cross-section below MIN_CROSS_SECTION is not evidence -- the same
    floor drop_thin_years applies, for the same reason."""
    df = _panel(n_years=3, n=60)
    thin = _panel(n_years=1, n=5, seed=9)
    thin["fiscal_year"] = 2030
    s = blend_key_summary(pd.concat([df, thin], ignore_index=True))
    assert 2030 not in [r["fiscal_year"] for r in s["per_year"]]


def test_status_block_renders_the_ranking_key_section():
    status = {
        "accepted": False, "dsr": 0.4, "confidence": 0.925,
        "n_periods": 5, "n_companies": 127,
        "per_year": [{"fiscal_year": 2019, "ic": 0.05, "spread": 0.03}],
        "mean_ic": 0.05, "mean_spread": 0.07,
        "t_stat": 2.0, "p_value": 0.04,
        "ranking_key": {
            "key": "blend_score", "top_n": 25, "n_periods": 7,
            "mean_ic": 0.108, "ic_p_value": 0.047, "years_ic_positive": 7,
            "mean_top_n_excess": 0.0635, "top_n_p_value": 0.086,
            "years_top_n_positive": 6,
        },
    }
    out = "\n".join(_format_optimizer_status(status))
    assert "RANKING KEY" in out
    assert "blend_score" in out
    assert "+0.108" in out
    assert "positive 7/7" in out
    assert "+6.35%/yr" in out
    assert "survivorship" in out.lower()


def test_status_block_omits_the_section_for_an_older_file():
    """A weights file SCP'd before the blend existed must still render."""
    status = {
        "accepted": False, "n_periods": 5, "per_year": [],
        "mean_ic": 0.05, "t_stat": 2.0, "p_value": 0.04,
        "ranking_key": None,
    }
    out = "\n".join(_format_optimizer_status(status))
    assert "RANKING KEY" not in out
    assert "CONFIDENCE" in out


def test_summary_reads_the_scores_panel_not_the_gate_panel(tmp_path):
    """Regression: blend_key_summary was first wired to load_gate_panel's
    output, which merges the *fundamentals* panel with only
    fwd_excess_return_1y from the scored one -- so blend_score was absent and
    the block was silently omitted from every weights file."""
    scores = tmp_path / "panel_scores.csv"
    _panel().to_csv(scores, index=False)
    s = blend_key_summary(panel_df=None, scores_path=str(scores))
    assert s is not None and s["mean_ic"] > 0.2


def test_summary_ignores_a_panel_missing_the_key_and_falls_back(tmp_path):
    """A gate panel (no blend_score) must not silently yield None when the
    scores panel on disk does have it."""
    scores = tmp_path / "panel_scores.csv"
    _panel().to_csv(scores, index=False)
    gate_like = _panel().drop(columns=["blend_score"])
    s = blend_key_summary(panel_df=gate_like, scores_path=str(scores))
    assert s is not None and s["n_periods"] == 5


def test_summary_returns_none_when_no_panel_exists(tmp_path):
    assert blend_key_summary(panel_df=None,
                             scores_path=str(tmp_path / "nope.csv")) is None


def test_confidence_verdict_judges_the_ranking_key_not_the_gate():
    """The verdict must describe what the system runs. Judging the gate's
    composite while printing the ranking key's p-value below it produced a
    self-contradicting block: "CONFIDENCE: LOW ... p=0.08" over "IC p=0.047".
    """
    status = {
        "accepted": False, "n_periods": 7, "per_year": [],
        # gate composite misses 5%...
        "mean_ic": 0.061, "t_stat": 2.12, "p_value": 0.08,
        # ...while the key actually ranked on clears it
        "ranking_key": {
            "key": "blend_score", "top_n": 25, "n_periods": 7,
            "mean_ic": 0.108, "ic_t_stat": 2.50, "ic_p_value": 0.047,
            "years_ic_positive": 7,
            "mean_top_n_excess": 0.0635, "top_n_p_value": 0.086,
            "years_top_n_positive": 6,
        },
    }
    out = "\n".join(_format_optimizer_status(status))
    assert "CONFIDENCE: MODERATE" in out
    assert "blend_score beats chance" in out
    assert "CONFIDENCE: LOW" not in out


def test_confidence_falls_back_to_the_gate_without_a_ranking_key():
    status = {
        "accepted": False, "n_periods": 7, "per_year": [],
        "mean_ic": 0.061, "t_stat": 2.12, "p_value": 0.08,
        "ranking_key": None,
    }
    out = "\n".join(_format_optimizer_status(status))
    assert "CONFIDENCE: LOW" in out
