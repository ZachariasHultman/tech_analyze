"""Item 7 landmine regression: _build_export's drop_cols removes any column
ending in "_score" -- combined_score must be explicitly spared so it survives
CSV export.
"""
import pandas as pd

from analyzer.summary_manager import SummaryManager


def test_combined_score_survives_csv_export(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    m = SummaryManager()
    m.summary = pd.DataFrame(
        {
            "sector": ["Industri", "Industri"],
            "points": [3.0, -1.0],
            "piotroski f-score status": [7.0, 2.0],
            "piotroski f-score status_score": [1.0, -1.0],
            "quality_pct": [0.9, 0.2],
            "value_pct": [0.8, 0.3],
            "combined_score": [0.72, 0.06],
        },
        index=["Alpha AB 111", "Beta AB 222"],
    )
    m.summary_investment = pd.DataFrame()

    m._display(save_df=True)

    out = pd.read_csv(tmp_path / "summary.csv", index_col=0)
    assert "combined_score" in out.columns
    # Per-metric *_score column was dropped; combined_score kept.
    assert "piotroski f-score status_score" not in out.columns
    # Sorted by combined_score descending.
    assert list(out.index) == ["Alpha AB 111", "Beta AB 222"]
    assert float(out.loc["Alpha AB 111", "combined_score"]) == 0.72
