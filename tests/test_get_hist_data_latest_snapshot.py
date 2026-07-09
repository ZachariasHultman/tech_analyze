"""Regression test: get_hist_data() must keep the most recent snapshot per
company (by filename date), not an arbitrary filesystem-iteration-order
pick. Previously `.groupby(level=0).first()` on unsorted rows silently
discarded newer --save runs for roughly half the tickers on disk (verified
against the real data/ dir: e.g. the Feb snapshot was kept over the Jun one
for one company and the reverse for another, purely by glob() order).
"""
import pandas as pd

from analyzer.historical_calc import get_hist_data


def _write_snapshot(data_dir, company, snap_date, pe_value):
    csv_path = data_dir / f"{company}_{snap_date}.csv"
    pd.DataFrame({"pe": [f'[{{"date": "{snap_date}", "value": {pe_value}}}]'],
                  "sector": ["Unknown"]}).to_csv(csv_path, index=False)
    return csv_path


def test_get_hist_data_keeps_most_recent_snapshot_per_company(tmp_path):
    # Two snapshots for the same company, older written to disk AFTER the
    # newer one (mimics arbitrary glob/filesystem ordering) — must still
    # pick the one with the later date in the filename.
    _write_snapshot(tmp_path, "Acme 123", "2026-06-03", pe_value=20)
    _write_snapshot(tmp_path, "Acme 123", "2026-02-18", pe_value=10)

    df = get_hist_data(data_dir=str(tmp_path))

    assert len(df) == 1
    assert df.loc["Acme 123", "pe"].iloc[0]["value"] == 20


def test_get_hist_data_one_row_per_company_with_multiple_companies(tmp_path):
    _write_snapshot(tmp_path, "Acme 123", "2026-02-18", pe_value=10)
    _write_snapshot(tmp_path, "Acme 123", "2026-06-03", pe_value=20)
    _write_snapshot(tmp_path, "Beta 456", "2026-06-03", pe_value=5)

    df = get_hist_data(data_dir=str(tmp_path))

    assert sorted(df.index) == ["Acme 123", "Beta 456"]
    assert df.loc["Acme 123", "pe"].iloc[0]["value"] == 20
