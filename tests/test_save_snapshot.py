"""Regression test for item 7: save_snapshot overwrites (not appends) when
called twice for the same day/path — avoids accumulating duplicate rows on
a re-run of --save.
"""
import pandas as pd

from analyzer.helper import save_snapshot


def test_save_snapshot_dict_overwrites_on_second_call(tmp_path):
    csv_path = tmp_path / "snap.csv"
    save_snapshot({"pe": 10.0}, csv_path, asof="2024-01-01")
    save_snapshot({"pe": 11.0}, csv_path, asof="2024-01-01")

    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert df.iloc[0]["pe"] == 11.0


def test_save_snapshot_dataframe_overwrites_on_second_call(tmp_path):
    csv_path = tmp_path / "snap_df.csv"
    save_snapshot(pd.DataFrame({"value": [1, 2]}), csv_path, asof="2024-01-01")
    save_snapshot(pd.DataFrame({"value": [9]}), csv_path, asof="2024-01-01")

    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert df.iloc[0]["value"] == 9
