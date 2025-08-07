import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from analyzer.metrics import sector_thresholds_old

from optimize_functions import (
    build_sector_threshold_grid,
    iterate_thresholds_by_sector,
    consolidate_best_thresholds,
    sanity_checks,
    detect_misaligned_scoring,
)
from optimize_config import *
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# Load and normalize
df = pd.read_csv("metrics_by_timespan.csv")

df.columns = df.columns.str.strip().str.lower()

metrics = list(sector_thresholds_old.keys())

metrics = ["de status"]
metric_cols = [m.lower() for m in metrics]
required_cols = ["company", "sector", "total_return", "timespan"] + metric_cols
df = df[[col for col in required_cols if col in df.columns]]
df.dropna(subset=["sector", "total_return"], inplace=True)

# --- Build Threshold Grid ---
sector_threshold_grid = {}
usable_metrics_per_sector = {}
sector_threshold_grid, usable_metrics_per_sector = build_sector_threshold_grid(
    metrics, sector_threshold_grid, sector_thresholds_old, usable_metrics_per_sector
)


# --- Run and Save Top 3 Threshold Sets Per Sector ---
results, debug_rows = iterate_thresholds_by_sector(
    df, sector_threshold_grid, usable_metrics_per_sector
)
debug_df = pd.DataFrame(debug_rows)
df_final = consolidate_best_thresholds(results)
df_final.to_csv("output/top_thresholds_combined.csv", index=False)
debug_df.to_csv("output/debug_rows.csv", index=False)
sanity_checks(df, df_final)

misaligned = detect_misaligned_scoring(df_final)

if not misaligned.empty:
    print("⚠️ Potential scoring misalignments detected:\n")
    print(
        misaligned[
            [
                "timespan",
                "sector",
                "thresholds",
                "avg_points",
                "avg_return",
                "avg_correlation",
            ]
        ]
    )
else:
    print("No scoring misalignments detected.")


print("Saved top 3 threshold combinations per sector to 'top_thresholds_by_sector.csv'")
print(df_final)
