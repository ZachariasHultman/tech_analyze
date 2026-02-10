import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from analyzer.metrics import (
    RATIO_SPECS,
    GLOBAL_THRESHOLDS,
    HIGHEST_WEIGHT_METRICS,
    HIGH_WEIGHT_METRICS,
    LOW_WEIGHT_METRICS,
)

from optimize_functions import (
    build_sector_threshold_grid,
    iterate_thresholds_by_sector,
    consolidate_best_thresholds,
    sanity_checks,
)
from optimize_config import *
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# Load and normalize
df = pd.read_csv("metrics_by_timespan.csv")
df.columns = df.columns.str.strip().str.lower()

# Build seeds from current thresholds (sector-agnostic → use "all" as sector key)
all_metrics = list(
    set(HIGHEST_WEIGHT_METRICS) | set(HIGH_WEIGHT_METRICS) | set(LOW_WEIGHT_METRICS)
)

seeds_by_sector = {}
for m in all_metrics:
    if m in RATIO_SPECS:
        thr = RATIO_SPECS[m]["thr"]
    elif m in GLOBAL_THRESHOLDS:
        thr = GLOBAL_THRESHOLDS[m]
    else:
        continue
    # Use every sector present in the data as seed (sector-agnostic thresholds)
    seeds_by_sector[m] = {s: thr for s in df["sector"].dropna().unique()}

metric_cols = [m.lower() for m in all_metrics]
required_cols = ["company", "sector", "total_return", "timespan"] + metric_cols
df = df[[col for col in required_cols if col in df.columns]]
df.dropna(subset=["sector", "total_return"], inplace=True)

# --- Build Threshold Grid ---
sector_threshold_grid = {}
usable_metrics_per_sector = {}
sector_threshold_grid, usable_metrics_per_sector = build_sector_threshold_grid(
    all_metrics, sector_threshold_grid, seeds_by_sector, usable_metrics_per_sector
)

os.makedirs("output", exist_ok=True)

# --- Run and Save Top 3 Threshold Sets Per Sector ---
results = iterate_thresholds_by_sector(
    df, sector_threshold_grid, usable_metrics_per_sector
)
df_final = consolidate_best_thresholds(results)
df_final.to_csv("output/top_thresholds_combined.csv", index=False)
sanity_checks(df, df_final)

print("Saved top 3 threshold combinations to 'output/top_thresholds_combined.csv'")
print(df_final)
