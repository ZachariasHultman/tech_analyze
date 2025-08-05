"""
Analyzes and visualizes metric distributions by sector to evaluate threshold alignment.

This script loads a dataset containing sector-based financial or operational metrics
(e.g., 'de status') and visualizes the distribution of values per sector using boxplots.
Outliers are shown to highlight data variability. The goal is to support manual review
or refinement of initial thresholds used in scoring models.

Typical use cases:
- Verifying whether metric thresholds are within realistic value ranges
- Identifying sectors where thresholds might miss key trends
- Supporting threshold optimization by visual inspection

Requirements:
- Input CSV file must include 'sector' and the target metric as columns (e.g., 'de status')

Example:
    $ python check_metric_thresholds.py --metric "de status" --file metrics_by_timespan.csv

Arguments (if applicable via CLI):
    --metric: The name of the metric column to visualize (e.g., "de status")
    --file: Path to the input CSV file containing sector and metric data

Output:
    - A boxplot showing the distribution of the specified metric per sector
    - Optional visual markers or overlays (e.g., initial thresholds, outliers)

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the uploaded CSV file for analysis
csv_path = "./metrics_by_timespan.csv"
df = pd.read_csv(csv_path)

# Normalize column names
df.columns = df.columns.str.strip().str.lower()
metric = "de status"
# Filter to relevant columns
if metric in df.columns and "sector" in df.columns:
    data = df[["sector", "de status"]].dropna()
else:
    raise ValueError("Missing required columns: 'sector' and 'de status'.")

# Create boxplot comparing 'de status' distribution per sector
plt.figure(figsize=(14, 6))
sns.boxplot(data=data, x="sector", y="de status")
plt.xticks(rotation=90)
plt.title("Distribution of 'de status' per Sector")
plt.tight_layout()
plt.grid(True)
plt.show()
