import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


# ─────────────────────────────────────────────────────────────────────────────
def load_data(file_path, metric):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()
    metric = metric.strip().lower()

    if "sector" not in df.columns or metric not in df.columns:
        raise ValueError(f"Missing required columns: 'sector' and '{metric}'.")

    return df, metric


def load_thresholds(threshold_file, metric_name):
    if not os.path.exists(threshold_file):
        print(f"[WARNING] Threshold file '{threshold_file}' not found.")
        return {}

    df = pd.read_csv(threshold_file)
    df["thresholds"] = df["thresholds"].apply(eval)  # convert from string to dict

    # Filter only matching metric
    df = df[df["thresholds"].apply(lambda d: metric_name in d)]

    thresholds = {}
    for (sector, timespan), group in df.groupby(["sector", "timespan"]):
        # Use the row with highest correlation per (sector, timespan)
        best_row = group.sort_values("avg_correlation", ascending=False).iloc[0]
        threshold = best_row["thresholds"][metric_name]
        thresholds[(sector, timespan)] = threshold

    return thresholds


def print_summary_statistics(df, metric):
    summary = (
        df.groupby("sector")[metric]
        .describe()[["min", "25%", "50%", "75%", "max"]]
        .rename(columns={"25%": "Q1", "50%": "Median", "75%": "Q3"})
    )
    print(f"\nSummary statistics for '{metric}' per sector:\n")
    print(summary.round(3))


def plot_boxplot(df, metric):
    data = df[["sector", metric]].dropna()
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=data, x="sector", y=metric)
    plt.xticks(rotation=90)
    plt.title(f"Boxplot: Distribution of '{metric}' per Sector")
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def plot_metric_distribution_by_sector_and_timespan(df, metric, thresholds_dict):
    # thresholds_dict: {(sector, timespan): (ok, nok)}

    sectors = df["sector"].unique()

    for sector in sectors:
        plt.figure(figsize=(10, 6))
        sector_df = df[df["sector"] == sector]

        # Only plot timespans that exist in the thresholds
        timespans = [
            ts
            for ts in sector_df["timespan"].unique()
            if (sector, ts) in thresholds_dict
        ]
        if not timespans:
            continue  # Skip this sector if no valid timespans

        palette = sns.color_palette("husl", len(timespans))
        color_map = dict(zip(timespans, palette))

        # Plot one KDE per timespan
        for ts in timespans:
            data_ts = sector_df[sector_df["timespan"] == ts]
            sns.kdeplot(
                data=data_ts,
                x=metric,
                fill=True,
                linewidth=2,
                alpha=0.3,
                label=f"{ts} distribution",
                color=color_map[ts],
            )
            if len(data_ts[metric].dropna()) < 3:
                continue

            # Plot OK and NOK thresholds
            ok, nok = thresholds_dict[(sector, ts)]
            plt.axvline(
                x=ok,
                color=color_map[ts],
                linestyle="--",
                linewidth=1.5,
                label=f"{ts} OK",
                alpha=0.8,
            )
            plt.axvline(
                x=nok,
                color=color_map[ts],
                linestyle=":",
                linewidth=1.5,
                label=f"{ts} NOK",
                alpha=0.8,
            )

        plt.title(f"{metric} distribution in sector: {sector}")
        plt.xlabel(metric)
        plt.grid(True)
        plt.tight_layout()
        plt.legend(title="Legend", loc="upper right", fontsize="small")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot metric distribution by sector")
    parser.add_argument(
        "--metric", required=True, help="Metric to analyze (e.g., 'de status')"
    )
    parser.add_argument("--file", required=True, help="Path to CSV file")
    args = parser.parse_args()

    df, metric = load_data(args.file, args.metric)

    # Load thresholds from file if exists
    threshold_path = "./output/top_thresholds_combined.csv"
    thresholds_dict = load_thresholds(threshold_path, metric)

    print_summary_statistics(df, metric)
    plot_boxplot(df, metric)

    if "timespan" in df.columns:
        plot_metric_distribution_by_sector_and_timespan(df, metric, thresholds_dict)


if __name__ == "__main__":
    main()
