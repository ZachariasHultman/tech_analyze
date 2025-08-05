import pandas as pd
import json
from itertools import product
from analyzer.summary_manager import SummaryManager
from analyzer.data_processing import calculate_score
import ast
import numpy as np
from collections import defaultdict
from optimize_config import *
from collections import defaultdict


def safe_parse(val):
    try:
        parsed = ast.literal_eval(val) if isinstance(val, str) else val
        # Ensure it's a list of 2 numerical values
        if (
            isinstance(parsed, list)
            and len(parsed) == 2
            and all(isinstance(x, (int, float)) and x is not None for x in parsed)
        ):
            return parsed
        elif (
            isinstance(parsed, list)
            and len(parsed) == 1
            and isinstance(parsed[0], list)
            and len(parsed[0]) == 2
            and all(isinstance(x, (int, float)) and x is not None for x in parsed[0])
        ):
            return parsed[0]
        else:
            print("Invalid parsed structure, defaulting to None,None:", parsed)

    except Exception:
        print("Non-list or malformed value, defaulting to None,None:", val)
    return [None, None]


def refine_thresholds(current_thresholds, step=0.05):
    def expand_range(value, factor=step):
        return [round(value * (1 - factor), 4), round(value * (1 + factor), 4)]

    refined = set()

    if isinstance(current_thresholds[0], (float, int)):
        # Tuple-style: [ok_val, nok_val]
        ok_vals = expand_range(current_thresholds[0])
        nok_vals = expand_range(current_thresholds[1])
        for ok in ok_vals:
            for nok in nok_vals:
                refined.add((ok, nok))

    elif (
        isinstance(current_thresholds[0], (list, tuple))
        and len(current_thresholds) == 2
    ):
        # CAGR-PE style: [[ok_cagr, ok_pe], [nok_cagr, nok_pe]]
        ok_cagr_vals = expand_range(current_thresholds[0][0])
        ok_pe_vals = expand_range(current_thresholds[0][1])
        nok_cagr_vals = expand_range(current_thresholds[1][0])
        nok_pe_vals = expand_range(current_thresholds[1][1])
        for ok_cagr in ok_cagr_vals:
            for ok_pe in ok_pe_vals:
                for nok_cagr in nok_cagr_vals:
                    for nok_pe in nok_pe_vals:
                        refined.add(((ok_cagr, ok_pe), (nok_cagr, nok_pe)))

    return list(refined)


# --- Evaluate Threshold Sets by Score ---
def iterate_thresholds_by_sector(
    df, sector_grid, usable_metrics_per_sector, summary_class=SummaryManager
):
    """
    Iteratively evaluates metric thresholds per sector and timespan to optimize correlation
    between computed score (points) and historical total return.

    For each (sector, timespan, metric) combination, it:
    - Applies a range of threshold pairs to calculate scores.
    - Computes average score, average return, and their correlation.
    - Tracks and refines the best-performing thresholds across multiple iterations
      if correlation is below a configured threshold (CORRELATION_THRESHOLD).
    - Returns all tested configurations sorted by descending correlation.

    Parameters:
        df (pd.DataFrame): Cleaned input dataset containing sectors, metrics, and returns.
        sector_grid (dict): Precomputed dictionary of threshold pairs per sector and metric.
        usable_metrics_per_sector (dict): Mapping of valid metrics per sector.
        summary_class (class): A class to handle score processing, defaults to SummaryManager.

    Returns:
        list of tuples: Each tuple contains:
            - timespan (str)
            - sector (str)
            - thresholds (dict): Metric to threshold pair
            - summary object
            - average score (float)
            - average return (float)
            - correlation (float)
    """
    results = []
    best_thresholds = {}

    for timespan in df["timespan"].unique():
        df_timespan = df[df["timespan"] == timespan]

        for sector in df_timespan["sector"].unique():
            sector_df = df_timespan[df_timespan["sector"] == sector].copy()
            valid_metrics = usable_metrics_per_sector.get(sector, [])

            for metric in valid_metrics:
                thresholds_list = sector_grid.get(metric, {}).get(sector, [])
                if not thresholds_list:
                    continue

                explored = set()
                best_score = -np.inf
                best_threshold = None

                for iteration in range(MAX_ITERATIONS):
                    threshold_candidates = (
                        thresholds_list
                        if iteration == 0 or best_threshold is None
                        else refine_thresholds(best_threshold)
                    )

                    for threshold in threshold_candidates:
                        threshold_key = json.dumps(threshold)
                        if threshold_key in explored:
                            continue
                        explored.add(threshold_key)

                        thresholds = {metric: threshold}
                        sm = summary_class()
                        sm.process_historical(
                            sector_df, [metric], thresholds=thresholds
                        )
                        calculate_score(sm, metrics_to_score=[metric])

                        df_eval = sm.summary.copy()
                        if df_eval.empty or "points" not in df_eval.columns:
                            continue

                        scores = df_eval["points"]
                        returns = df_eval.index.to_series().map(
                            sector_df.set_index("company")["total_return"]
                        )

                        valid_mask = (~scores.isna()) & (~returns.isna())
                        scores = scores[valid_mask]
                        returns = returns[valid_mask]

                        if len(scores) < 3:
                            continue

                        correlation = scores.corr(returns)
                        avg_points = scores.mean()
                        avg_return = returns.mean()
                        spread = (
                            returns[scores > 0].mean() - returns[scores <= 0].mean()
                        )
                        sign_corr = np.sign(scores).corr(np.sign(returns))

                        composite_score = (
                            (correlation if not np.isnan(correlation) else 0) * 0.4
                            + (spread if not np.isnan(spread) else 0) * 0.4
                            + (sign_corr if not np.isnan(sign_corr) else 0) * 0.2
                        )

                        key = (timespan, sector, metric)
                        _, best_corr_value = best_thresholds.get(key, (None, None))
                        if correlation is not None and (
                            best_corr_value is None or correlation > best_corr_value
                        ):
                            best_thresholds[key] = (threshold, correlation)

                        results.append(
                            (
                                timespan,
                                sector,
                                {metric: threshold},
                                sm,
                                avg_points,
                                avg_return,
                                correlation,
                            )
                        )

                    if best_score >= CORRELATION_THRESHOLD:
                        break

    return sorted(
        results,
        key=lambda x: (
            x[4] is not None,
            x[6] is not None,
            x[6] if x[6] is not None else float("-inf"),
        ),
        reverse=True,
    )


def consolidate_best_thresholds(results):
    """
    Consolidates the best threshold for each metric per sector and timespan.

    Keeps one best-performing threshold per metric in every (sector, timespan) combination,
    based on maximum correlation.

    Returns:
        pd.DataFrame: One row per (timespan, sector) with all best metric thresholds.
    """

    grouped = defaultdict(dict)

    for timespan, sector, threshold_dict, sm, avg_points, avg_return, corr in results:
        metric = list(threshold_dict.keys())[0]
        current = grouped[(timespan, sector)].get(metric)

        # Replace only if better correlation
        if (
            current is None
            or current["correlation"] is None
            or (corr is not None and corr > current["correlation"])
        ):
            grouped[(timespan, sector)][metric] = {
                "threshold": threshold_dict[metric],
                "avg_points": avg_points,
                "avg_return": avg_return,
                "correlation": corr,
                "num_companies": len(sm.summary),
            }

    final = []
    for (timespan, sector), metric_data in grouped.items():
        combined_thresholds = {
            metric: data["threshold"] for metric, data in metric_data.items()
        }

        avg_points = np.mean(
            [
                data["avg_points"]
                for data in metric_data.values()
                if data["avg_points"] is not None
            ]
        )
        avg_return = np.mean(
            [
                data["avg_return"]
                for data in metric_data.values()
                if data["avg_return"] is not None
            ]
        )
        avg_corr = np.mean(
            [
                data["correlation"]
                for data in metric_data.values()
                if data["correlation"] is not None
            ]
        )
        num_companies = max(data["num_companies"] for data in metric_data.values())

        final.append(
            {
                "timespan": timespan,
                "sector": sector,
                "thresholds": json.dumps(combined_thresholds),
                "avg_points": round(avg_points, 2),
                "avg_return": round(avg_return, 2),
                "avg_correlation": round(avg_corr, 2),
                "num_companies": num_companies,
            }
        )

    return pd.DataFrame(final)


def sanity_checks(df, df_final):

    # 1. Expected rows = unique sectors * unique timespans
    expected_rows = df["sector"].nunique() * df["timespan"].nunique()
    actual_rows = len(df_final)

    if actual_rows < expected_rows:
        missing = expected_rows - actual_rows
        print(
            f"[WARN] Missing {missing} (sector, timespan) combinations in final output."
        )

    # 2. Check for NaNs in critical fields
    nan_cols = df_final[["avg_points", "avg_return", "avg_correlation"]].isna().sum()
    if nan_cols.any():
        print("[WARN] Missing values detected:")
        print(nan_cols[nan_cols > 0])

    # 3. Check that each 'thresholds' field is populated and valid JSON
    invalid_thresholds = (
        df_final["thresholds"]
        .apply(lambda x: not isinstance(json.loads(x), dict))
        .sum()
    )
    if invalid_thresholds > 0:
        print(f"[WARN] {invalid_thresholds} thresholds could not be parsed as dict.")

    # 4. Optional: Check correlation bounds
    if (df_final["avg_correlation"] > 1).any() or (
        df_final["avg_correlation"] < -1
    ).any():
        print("[WARN] Correlation values outside expected [-1, 1] range.")

    # 5. Optional: Print top low-correlation rows for manual review
    low_corr = df_final[df_final["avg_correlation"] < CORRELATION_THRESHOLD]
    if not low_corr.empty:
        print(
            f"[INFO] {len(low_corr)} rows have weak correlation ({CORRELATION_THRESHOLD}):"
        )
        print(low_corr[["sector", "timespan", "avg_correlation"]])


def build_sector_threshold_grid(
    metrics, sector_threshold_grid, sector_thresholds_old, usable_metrics_per_sector
):
    """
    Constructs a grid of threshold combinations for each metric and sector.

    Expands base thresholds using STEP_SIZE and STEP_PER_DIRECTION.
    Automatically handles direction of thresholds using metric naming.

    Returns:
        - sector_threshold_grid (dict): Populated threshold combinations per metric/sector
        - usable_metrics_per_sector (dict): Metrics applicable per sector
    """
    step_range = range(-STEP_PER_DIRECTION, STEP_PER_DIRECTION + 1)
    bad_if_high_keywords = ["de status", "peg status", "net debt - ebitda status"]

    for metric in metrics:
        sector_threshold_grid[metric] = {}

        for sector, thresholds in sector_thresholds_old[metric].items():
            try:
                if metric == "cagr-pe compare status":
                    ok_base = (float(thresholds[0][0]), float(thresholds[0][1]))
                    nok_base = (float(thresholds[1][0]), float(thresholds[1][1]))

                    ok_cagr_values = [
                        round(ok_base[0] * (1 + STEP_SIZE * i), 4) for i in step_range
                    ]
                    ok_pe_values = [
                        round(ok_base[1] * (1 + STEP_SIZE * j), 4) for j in step_range
                    ]
                    nok_cagr_values = [
                        round(nok_base[0] * (1 + STEP_SIZE * i), 4) for i in step_range
                    ]
                    nok_pe_values = [
                        round(nok_base[1] * (1 + STEP_SIZE * j), 4) for j in step_range
                    ]

                    ok_pairs = list(product(ok_cagr_values, ok_pe_values))
                    nok_pairs = list(product(nok_cagr_values, nok_pe_values))
                    sector_threshold_grid[metric][sector] = list(
                        product(ok_pairs, nok_pairs)
                    )

                elif isinstance(thresholds, tuple):
                    high, low = float(thresholds[0]), float(thresholds[1])
                    if (not high and not low) or (high == 0 and low == 0):
                        continue

                    good_if_high = not any(k in metric for k in bad_if_high_keywords)
                    ok_val, nok_val = (high, low) if good_if_high else (low, high)

                    ok_values = [
                        round(ok_val * (1 + STEP_SIZE * i), 4) for i in step_range
                    ]
                    nok_values = [
                        round(nok_val * (1 + STEP_SIZE * j), 4) for j in step_range
                    ]
                    sector_threshold_grid[metric][sector] = list(
                        product(ok_values, nok_values)
                    )

                else:
                    continue

                usable_metrics_per_sector.setdefault(sector, []).append(metric)

            except (ValueError, TypeError, IndexError):
                continue

    return sector_threshold_grid, usable_metrics_per_sector
