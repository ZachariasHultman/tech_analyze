import pandas as pd
import json
from itertools import product
from analyzer.summary_manager import SummaryManager
from analyzer.data_processing import calculate_score
import ast
import numpy as np
from collections import defaultdict
from optimize_config import *


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
    results = []
    best_thresholds = {}

    for timespan in df["timespan"].unique():
        df_timespan = df[df["timespan"] == timespan]

        for sector in df_timespan["sector"].unique():
            sector_df = df_timespan[df_timespan["sector"] == sector].copy()
            valid_metrics = usable_metrics_per_sector.get(sector, [])
            print(
                f"Processing sector: {sector} | timespan: {timespan} | rows: {len(sector_df)} | metrics: {valid_metrics}"
            )

            for col in valid_metrics:
                if col in sector_df.columns:
                    if col == "cagr-pe compare status":
                        sector_df[col] = sector_df[col].apply(safe_parse)
                    else:
                        sector_df[col] = sector_df[col].apply(
                            lambda x: (
                                ast.literal_eval(x)
                                if isinstance(x, str) and x.startswith("[")
                                else x
                            )
                        )

            for metric in valid_metrics:
                if sector not in sector_grid.get(metric, {}):
                    continue

                initial_thresholds = sector_grid[metric][sector]
                explored = set()

                for iteration in range(MAX_ITERATIONS):
                    key = (timespan, sector, metric)
                    if iteration == 0 or best_thresholds.get(key) is None:
                        threshold_list = initial_thresholds
                    else:
                        best_threshold, best_corr = best_thresholds[key]
                        if best_corr is not None and best_corr >= CORRELATION_THRESHOLD:
                            break  # good enough, stop iterating
                        print(f"Refining thresholds for {key} for sector {sector}")
                        threshold_list = refine_thresholds(best_threshold)

                    for threshold_pair in threshold_list:
                        threshold_key = json.dumps(threshold_pair)
                        if threshold_key in explored:
                            continue
                        explored.add(threshold_key)

                        thresholds = {metric: threshold_pair}

                        sm = summary_class()
                        sm.process_historical(
                            sector_df, [metric], thresholds=thresholds
                        )
                        calculate_score(sm, metrics_to_score=[metric])

                        company_scores = []
                        returns = []
                        for _, row in sm.summary.iterrows():
                            company = row.name
                            if (
                                "points" in row
                                and row["points"] not in [None, [None]]
                                and company in sector_df["company"].values
                            ):
                                score = row["points"]
                                total_return = sector_df.loc[
                                    sector_df["company"] == company, "total_return"
                                ].values[0]
                                if total_return not in [None, [None]]:
                                    company_scores.append(score)
                                    returns.append(total_return)

                        avg_points = (
                            sum(company_scores) / len(company_scores)
                            if company_scores
                            else None
                        )
                        avg_return = sum(returns) / len(returns) if returns else None

                        correlation = None
                        company_scores = [
                            float(x)
                            for x in company_scores
                            if isinstance(x, (int, float, np.number))
                        ]
                        returns = [
                            float(x)
                            for x in returns
                            if isinstance(x, (int, float, np.number))
                        ]
                        if (
                            company_scores
                            and returns
                            and len(company_scores) == len(returns)
                        ):
                            try:
                                correlation = np.corrcoef(company_scores, returns)[0, 1]
                                if np.isnan(correlation):
                                    correlation = None
                            except Exception:
                                pass

                        if correlation is None:
                            print(
                                f"[INFO] Skipped correlation: {timespan}, {sector}, thresholds={thresholds}"
                            )

                        key = (timespan, sector, metric)
                        best_corr = best_thresholds.get(key, None)
                        _, best_corr_value = (
                            best_corr if best_corr is not None else (None, None)
                        )
                        if correlation is not None and (
                            best_corr_value is None or correlation > best_corr_value
                        ):
                            best_thresholds[key] = (threshold_pair, correlation)

                        results.append(
                            (
                                timespan,
                                sector,
                                {metric: threshold_pair},
                                sm,
                                avg_points,
                                avg_return,
                                correlation,
                            )
                        )

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
    grouped = defaultdict(dict)

    for timespan, sector, threshold_dict, sm, avg_points, avg_return, corr in results:
        metric = list(threshold_dict.keys())[0]
        current_best = grouped[(timespan, sector)].get(metric)

        if (
            current_best is None
            or current_best["correlation"] is None
            or (corr is not None and corr > current_best["correlation"])
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
        # Optionally, avg_points/returns could be averages across metrics
        final.append(
            {
                "timespan": timespan,
                "sector": sector,
                "thresholds": json.dumps(combined_thresholds),
                "avg_points": round(
                    np.mean(
                        [
                            data["avg_points"]
                            for data in metric_data.values()
                            if data["avg_points"] is not None
                        ]
                    ),
                    2,
                ),
                "avg_return": round(
                    np.mean(
                        [
                            data["avg_return"]
                            for data in metric_data.values()
                            if data["avg_return"] is not None
                        ]
                    ),
                    2,
                ),
                "avg_correlation": round(
                    np.mean(
                        [
                            data["correlation"]
                            for data in metric_data.values()
                            if data["correlation"] is not None
                        ]
                    ),
                    2,
                ),
                "num_companies": max(
                    [data["num_companies"] for data in metric_data.values()]
                ),
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
    step_range = range(-STEP_PER_DIRECTION, STEP_PER_DIRECTION + 1)

    for metric in metrics:
        m = metric
        sector_threshold_grid[m] = {}

        for sector, thresholds in sector_thresholds_old[metric].items():
            try:
                if metric == "cagr-pe compare status":
                    # Each threshold is a tuple: [(ok_cagr, ok_pe), (nok_cagr, nok_pe)]
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
                    sector_threshold_grid[m][sector] = list(
                        product(ok_pairs, nok_pairs)
                    )

                elif isinstance(thresholds, tuple):
                    high = float(thresholds[0])
                    low = float(thresholds[1])
                    if (not high and not low) or (high == 0 and low == 0):
                        continue
                    ok_values = [
                        round(high * (1 + STEP_SIZE * i), 4) for i in step_range
                    ]
                    nok_values = [
                        round(low * (1 + STEP_SIZE * j), 4) for j in step_range
                    ]
                    sector_threshold_grid[m][sector] = list(
                        product(ok_values, nok_values)
                    )
                else:
                    continue

                usable_metrics_per_sector.setdefault(sector, []).append(m)

            except (ValueError, TypeError, IndexError):
                continue
    return sector_threshold_grid, usable_metrics_per_sector
