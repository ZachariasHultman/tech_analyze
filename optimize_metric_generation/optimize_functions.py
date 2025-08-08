import pandas as pd
import json
from itertools import product
from analyzer.summary_manager import SummaryManager
from analyzer.data_processing import calculate_score
import ast
import numpy as np
from optimize_config import *
from analyzer.metrics import extract_sector
from collections import defaultdict


BAD_IF_HIGH_KEYWORDS = {"de status", "peg status", "net debt - ebitda status"}


def metric_direction(metric: str) -> int:
    """
    +1  => higher is better
    -1  => lower is better
    """
    return -1 if any(k in metric for k in BAD_IF_HIGH_KEYWORDS) else +1


def normalize_threshold_tuple(metric: str, pair):
    """
    Ensure tuple is (NOK, OK) with correct inequality based on direction.
    For higher-is-better: NOK < OK
    For lower-is-better:  NOK > OK
    Swaps if necessary; returns (nok, ok) floats.
    """
    nok, ok = float(pair[0]), float(pair[1])
    if metric_direction(metric) == +1:  # higher is better
        if nok >= ok:
            nok, ok = min(nok, ok), max(nok, ok)
    else:  # lower is better
        if nok <= ok:
            nok, ok = max(nok, ok), min(nok, ok)
    return (nok, ok)


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
    """
    Canonical order is (NOK, OK).
    Expand small neighborhoods around the base thresholds.
    Supports:
      - simple: (nok, ok)
      - composite: ((nok_cagr, nok_pe), (ok_cagr, ok_pe))  # see note below
    """

    def expand_range(value, factor=step):
        return [round(value * (1 - factor), 4), round(value * (1 + factor), 4)]

    refined = set()

    # --- simple (nok, ok) ---
    if isinstance(current_thresholds[0], (float, int)) and isinstance(
        current_thresholds[1], (float, int)
    ):
        nok_vals = expand_range(float(current_thresholds[0]))
        ok_vals = expand_range(float(current_thresholds[1]))
        for nok in nok_vals:
            for ok in ok_vals:
                refined.add((nok, ok))

    # --- composite two-tuples ((nok_cagr, nok_pe), (ok_cagr, ok_pe)) ---
    elif (
        isinstance(current_thresholds[0], (list, tuple))
        and isinstance(current_thresholds[1], (list, tuple))
        and len(current_thresholds[0]) == 2
        and len(current_thresholds[1]) == 2
    ):
        nok_cagr_vals = expand_range(float(current_thresholds[0][0]))
        nok_pe_vals = expand_range(float(current_thresholds[0][1]))
        ok_cagr_vals = expand_range(float(current_thresholds[1][0]))
        ok_pe_vals = expand_range(float(current_thresholds[1][1]))
        for nok_cagr in nok_cagr_vals:
            for nok_pe in nok_pe_vals:
                for ok_cagr in ok_cagr_vals:
                    for ok_pe in ok_pe_vals:
                        refined.add(((nok_cagr, nok_pe), (ok_cagr, ok_pe)))

    return list(refined)


# --- Evaluate Threshold Sets by Score ---


def _normalize_sector_value(val):
    try:
        return extract_sector(val)  # val is a string like "Industri"
    except Exception:
        return val


def _norm_keys_dict(d):
    """Normalize sector keys for {sector: ...} mappings."""
    return {_normalize_sector_value(k): v for k, v in d.items()}


def _norm_grid_keys(grid):
    """Normalize sector keys in {metric: {sector: [(nok, ok), ...]}}."""
    return {metric: _norm_keys_dict(by_sector) for metric, by_sector in grid.items()}


def iterate_thresholds_by_sector(
    df, sector_grid, usable_metrics_per_sector, summary_class=SummaryManager
):
    """
    Keeps gate; if no candidate passes per (timespan, sector, metric), fallback to best-by-objective.
    Enforces diversity on NOK and OK. Excludes neutral-only candidates.
    """
    # --- normalize sector names in BOTH data and dicts ---
    df = df.copy()
    if "sector" in df.columns:
        df["sector"] = df["sector"].apply(_normalize_sector_value)

    sector_grid = _norm_grid_keys(sector_grid)
    usable_metrics_per_sector = _norm_keys_dict(usable_metrics_per_sector)

    rows_by_key = defaultdict(list)

    stats_drop = {
        "no_candidates": 0,
        "lt3_rows": 0,
        "all_neutral": 0,
        "below_gate": 0,
        "to_wide": 0,
    }
    too_wide = defaultdict(lambda: {"count": 0, "samples": []})

    for timespan in df["timespan"].unique():
        df_timespan = df[df["timespan"] == timespan]

        for sector in df_timespan["sector"].unique():
            sector_df = df_timespan[df_timespan["sector"] == sector].copy()
            valid_metrics = usable_metrics_per_sector.get(sector, [])
            if not valid_metrics:
                stats_drop["no_candidates"] += 1
                continue

            for metric in valid_metrics:
                thresholds_list = sector_grid.get(metric, {}).get(sector, [])
                if not thresholds_list:
                    stats_drop["no_candidates"] += 1
                    continue

                explored = set()
                best_threshold = None

                for iteration in range(MAX_ITERATIONS):
                    threshold_candidates = (
                        thresholds_list
                        if iteration == 0 or best_threshold is None
                        else refine_thresholds(best_threshold)
                    )

                    for threshold in threshold_candidates:
                        t_key = json.dumps(threshold)
                        if t_key in explored:
                            continue
                        explored.add(t_key)

                        # Canonicalize (NOK, OK)
                        if (
                            isinstance(threshold, (list, tuple))
                            and len(threshold) == 2
                            and all(isinstance(x, (int, float)) for x in threshold)
                        ):
                            threshold = normalize_threshold_tuple(metric, threshold)

                        thresholds = {metric: threshold}

                        sm = summary_class()
                        sm.process_historical(
                            sector_df, [metric], thresholds=thresholds
                        )
                        calculate_score(sm, metrics_to_score=[metric])

                        df_eval = sm.summary
                        if (
                            df_eval is None
                            or df_eval.empty
                            or "points" not in df_eval.columns
                        ):
                            continue

                        scores = df_eval["points"]
                        returns = sector_df.set_index("company")[
                            "total_return"
                        ].reindex(df_eval.index)

                        valid_mask = (~scores.isna()) & (~returns.isna())
                        scores = scores[valid_mask]
                        returns = returns[valid_mask]
                        if len(scores) < 3:
                            stats_drop["lt3_rows"] += 1
                            continue

                        # exclude neutrals from stats
                        pos_mask = scores > 0
                        neg_mask = scores < 0
                        pos_n, neg_n = int(pos_mask.sum()), int(neg_mask.sum())

                        if pos_n <= BUCKET_SIZE and neg_n <= BUCKET_SIZE:

                            stats_drop["to_wide"] += 1
                            key = (sector, timespan, metric)
                            rec = too_wide[key]
                            rec["count"] += 1
                            # capture up to 3 sample threshold pairs for this key
                            pair = next(iter(thresholds.values()))  # (NOK, OK)
                            if len(rec["samples"]) < 3:
                                # store as a plain tuple of floats to avoid JSON noise
                                rec["samples"].append((float(pair[0]), float(pair[1])))
                            continue  # (usually you want to skip evaluation for this candidate)

                        if pos_n == 0 and neg_n == 0:
                            stats_drop["all_neutral"] += 1
                            continue

                        pearson = scores.corr(returns)
                        spearman = scores.corr(returns, method="spearman")
                        correlation = spearman if not np.isnan(spearman) else pearson
                        correlation = (
                            0.0
                            if correlation is None or np.isnan(correlation)
                            else float(correlation)
                        )

                        avg_points = float(scores.mean())
                        avg_return = float(returns.median())

                        pos_mean = returns[pos_mask].mean() if pos_n > 0 else np.nan
                        neg_mean = returns[neg_mask].mean() if neg_n > 0 else np.nan
                        spread = (
                            0.0
                            if (np.isnan(pos_mean) or np.isnan(neg_mean))
                            else float(pos_mean - neg_mean)
                        )

                        objective = (
                            correlation * OBJECTIVE_CORRELATION_WEIGHT
                            + (0.0 if np.isnan(spread) else spread)
                            * OBJECTIVE_SPREAD_WEIGHT
                        )
                        passed = (objective >= COMPOSITE_SCORE_THRESHOLD) and (
                            correlation >= CORRELATION_THRESHOLD
                        )

                        if objective < COMPOSITE_SCORE_THRESHOLD:
                            stats_drop["below_gate"] += 1
                            # still collect into a side bucket (we’ll fallback later)
                            pass

                        nok_val, ok_val = next(iter(thresholds.values()))
                        row = {
                            "timespan": timespan,
                            "sector": sector,
                            "metric": metric,
                            "thresholds": {metric: (float(nok_val), float(ok_val))},
                            "avg_points": avg_points,
                            "avg_return": avg_return,
                            "avg_correlation": correlation,
                            "objective": float(objective),
                            "_nok": float(nok_val),
                            "_ok": float(ok_val),  # for diversity
                            "_passed": passed,
                        }
                        key = (timespan, sector, metric)
                        rows_by_key[key].append(row)

                        # simple refinement anchor
                        if best_threshold is None or (
                            objective > 0 and correlation >= 0
                        ):
                            best_threshold = threshold

        # ---- pick the single best per (timespan, sector, metric) ----
    results = []
    for key, lst in rows_by_key.items():
        passed = [r for r in lst if r.get("_passed", False)]
        if not passed:
            # skip this (sector, timespan, metric) entirely
            continue
        best = max(passed, key=lambda d: d.get("objective", float("-inf")))
        for k in ("_nok", "_ok", "_passed"):
            best.pop(k, None)
        results.append(best)

    # deterministic order
    results.sort(key=lambda d: (d["timespan"], d["sector"], d["metric"]))
    print(f"[INFO] candidates drops={stats_drop}")
    # compact “too wide” summary: top 10 keys by count
    if too_wide:
        TOP_SHOW = 10
        items = sorted(too_wide.items(), key=lambda kv: kv[1]["count"], reverse=True)[
            :TOP_SHOW
        ]
        parts = []
        for (sec, tsp, met), rec in items:
            samp = ", ".join(f"({a:.4g},{b:.4g})" for (a, b) in rec["samples"])
            parts.append(f"{sec}|{tsp}|{met}: {rec['count']} [ex: {samp}]")
        print("[INFO] too_wide_by_key:", " ; ".join(parts))
    return results


def consolidate_best_thresholds(results, top_k=3):
    """
    results: list[dict] produced by iterate_thresholds_by_sector.
      Each dict MUST include at least:
        sector, timespan, metric, thresholds, avg_points, avg_return, avg_correlation, objective
      'thresholds' can be a tuple/dict; we serialize to JSON for the CSV.

    Returns a dataframe with:
      ['sector','timespan','metric','rank','thresholds','avg_points','avg_return','avg_correlation','objective']
      One row per (sector, timespan, metric, rank) up to top_k.
    """
    if not results:
        return pd.DataFrame(
            columns=[
                "sector",
                "timespan",
                "metric",
                "rank",
                "thresholds",
                "avg_points",
                "avg_return",
                "avg_correlation",
                "objective",
            ]
        )

    df = pd.DataFrame(results)

    # Ensure needed columns exist (fill if absent)
    for col in ["avg_points", "avg_return", "avg_correlation", "objective"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Serialize thresholds to JSON string for portability
    def _to_jsonable(x):
        try:
            json.dumps(x)  # already jsonable
            return x
        except Exception:
            if hasattr(x, "_asdict"):
                return x._asdict()
            try:
                return tuple(x)
            except Exception:
                return str(x)

    if "thresholds" not in df.columns:
        # try to recover from possible alternative column names
        alt = next((c for c in df.columns if c.startswith("threshold")), None)
        if alt is not None:
            df["thresholds"] = df[alt]
        else:
            df["thresholds"] = None

    df["thresholds"] = df["thresholds"].apply(_to_jsonable).apply(json.dumps)

    # Rank within (sector, timespan, metric) by objective (descending = better)
    # If you optimize something else, adjust ascending accordingly.
    df["rank"] = (
        df.sort_values(
            ["sector", "timespan", "metric", "objective"],
            ascending=[True, True, True, False],
        )
        .groupby(["sector", "timespan", "metric"])
        .cumcount()
        + 1
    )

    # Keep only top_k
    df = df[df["rank"] <= top_k]

    # Order columns
    cols = [
        "sector",
        "timespan",
        "metric",
        "rank",
        "thresholds",
        "avg_points",
        "avg_return",
        "avg_correlation",
        "objective",
    ]
    return df[cols].reset_index(drop=True)


def sanity_checks(df_raw, df_final):
    """
    df_raw: original metrics_by_timespan dataframe (lowercased columns).
    df_final: output from consolidate_best_thresholds(...).
    Prints warnings; raises nothing.
    """

    # 0) Basic schema info
    if df_final is None or len(df_final) == 0:
        print("[WARN] df_final is empty.")
        return

    # 1) Expected rows: depends on what df_final aggregates by.
    # If df_final has timespan AND sector AND metric → expect S*T*M*top_k
    # If it lacks timespan → only S*M*top_k
    uniq_sectors = df_raw["sector"].dropna().unique()
    uniq_timespans = (
        df_raw["timespan"].dropna().unique() if "timespan" in df_raw.columns else []
    )
    S, T = len(uniq_sectors), len(uniq_timespans)

    has_sector = "sector" in df_final.columns
    has_timespan = "timespan" in df_final.columns
    has_metric = "metric" in df_final.columns
    has_rank = "rank" in df_final.columns

    top_k = (
        int(df_final["rank"].max())
        if has_rank and not df_final["rank"].isna().all()
        else 1
    )

    # 2) Column presence checks
    needed_cols = ["thresholds"]
    optional_stats = ["avg_points", "avg_return", "avg_correlation"]
    for c in needed_cols:
        if c not in df_final.columns:
            print(f"[WARN] Column '{c}' not found in df_final.")

    # 3) Validate thresholds JSON if present
    if "thresholds" in df_final.columns:

        def _is_valid_json_dict(x):
            try:
                obj = json.loads(x) if isinstance(x, str) else x
                return isinstance(obj, (dict, list, tuple))
            except Exception:
                return False

        invalid_thresholds = (~df_final["thresholds"].apply(_is_valid_json_dict)).sum()
        if invalid_thresholds > 0:
            print(
                f"[WARN] {invalid_thresholds} 'thresholds' entries not valid JSON/dict-like."
            )

    # 4) Stats columns (avg_points/avg_return/avg_correlation) if present
    present_stats = [c for c in optional_stats if c in df_final.columns]
    if present_stats:
        nan_cols = df_final[present_stats].isna().sum()
        if nan_cols.any():
            print(f"[WARN] NaNs in stats columns:\n{nan_cols.to_dict()}")
        if "avg_correlation" in df_final.columns:
            if (df_final["avg_correlation"] > 1).any() or (
                df_final["avg_correlation"] < -1
            ).any():
                print("[WARN] Correlation values outside expected [-1, 1] range.")
    else:
        print("[INFO] Stats columns not present in df_final; skipping stats checks.")


def build_sector_threshold_grid(
    metrics, sector_threshold_grid, seeds_by_sector, usable_metrics_per_sector
):
    """
    For each metric & sector, build a list of (NOK, OK) candidate pairs by
    varying BOTH ends around the seed pair, while preserving ordering
    per metric direction.
    """
    sector_threshold_grid.clear()
    usable_metrics_per_sector.clear()

    # step indices like [-1, 0, +1] when STEP_PER_DIRECTION = 1
    step_idx = range(-STEP_PER_DIRECTION, STEP_PER_DIRECTION + 1)

    def vary(val):
        return [round(float(val) * (1 + STEP_SIZE * k), 6) for k in step_idx]

    for metric in metrics:
        sector_threshold_grid[metric] = {}
        for sector, seed in seeds_by_sector.get(metric, {}).items():
            # seed must be a simple pair
            if not (isinstance(seed, (list, tuple)) and len(seed) == 2):
                continue

            # normalize seed to canonical (NOK, OK) for this metric
            nok0, ok0 = normalize_threshold_tuple(metric, (seed[0], seed[1]))

            # generate variations on BOTH ends
            nok_vals = vary(nok0)
            ok_vals = vary(ok0)

            # enforce ordering per direction and deduplicate
            good_if_high = metric_direction(metric) == +1

            cands = set()
            for nv in nok_vals:
                for ov in ok_vals:
                    if good_if_high:
                        # require nok < ok; keep a small margin to avoid collapsing to neutral
                        if nv < ov:
                            cands.add((round(nv, 6), round(ov, 6)))
                    else:
                        # lower-is-better: require nok > ok
                        if nv > ov:
                            cands.add((round(nv, 6), round(ov, 6)))

            # ensure the normalized seed is present
            cands.add((round(nok0, 6), round(ok0, 6)))

            cands = sorted(cands)
            if not cands:
                continue

            sector_threshold_grid[metric][sector] = cands

            # mark metric usable for this sector
            usable_metrics_per_sector.setdefault(sector, []).append(metric)

    return sector_threshold_grid, usable_metrics_per_sector
