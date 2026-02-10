# ----------------------------------------------------------------------
#  Correlation analysis: score vs forward return
#  – Phase A: baseline report (current weights/thresholds)
#  – Phase B: weight/threshold sweep to find optimal params
# ----------------------------------------------------------------------

import sys
import os
import warnings
import json
from pathlib import Path
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from analyzer.summary_manager import SummaryManager
from analyzer.data_processing import calculate_score
from analyzer.metrics import (
    RATIO_SPECS,
    GLOBAL_THRESHOLDS,
    DIRECTION_OVERRIDES,
    HIGHEST_WEIGHT_METRICS,
    HIGH_WEIGHT_METRICS,
    LOW_WEIGHT_METRICS,
    get_metrics_threshold,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ======================================================================
# Helpers
# ======================================================================

def _all_scored_metrics():
    """Return every metric that has a weight > 0."""
    return list(
        set(HIGHEST_WEIGHT_METRICS)
        | set(HIGH_WEIGHT_METRICS)
        | set(LOW_WEIGHT_METRICS)
    )


def _load_timespan_csv(path="metrics_by_timespan.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df.dropna(subset=["total_return"], inplace=True)
    return df


def _score_snapshot(df_slice, metrics_to_score=None, thresholds=None):
    """Score a slice of historical data using SummaryManager.

    Returns a DataFrame with company index and 'points' + per-metric scores.
    """
    sm = SummaryManager()
    sm.process_historical(df_slice, metrics_to_score or _all_scored_metrics(),
                          thresholds=thresholds)
    calculate_score(sm, metrics_to_score=metrics_to_score)

    out = sm.summary
    if out is None or (isinstance(out, pd.DataFrame) and out.empty):
        out = sm.summary_investment
    if out is None or (isinstance(out, pd.DataFrame) and out.empty):
        return pd.DataFrame()

    # Merge both summaries if both have data
    frames = []
    if sm.summary is not None and not (isinstance(sm.summary, pd.DataFrame) and sm.summary.empty):
        s = sm.summary if isinstance(sm.summary, pd.DataFrame) else pd.DataFrame(sm.summary).T
        frames.append(s)
    if sm.summary_investment is not None and not (isinstance(sm.summary_investment, pd.DataFrame) and sm.summary_investment.empty):
        s = sm.summary_investment if isinstance(sm.summary_investment, pd.DataFrame) else pd.DataFrame(sm.summary_investment).T
        frames.append(s)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames)


# ======================================================================
# Phase A: Baseline correlation report
# ======================================================================

def baseline_correlation(csv_path="metrics_by_timespan.csv"):
    """Compute how well the CURRENT scoring system predicts forward returns.

    For each timespan window in the historical data:
      1. Score all companies using current weights/thresholds
      2. Correlate total score with total_return (Spearman + Pearson)
      3. Compare mean return of top-scoring vs bottom-scoring quintile

    Returns a summary DataFrame and prints a report.
    """
    df = _load_timespan_csv(csv_path)
    if df.empty:
        print("[WARN] No data found.")
        return pd.DataFrame()

    results = []

    for timespan in sorted(df["timespan"].unique()):
        df_ts = df[df["timespan"] == timespan].copy()
        if len(df_ts) < 5:
            continue

        scored = _score_snapshot(df_ts)
        if scored.empty or "points" not in scored.columns:
            continue

        # Align returns with scores
        returns = df_ts.set_index("company")["total_return"]
        scores = pd.to_numeric(scored["points"], errors="coerce")

        common = scores.index.intersection(returns.index)
        if len(common) < 5:
            continue

        s = scores.loc[common].astype(float)
        r = returns.loc[common].astype(float)
        valid = s.notna() & r.notna()
        s, r = s[valid], r[valid]

        if len(s) < 5:
            continue

        # Correlations
        pearson_r, pearson_p = sp_stats.pearsonr(s, r)
        spearman_r, spearman_p = sp_stats.spearmanr(s, r)

        # Quintile analysis
        n = len(s)
        q_size = max(n // 5, 1)
        ranked = s.sort_values(ascending=False)
        top_names = ranked.index[:q_size]
        bot_names = ranked.index[-q_size:]

        top_return = r.loc[top_names].mean()
        bot_return = r.loc[bot_names].mean()
        spread = top_return - bot_return

        # Per-metric correlation
        metric_corrs = {}
        score_cols = [c for c in scored.columns if c.endswith("_score")]
        for sc in score_cols:
            metric_name = sc.replace("_score", "")
            ms = pd.to_numeric(scored.loc[common, sc], errors="coerce")
            valid_m = ms.notna() & r.notna()
            if valid_m.sum() >= 5:
                sr, _ = sp_stats.spearmanr(ms[valid_m], r[valid_m])
                metric_corrs[metric_name] = round(sr, 4)

        results.append({
            "timespan": timespan,
            "n_companies": len(s),
            "pearson_r": round(pearson_r, 4),
            "pearson_p": round(pearson_p, 4),
            "spearman_r": round(spearman_r, 4),
            "spearman_p": round(spearman_p, 4),
            "top_quintile_return": round(top_return, 4),
            "bot_quintile_return": round(bot_return, 4),
            "spread": round(spread, 4),
            "metric_correlations": metric_corrs,
        })

    if not results:
        print("[WARN] No valid timespan windows to analyze.")
        return pd.DataFrame()

    # Build summary
    summary = pd.DataFrame(results)

    # Print report
    print("\n" + "=" * 70)
    print("  BASELINE CORRELATION REPORT")
    print("  Score vs Forward Total Return")
    print("=" * 70)

    for _, row in summary.iterrows():
        ts = row["timespan"]
        n = row["n_companies"]
        print(f"\n--- {ts} ({n} companies) ---")
        print(f"  Spearman ρ = {row['spearman_r']:+.4f}  (p={row['spearman_p']:.4f})")
        print(f"  Pearson  r = {row['pearson_r']:+.4f}  (p={row['pearson_p']:.4f})")
        print(f"  Top quintile avg return: {row['top_quintile_return']:+.2%}")
        print(f"  Bot quintile avg return: {row['bot_quintile_return']:+.2%}")
        print(f"  Spread (top - bot):      {row['spread']:+.2%}")

        mc = row.get("metric_correlations", {})
        if mc:
            print("  Per-metric Spearman with return:")
            sorted_mc = sorted(mc.items(), key=lambda x: abs(x[1]), reverse=True)
            for mname, mcorr in sorted_mc:
                arrow = "+" if mcorr > 0 else ""
                print(f"    {mname:40s} {arrow}{mcorr:.4f}")

    # Overall summary
    print("\n" + "-" * 70)
    avg_spearman = summary["spearman_r"].mean()
    avg_spread = summary["spread"].mean()
    print(f"  Avg Spearman across timespans:  {avg_spearman:+.4f}")
    print(f"  Avg spread (top - bot):         {avg_spread:+.2%}")

    if avg_spearman > 0.3:
        print("  → Strong positive correlation. Scoring system is predictive.")
    elif avg_spearman > 0.1:
        print("  → Moderate positive correlation. Some predictive value.")
    elif avg_spearman > -0.1:
        print("  → Weak/no correlation. Scoring system needs improvement.")
    else:
        print("  → Negative correlation. Scoring system is counter-predictive!")

    print("=" * 70 + "\n")

    # Save
    out_path = "correlation_baseline.csv"
    export = summary.drop(columns=["metric_correlations"], errors="ignore")
    export.to_csv(out_path, index=False)
    print(f"Saved baseline report to {out_path}")

    # Save per-metric correlations
    mc_rows = []
    for _, row in summary.iterrows():
        for m, c in row.get("metric_correlations", {}).items():
            mc_rows.append({"timespan": row["timespan"], "metric": m, "spearman_r": c})
    if mc_rows:
        mc_df = pd.DataFrame(mc_rows)
        mc_df.to_csv("correlation_per_metric.csv", index=False)
        print(f"Saved per-metric correlations to correlation_per_metric.csv")

    return summary


# ======================================================================
# Phase B: Weight / threshold optimization
# ======================================================================

WEIGHT_OPTIONS = [0.0, 1.0, 1.5, 2.0]


def _build_weight_combos(metrics, n_samples=200):
    """Generate diverse weight combinations for the given metrics.

    Full grid is too large, so we sample strategically:
    - Always include current weights as baseline
    - Random samples from the weight grid
    - Ablation: zero out one metric at a time
    """
    combos = []

    # 1. Current weights
    current = {}
    for m in metrics:
        if m in HIGHEST_WEIGHT_METRICS:
            current[m] = 2.0
        elif m in HIGH_WEIGHT_METRICS:
            current[m] = 1.5
        elif m in LOW_WEIGHT_METRICS:
            current[m] = 1.0
        else:
            current[m] = 0.0
    combos.append(current)

    # 2. Ablation: drop one metric at a time
    for drop in metrics:
        c = current.copy()
        c[drop] = 0.0
        combos.append(c)

    # 3. Promotion: bump each metric up one tier
    for bump in metrics:
        c = current.copy()
        if c[bump] < 2.0:
            c[bump] = min(c[bump] + 0.5, 2.0)
            combos.append(c)

    # 4. Random samples
    rng = np.random.default_rng(42)
    for _ in range(n_samples - len(combos)):
        c = {m: float(rng.choice(WEIGHT_OPTIONS)) for m in metrics}
        # Ensure at least some metrics are nonzero
        if sum(c.values()) == 0:
            c[rng.choice(metrics)] = 1.0
        combos.append(c)

    return combos[:n_samples]


def _threshold_variations(metric, base_thr, steps=3, step_pct=0.15):
    """Generate threshold variations around a base (nok, ok) pair."""
    nok0, ok0 = float(base_thr[0]), float(base_thr[1])

    # Determine direction
    if metric in RATIO_SPECS:
        direction = RATIO_SPECS[metric]["dir"]
    else:
        direction = DIRECTION_OVERRIDES.get(metric, +1)

    variations = set()
    for i in range(-steps, steps + 1):
        for j in range(-steps, steps + 1):
            nok = round(nok0 * (1 + step_pct * i), 6)
            ok = round(ok0 * (1 + step_pct * j), 6)
            # Enforce ordering
            if direction == +1 and nok < ok:
                variations.add((nok, ok))
            elif direction == -1 and nok > ok:
                variations.add((nok, ok))

    # Always include base
    variations.add((nok0, ok0))
    return list(variations)


def optimize_weights_and_thresholds(
    csv_path="metrics_by_timespan.csv",
    n_weight_samples=100,
    threshold_steps=2,
    target_timespans=None,
):
    """Sweep weight combinations and threshold variations to maximize
    score-return correlation.

    Strategy:
    1. First optimize weights with current thresholds (fast)
    2. Then optimize thresholds with best weights (per-metric)
    3. Report best combination

    Returns dict with optimal params and correlation stats.
    """
    df = _load_timespan_csv(csv_path)
    if df.empty:
        print("[WARN] No data.")
        return {}

    metrics = _all_scored_metrics()

    # Filter to target timespans (default: TOTAL windows for less noise)
    if target_timespans is None:
        target_timespans = [t for t in df["timespan"].unique() if "TOTAL" in str(t)]
    if not target_timespans:
        target_timespans = list(df["timespan"].unique())

    df = df[df["timespan"].isin(target_timespans)]

    print(f"\n[OPTIMIZE] {len(df)} rows, {len(metrics)} metrics, "
          f"timespans: {target_timespans}")

    # ---- Step 1: Weight sweep with current thresholds ----
    print("\n[Step 1] Sweeping weight combinations...")
    weight_combos = _build_weight_combos(metrics, n_samples=n_weight_samples)

    best_weight_corr = -999
    best_weights = None
    weight_results = []

    for i, weights in enumerate(weight_combos):
        # Temporarily override HIGHEST/HIGH/LOW by patching SummaryManager
        corrs = []
        for timespan in target_timespans:
            df_ts = df[df["timespan"] == timespan].copy()
            if len(df_ts) < 5:
                continue

            sm = SummaryManager()
            # Inject weight overrides
            sm._weight_overrides = weights
            sm.process_historical(df_ts, metrics)
            calculate_score(sm, metrics_to_score=metrics)

            scored = sm.summary
            if scored is None or (isinstance(scored, pd.DataFrame) and scored.empty):
                scored = sm.summary_investment
            if scored is None or (isinstance(scored, pd.DataFrame) and scored.empty):
                continue
            if isinstance(scored, dict):
                scored = pd.DataFrame(scored).T

            returns = df_ts.set_index("company")["total_return"]
            scores = pd.to_numeric(scored.get("points", pd.Series(dtype=float)),
                                   errors="coerce")
            common = scores.index.intersection(returns.index)
            if len(common) < 5:
                continue

            s = scores.loc[common].astype(float)
            r = returns.loc[common].astype(float)
            valid = s.notna() & r.notna()
            if valid.sum() < 5:
                continue

            rho, _ = sp_stats.spearmanr(s[valid], r[valid])
            if not np.isnan(rho):
                corrs.append(rho)

        avg_corr = np.mean(corrs) if corrs else -999
        weight_results.append({"weights": weights, "avg_spearman": avg_corr})

        if avg_corr > best_weight_corr:
            best_weight_corr = avg_corr
            best_weights = weights

        if (i + 1) % 25 == 0:
            print(f"  ... {i+1}/{len(weight_combos)} tested, "
                  f"best so far: ρ={best_weight_corr:+.4f}")

    print(f"\n[Step 1] Best weight combo: ρ={best_weight_corr:+.4f}")
    if best_weights:
        nonzero = {k: v for k, v in best_weights.items() if v > 0}
        print(f"  Active metrics ({len(nonzero)}): {nonzero}")

    # ---- Step 2: Threshold sweep with best weights ----
    print("\n[Step 2] Sweeping thresholds per metric...")

    best_thresholds = {}
    for metric in metrics:
        # Get base threshold
        if metric in RATIO_SPECS:
            base = RATIO_SPECS[metric]["thr"]
        elif metric in GLOBAL_THRESHOLDS:
            base = GLOBAL_THRESHOLDS[metric]
        else:
            continue

        if best_weights and best_weights.get(metric, 0) == 0:
            continue  # Skip metrics with zero weight

        variations = _threshold_variations(metric, base, steps=threshold_steps)
        best_thr_corr = -999
        best_thr = base

        for thr in variations:
            overrides = {metric: {"nok": thr[0], "ok": thr[1]}}
            corrs = []

            for timespan in target_timespans:
                df_ts = df[df["timespan"] == timespan].copy()
                if len(df_ts) < 5:
                    continue

                sm = SummaryManager()
                if best_weights:
                    sm._weight_overrides = best_weights
                sm.process_historical(df_ts, [metric], thresholds=overrides)
                calculate_score(sm, metrics_to_score=[metric])

                scored = sm.summary
                if scored is None or (isinstance(scored, pd.DataFrame) and scored.empty):
                    scored = sm.summary_investment
                if scored is None or (isinstance(scored, pd.DataFrame) and scored.empty):
                    continue
                if isinstance(scored, dict):
                    scored = pd.DataFrame(scored).T

                returns = df_ts.set_index("company")["total_return"]
                scores = pd.to_numeric(scored.get("points", pd.Series(dtype=float)),
                                       errors="coerce")
                common = scores.index.intersection(returns.index)
                if len(common) < 5:
                    continue

                s = scores.loc[common].astype(float)
                r = returns.loc[common].astype(float)
                valid = s.notna() & r.notna()
                if valid.sum() < 5:
                    continue

                rho, _ = sp_stats.spearmanr(s[valid], r[valid])
                if not np.isnan(rho):
                    corrs.append(rho)

            avg = np.mean(corrs) if corrs else -999
            if avg > best_thr_corr:
                best_thr_corr = avg
                best_thr = thr

        best_thresholds[metric] = {
            "nok": best_thr[0],
            "ok": best_thr[1],
            "spearman": round(best_thr_corr, 4),
        }

    # ---- Report ----
    print("\n" + "=" * 70)
    print("  OPTIMIZATION RESULTS")
    print("=" * 70)

    print("\nOptimal weights:")
    if best_weights:
        for m in sorted(best_weights, key=lambda x: best_weights[x], reverse=True):
            w = best_weights[m]
            if w > 0:
                print(f"  {m:40s}  weight={w:.1f}")

    print("\nOptimal thresholds:")
    for m, t in sorted(best_thresholds.items()):
        print(f"  {m:40s}  nok={t['nok']:.4f}  ok={t['ok']:.4f}  ρ={t['spearman']:+.4f}")

    print(f"\nOverall best Spearman: {best_weight_corr:+.4f}")
    print("=" * 70)

    # Save results
    result = {
        "best_weights": best_weights,
        "best_thresholds": best_thresholds,
        "best_spearman": best_weight_corr,
    }

    # Save to JSON
    out_path = "optimization_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved optimization results to {out_path}")

    # Save weight sweep results
    wr_df = pd.DataFrame([
        {"avg_spearman": r["avg_spearman"], **r["weights"]}
        for r in weight_results
    ]).sort_values("avg_spearman", ascending=False)
    wr_df.to_csv("weight_sweep_results.csv", index=False)
    print(f"Saved weight sweep to weight_sweep_results.csv")

    return result


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Score-return correlation analysis")
    ap.add_argument("--baseline", action="store_true",
                    help="Run baseline correlation report")
    ap.add_argument("--optimize", action="store_true",
                    help="Run weight/threshold optimization sweep")
    ap.add_argument("--csv", default="metrics_by_timespan.csv",
                    help="Path to metrics_by_timespan.csv")
    ap.add_argument("--weight-samples", type=int, default=100,
                    help="Number of weight combos to try")
    ap.add_argument("--threshold-steps", type=int, default=2,
                    help="Steps per direction for threshold variation")
    args = ap.parse_args()

    if args.baseline:
        baseline_correlation(args.csv)
    elif args.optimize:
        optimize_weights_and_thresholds(
            csv_path=args.csv,
            n_weight_samples=args.weight_samples,
            threshold_steps=args.threshold_steps,
        )
    else:
        # Default: run baseline first
        print("Running baseline correlation analysis...")
        print("(Use --optimize for weight/threshold sweep)")
        baseline_correlation(args.csv)
