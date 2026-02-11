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
    df.columns = df.columns.str.strip()
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
# Phase B: Correlation-based weight optimization
# ======================================================================

def optimize_weights_and_thresholds(
    csv_path="metrics_by_timespan.csv",
    target_timespans=None,
    **_kwargs,
):
    """Assign weights proportional to each metric's Spearman correlation
    with forward returns. Simple and robust — avoids overfitting.

    Strategy:
    1. Compute per-metric Spearman correlation across TOTAL windows
    2. Drop metrics with negative or near-zero correlation (weight=0)
    3. Scale positive correlations to weights in [0, 2]
    4. Re-score with optimized weights and report improvement
    5. Compute per-company fundamental reliability score

    Returns dict with optimal weights and reliability scores.
    """
    df = _load_timespan_csv(csv_path)
    if df.empty:
        print("[WARN] No data.")
        return {}

    metrics = _all_scored_metrics()

    # Use TOTAL windows (most relevant for 3-5 year horizon)
    if target_timespans is None:
        target_timespans = [t for t in df["timespan"].unique() if "TOTAL" in str(t)]
    if not target_timespans:
        target_timespans = list(df["timespan"].unique())

    df_total = df[df["timespan"].isin(target_timespans)]

    print(f"\n[OPTIMIZE] {len(df_total)} rows, {len(metrics)} metrics, "
          f"timespans: {target_timespans}")

    # ---- Step 1: Per-metric Spearman correlation ----
    print("\n[Step 1] Computing per-metric correlations...")

    metric_corrs = {}  # metric -> list of (rho, n) across timespans

    for timespan in target_timespans:
        df_ts = df_total[df_total["timespan"] == timespan].copy()
        if len(df_ts) < 5:
            continue

        scored = _score_snapshot(df_ts)
        if scored.empty:
            continue

        returns = df_ts.set_index("company")["total_return"]
        score_cols = [c for c in scored.columns if c.endswith("_score")]

        for sc in score_cols:
            metric_name = sc.replace("_score", "")
            if metric_name not in metrics:
                continue

            ms = pd.to_numeric(scored.get(sc, pd.Series(dtype=float)), errors="coerce")
            common = ms.index.intersection(returns.index)
            if len(common) < 5:
                continue

            m_vals = ms.loc[common].astype(float)
            r_vals = returns.loc[common].astype(float)
            valid = m_vals.notna() & r_vals.notna()
            if valid.sum() < 5:
                continue

            rho, pval = sp_stats.spearmanr(m_vals[valid], r_vals[valid])
            if not np.isnan(rho):
                metric_corrs.setdefault(metric_name, []).append(
                    {"rho": rho, "p": pval, "n": int(valid.sum()), "ts": timespan}
                )

    # ---- Step 2: Assign weights from correlations ----
    print("\n[Step 2] Assigning correlation-based weights...")

    avg_corrs = {}
    for m, entries in metric_corrs.items():
        rhos = [e["rho"] for e in entries]
        avg_corrs[m] = np.mean(rhos)

    # Only keep metrics with positive average correlation
    positive_metrics = {m: r for m, r in avg_corrs.items() if r > 0.02}

    if not positive_metrics:
        print("[WARN] No metrics with positive correlation found.")
        return {}

    # Scale to [0, 2] range proportional to correlation strength
    max_corr = max(positive_metrics.values())
    optimized_weights = {}
    for m in metrics:
        if m in positive_metrics:
            # Scale: strongest metric gets 2.0, others proportionally
            raw = positive_metrics[m] / max_corr * 2.0
            # Round to nearest 0.25 for cleaner weights
            optimized_weights[m] = round(raw * 4) / 4
        else:
            optimized_weights[m] = 0.0

    # ---- Step 3: Re-score with optimized weights ----
    print("\n[Step 3] Re-scoring with optimized weights...")

    baseline_corrs = []
    optimized_corrs = []

    for timespan in target_timespans:
        df_ts = df_total[df_total["timespan"] == timespan].copy()
        if len(df_ts) < 5:
            continue

        returns = df_ts.set_index("company")["total_return"]

        # Baseline score
        scored_base = _score_snapshot(df_ts)
        if not scored_base.empty and "points" in scored_base.columns:
            s = pd.to_numeric(scored_base["points"], errors="coerce")
            common = s.index.intersection(returns.index)
            if len(common) >= 5:
                sv = s.loc[common].astype(float)
                rv = returns.loc[common].astype(float)
                valid = sv.notna() & rv.notna()
                if valid.sum() >= 5:
                    rho, _ = sp_stats.spearmanr(sv[valid], rv[valid])
                    if not np.isnan(rho):
                        baseline_corrs.append(rho)

        # Optimized score
        sm = SummaryManager()
        sm._weight_overrides = optimized_weights
        sm.process_historical(df_ts, metrics)
        calculate_score(sm, metrics_to_score=metrics)

        scored_opt = sm.summary
        if scored_opt is None or (isinstance(scored_opt, pd.DataFrame) and scored_opt.empty):
            scored_opt = sm.summary_investment
        if scored_opt is None or (isinstance(scored_opt, pd.DataFrame) and scored_opt.empty):
            continue
        if isinstance(scored_opt, dict):
            scored_opt = pd.DataFrame(scored_opt).T

        s = pd.to_numeric(scored_opt.get("points", pd.Series(dtype=float)), errors="coerce")
        common = s.index.intersection(returns.index)
        if len(common) >= 5:
            sv = s.loc[common].astype(float)
            rv = returns.loc[common].astype(float)
            valid = sv.notna() & rv.notna()
            if valid.sum() >= 5:
                rho, _ = sp_stats.spearmanr(sv[valid], rv[valid])
                if not np.isnan(rho):
                    optimized_corrs.append(rho)

    avg_baseline = np.mean(baseline_corrs) if baseline_corrs else 0
    avg_optimized = np.mean(optimized_corrs) if optimized_corrs else 0

    # ---- Step 4: Per-company fundamental reliability ----
    print("\n[Step 4] Computing per-company fundamental reliability...")

    reliability = _compute_reliability(df, target_timespans)

    # ---- Report ----
    print("\n" + "=" * 70)
    print("  OPTIMIZATION RESULTS")
    print("=" * 70)

    print("\nPer-metric avg Spearman correlation:")
    for m in sorted(avg_corrs, key=lambda x: avg_corrs[x], reverse=True):
        r = avg_corrs[m]
        w = optimized_weights.get(m, 0)
        status = "KEEP" if w > 0 else "DROP"
        print(f"  {m:40s}  ρ={r:+.4f}  weight={w:.2f}  [{status}]")

    # Show metrics with no data at all
    no_data = [m for m in metrics if m not in avg_corrs]
    if no_data:
        print(f"\n  No data available for: {', '.join(no_data)}")

    print(f"\n  Baseline avg Spearman:   {avg_baseline:+.4f}")
    print(f"  Optimized avg Spearman:  {avg_optimized:+.4f}")
    improvement = avg_optimized - avg_baseline
    print(f"  Improvement:             {improvement:+.4f}")

    if reliability is not None and not reliability.empty:
        print("\n" + "-" * 70)
        print("  PER-COMPANY FUNDAMENTAL RELIABILITY")
        print("  (How well does this company's score predict its returns?)")
        print("-" * 70)
        for _, row in reliability.head(20).iterrows():
            label = "RELIABLE" if row["reliable"] else "UNRELIABLE"
            print(f"  {row['company']:40s}  ρ={row['spearman']:+.4f}  "
                  f"n={row['n_windows']:2d}  [{label}]")
        print(f"  ... ({len(reliability)} companies total)")

        n_reliable = reliability["reliable"].sum()
        print(f"\n  Reliable companies: {n_reliable}/{len(reliability)}")

    print("=" * 70)

    # Save results
    result = {
        "optimized_weights": optimized_weights,
        "per_metric_correlations": {m: round(r, 4) for m, r in avg_corrs.items()},
        "baseline_spearman": round(avg_baseline, 4),
        "optimized_spearman": round(avg_optimized, 4),
    }

    out_path = "optimization_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved optimization results to {out_path}")

    if reliability is not None and not reliability.empty:
        reliability.to_csv("company_reliability.csv", index=False)
        print(f"Saved company reliability to company_reliability.csv")

    return result


# ======================================================================
# Per-company fundamental reliability
# ======================================================================

def _compute_reliability(df, target_timespans):
    """For each company, compute how well its fundamental score
    correlates with its returns across different time windows.

    A "reliable" company is one where good fundamentals actually
    translate into good returns. An "unreliable" company (like PayPal)
    has good fundamentals but disconnected price performance.

    Returns DataFrame with columns: company, spearman, n_windows, reliable
    """
    # Collect (score, return) pairs per company across all timespans
    company_pairs = defaultdict(lambda: {"scores": [], "returns": []})

    all_timespans = sorted(df["timespan"].unique())

    for timespan in all_timespans:
        df_ts = df[df["timespan"] == timespan].copy()
        if len(df_ts) < 5:
            continue

        scored = _score_snapshot(df_ts)
        if scored.empty or "points" not in scored.columns:
            continue

        returns = df_ts.set_index("company")["total_return"]
        scores = pd.to_numeric(scored["points"], errors="coerce")
        common = scores.index.intersection(returns.index)

        for company in common:
            s_val = scores.get(company)
            r_val = returns.get(company)
            if s_val is not None and r_val is not None:
                try:
                    s_f = float(s_val)
                    r_f = float(r_val)
                    if not (np.isnan(s_f) or np.isnan(r_f)):
                        company_pairs[company]["scores"].append(s_f)
                        company_pairs[company]["returns"].append(r_f)
                except (TypeError, ValueError):
                    pass

    # Compute per-company correlation
    rows = []
    for company, data in company_pairs.items():
        n = len(data["scores"])
        if n < 3:
            # Not enough windows to compute meaningful correlation
            rows.append({
                "company": company,
                "spearman": np.nan,
                "n_windows": n,
                "reliable": False,
            })
            continue

        rho, pval = sp_stats.spearmanr(data["scores"], data["returns"])
        # "Reliable" = positive correlation with p < 0.3
        # (lenient threshold because we have few data points per company)
        reliable = (not np.isnan(rho)) and rho > 0.1 and pval < 0.3
        rows.append({
            "company": company,
            "spearman": round(rho, 4) if not np.isnan(rho) else np.nan,
            "n_windows": n,
            "reliable": reliable,
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values("spearman", ascending=False, na_position="last")
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
                    help="Run correlation-based weight optimization")
    ap.add_argument("--csv", default="metrics_by_timespan.csv",
                    help="Path to metrics_by_timespan.csv")
    args = ap.parse_args()

    if args.baseline:
        baseline_correlation(args.csv)
    elif args.optimize:
        optimize_weights_and_thresholds(csv_path=args.csv)
    else:
        print("Running baseline correlation analysis...")
        print("(Use --optimize for weight optimization + reliability scoring)")
        baseline_correlation(args.csv)
