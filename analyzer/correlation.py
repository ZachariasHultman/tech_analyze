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
from scipy import optimize as sp_optimize

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from analyzer.summary_manager import SummaryManager
from analyzer.data_processing import calculate_score
from analyzer.config import QUALITY_METRICS
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
# Module-level constants (shared across both optimizers)
# ======================================================================

# Metrics whose correlation with returns is partly circular (price-derived,
# not fundamentals). Cap their weight so fundamentals remain the primary driver.
MOMENTUM_METRICS = {"price momentum status"}
MOMENTUM_WEIGHT_CAP = 1.0

# Minimum weight floors for academically proven metrics. These may show
# weak/negative correlation due to data-quality issues in the historical
# adapter, but are well-established in research. "earnings quality status"
# excluded: OCF is never populated in either the live or historical path, so
# this metric is permanently 0. Re-add if OCF wiring is done later.
WEIGHT_FLOORS = {
    "piotroski f-score status": 0.5,
    "dividend yield status": 0.25,
}

# Minimum separation enforced between a metric's nok/ok thresholds whenever
# a search (grid or Nelder-Mead) proposes a pair that violates direction
# ordering (nok < ok for direction=+1, nok > ok for direction=-1). Kept as
# one named constant rather than a hardcoded literal repeated at each
# enforcement site, so the value only needs to be reasoned about once.
THRESHOLD_ORDER_EPS = 1e-4


# ======================================================================
# Helpers
# ======================================================================


def _quintile_spread(scores, returns):
    """Long-short decision-relevance metric: mean(top bucket) - mean(bottom bucket).

    Sort by score, take the top and bottom buckets, return the difference in
    their mean forward returns. Uses quintiles when n >= 25, terciles below
    that (terciles need ~6 points for 2 per bucket). Returns np.nan when there
    are too few points to split at all (n < 6).
    """
    s = pd.Series(list(scores) if not isinstance(scores, pd.Series) else scores).astype(float)
    r = pd.Series(list(returns) if not isinstance(returns, pd.Series) else returns).astype(float)
    if len(s) != len(r):
        return np.nan
    # Align positionally when indices differ (e.g. plain lists/arrays)
    if not s.index.equals(r.index):
        s = s.reset_index(drop=True)
        r = r.reset_index(drop=True)
    valid = s.notna() & r.notna()
    s, r = s[valid], r[valid]
    n = len(s)
    if n < 6:
        return np.nan
    frac = 5 if n >= 25 else 3
    q = max(n // frac, 1)
    order = s.sort_values(ascending=False)
    top = r.loc[order.index[:q]].mean()
    bot = r.loc[order.index[-q:]].mean()
    return float(top - bot)

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
        # Drop columns that are entirely NA to avoid FutureWarning on concat
        s = s.dropna(axis=1, how="all")
        if not s.empty:
            frames.append(s)
    if sm.summary_investment is not None and not (isinstance(sm.summary_investment, pd.DataFrame) and sm.summary_investment.empty):
        s = sm.summary_investment if isinstance(sm.summary_investment, pd.DataFrame) else pd.DataFrame(sm.summary_investment).T
        s = s.dropna(axis=1, how="all")
        if not s.empty:
            frames.append(s)

    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]
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

    n_companies = df["company"].nunique()
    if n_companies < 30:
        print(f"\n[WARN] Only {n_companies} unique companies in historical data.")
        print("  Spearman correlations are noisy below ~30 companies.")
        print("  Use --watchlists to include more Avanza watchlists and grow the universe.\n")

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

        # Quality-sleeve score vs forward-fundamentals target. Validates
        # "is this a solid company?" separately from price: does a high
        # quality score line up with fundamentals that actually held up
        # over the window?
        quality_fwd_sp = None
        q_cols = [m + "_score" for m in QUALITY_METRICS if m + "_score" in scored.columns]
        if q_cols and "fund_forward_score" in df_ts.columns:
            qscore = scored[q_cols].sum(axis=1)
            fwd = df_ts.set_index("company")["fund_forward_score"]
            cq = qscore.index.intersection(fwd.index)
            qs = pd.to_numeric(qscore.loc[cq], errors="coerce")
            fs = pd.to_numeric(fwd.loc[cq], errors="coerce")
            valid_q = qs.notna() & fs.notna()
            if valid_q.sum() >= 5:
                qsr, _ = sp_stats.spearmanr(qs[valid_q], fs[valid_q])
                if not np.isnan(qsr):
                    quality_fwd_sp = round(qsr, 4)

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
            "quality_fwd_spearman": quality_fwd_sp,
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
        qfs = row.get("quality_fwd_spearman")
        if qfs is not None:
            print(f"  Quality vs fwd-fundamentals ρ = {qfs:+.4f}")

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
    if "quality_fwd_spearman" in summary.columns:
        avg_qfs = pd.to_numeric(summary["quality_fwd_spearman"], errors="coerce").mean()
        if not pd.isna(avg_qfs):
            print(f"  Avg quality vs fwd-fundamentals ρ: {avg_qfs:+.4f}")

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

    # Objective windows: 3Y/5Y TOTAL only. These are the cumulative multi-year
    # windows the scoring system is meant to predict; the YoY sub-windows are
    # reserved for the per-company reliability calc (item 3).
    if target_timespans is None:
        all_ts = df["timespan"].unique()
        target_timespans = [
            t for t in all_ts
            if "TOTAL" in str(t) and ("3Y" in str(t) or "5Y" in str(t))
        ]
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

    # MOMENTUM_METRICS / MOMENTUM_WEIGHT_CAP / WEIGHT_FLOORS are module-level
    # constants (shared with combo), defined near the top of this file.

    # Scale to [0, 2] range proportional to correlation strength
    # Exclude momentum from max_corr so fundamental metrics set the scale
    fundamental_corrs = {m: r for m, r in positive_metrics.items() if m not in MOMENTUM_METRICS}
    max_corr = max(fundamental_corrs.values()) if fundamental_corrs else max(positive_metrics.values())

    optimized_weights = {}
    for m in metrics:
        if m in positive_metrics:
            # Scale: strongest fundamental metric gets 2.0, others proportionally
            raw = positive_metrics[m] / max_corr * 2.0
            # Round to nearest 0.25 for cleaner weights
            w = round(raw * 4) / 4
            # Cap momentum-like metrics
            if m in MOMENTUM_METRICS:
                w = min(w, MOMENTUM_WEIGHT_CAP)
            optimized_weights[m] = max(w, WEIGHT_FLOORS.get(m, 0.0))
        else:
            # Apply floor even if correlation was negative/zero
            optimized_weights[m] = WEIGHT_FLOORS.get(m, 0.0)

    # ---- Step 3: Optimize thresholds per-metric ----
    print("\n[Step 3] Optimizing thresholds per-metric...")

    default_thresholds = _get_default_thresholds()
    # Always start the grid search from the fixed defaults in metrics.py, not
    # the previous run's result. Warm-starting from prior output compounds:
    # _threshold_grid_for_metric's search always lands on the edge of its
    # +/-2-step window (n_steps=2, step = 0.3*span, so max shift = 0.6*span),
    # so re-running --optimize repeatedly walked a threshold arbitrarily far
    # in one direction with no bound and no re-anchoring -- confirmed on real
    # data: roe_pe ratio status's "ok" threshold moved by exactly the same
    # -0.6*span delta on every successive run (+0.07 -> -0.71 -> -1.1 -> ...),
    # not converging, just accumulating noise from a small (~127-company)
    # sample. Each run now re-derives thresholds fresh from the same stable
    # baseline, so repeated runs on similar data give similar results
    # instead of drifting further every time.
    optimized_thresholds = dict(default_thresholds)

    for m in metrics:
        if optimized_weights.get(m, 0) == 0:
            continue  # skip dropped metrics
        if m not in optimized_thresholds:
            if m in default_thresholds:
                optimized_thresholds[m] = default_thresholds[m]
            else:
                continue

        cur = optimized_thresholds[m]
        candidates = _threshold_grid_for_metric(m, cur["nok"], cur["ok"], n_steps=2)

        best_thr = cur
        best_spread = -np.inf

        for cand in candidates:
            trial_thr = dict(optimized_thresholds)
            trial_thr[m] = cand
            # Objective: quintile long-short spread (decision-relevant)
            spread = _avg_quintile_spread_across_windows(
                optimized_weights, df_total, target_timespans, metrics, trial_thr
            )
            if spread > best_spread:
                best_spread = spread
                best_thr = cand

        optimized_thresholds[m] = best_thr

    # ---- Step 4: Re-score with optimized weights + thresholds ----
    print("\n[Step 4] Re-scoring with optimized weights and thresholds...")

    baseline_weights = {
        m: (2.0 if m in HIGHEST_WEIGHT_METRICS else 1.5 if m in HIGH_WEIGHT_METRICS else 1.0)
        for m in metrics
    }
    # Primary objective: quintile spread. Spearman kept as a diagnostic.
    spread_baseline = _avg_quintile_spread_across_windows(
        baseline_weights, df_total, target_timespans, metrics
    )
    spread_optimized = _avg_quintile_spread_across_windows(
        optimized_weights, df_total, target_timespans, metrics, optimized_thresholds
    )
    avg_baseline = _avg_spearman_across_windows(
        baseline_weights, df_total, target_timespans, metrics
    )
    avg_optimized = _avg_spearman_across_windows(
        optimized_weights, df_total, target_timespans, metrics, optimized_thresholds
    )

    # ---- Step 5: Per-company fundamental reliability ----
    print("\n[Step 5] Computing per-company fundamental reliability...")

    # Reliability uses only the non-overlapping 5Y_YoY-* windows (3Y_YoY-k and
    # 5Y_YoY-k are the same calendar window; TOTAL windows overlap each other),
    # so each company's sample is genuinely independent (n capped at 5).
    yoy_timespans = [t for t in df["timespan"].unique() if str(t).startswith("5Y_YoY-")]
    reliability = _compute_reliability(df, yoy_timespans)

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

    print(f"\n  [objective] Baseline avg quintile spread:   {spread_baseline:+.4f}")
    print(f"  [objective] Optimized avg quintile spread:  {spread_optimized:+.4f}")
    print(f"  [objective] Improvement:                    {spread_optimized - spread_baseline:+.4f}")
    print(f"\n  [diagnostic] Baseline avg Spearman:   {avg_baseline:+.4f}")
    print(f"  [diagnostic] Optimized avg Spearman:  {avg_optimized:+.4f}")
    print(f"  [diagnostic] Improvement:             {avg_optimized - avg_baseline:+.4f}")

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

    # Report threshold changes
    print("\nThreshold changes:")
    for m in sorted(optimized_thresholds):
        old = default_thresholds.get(m)
        new = optimized_thresholds[m]
        if old and (old["nok"] != new["nok"] or old["ok"] != new["ok"]):
            print(f"  {m:40s}  ({old['nok']}, {old['ok']}) → ({new['nok']}, {new['ok']})")

    # Save results
    # Convert thresholds to serializable format (nok, ok) tuples
    thr_serializable = {m: {"nok": t["nok"], "ok": t["ok"]}
                        for m, t in optimized_thresholds.items()}
    result = {
        "optimized_weights": optimized_weights,
        "optimized_thresholds": thr_serializable,
        "per_metric_correlations": {m: round(r, 4) for m, r in avg_corrs.items()},
        "baseline_spread": round(spread_baseline, 4),
        "optimized_spread": round(spread_optimized, 4),
        "baseline_spearman": round(avg_baseline, 4),
        "optimized_spearman": round(avg_optimized, 4),
    }

    out_path = "optimization_results_individual.json"
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

    Returns DataFrame with columns:
        company, spearman, spearman_shrunk, n_windows, reliable
    """
    # Collect (score, return) pairs per company across the restricted windows.
    # target_timespans is now honored (previously a dead parameter — the
    # function recomputed its own list). Callers pass only the non-overlapping
    # 5Y_YoY-* windows so each per-company sample is genuinely independent.
    company_pairs = defaultdict(lambda: {"scores": [], "returns": []})

    if target_timespans is not None:
        all_timespans = sorted(target_timespans)
    else:
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
            # 1-2 points can't produce a meaningful correlation at all.
            rows.append({
                "company": company,
                "spearman": np.nan,
                "spearman_shrunk": np.nan,
                "n_windows": n,
                "reliable": False,
            })
            continue

        rho, pval = sp_stats.spearmanr(data["scores"], data["returns"])
        # Smooth shrinkage toward 0 instead of a hard n-cutoff: with the
        # window restriction n is capped at 5, so any hard gate above n=5
        # would null every company. spearman_shrunk = rho * n/(n+10) keeps
        # small-sample estimates conservative (n=5 → factor ~0.33) without
        # discarding them. `reliable` stays as an informational rho>0.4
        # label but no longer gates anything downstream.
        if np.isnan(rho):
            shrunk = np.nan
        else:
            shrunk = rho * n / (n + 10)
        reliable = (not np.isnan(rho)) and rho > 0.4
        rows.append({
            "company": company,
            "spearman": round(rho, 4) if not np.isnan(rho) else np.nan,
            "spearman_shrunk": round(shrunk, 4) if not np.isnan(shrunk) else np.nan,
            "n_windows": n,
            "reliable": reliable,
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values(
        "spearman_shrunk", ascending=False, na_position="last"
    )
    return result


# ======================================================================
# Shared helpers for combo optimization
# ======================================================================


def _apply_weight_constraints(weights_dict):
    """Enforce momentum cap and weight floors, clamp to [0, 2]."""
    for m, w in weights_dict.items():
        w = max(0.0, min(2.0, w))
        if m in MOMENTUM_METRICS:
            w = min(w, MOMENTUM_WEIGHT_CAP)
        w = max(w, WEIGHT_FLOORS.get(m, 0.0))
        weights_dict[m] = round(w * 4) / 4  # snap to 0.25
    return weights_dict


def _get_default_thresholds():
    """Return the current hardcoded thresholds as {metric: {"nok": x, "ok": y}}."""
    thresholds = {}
    for m, spec in RATIO_SPECS.items():
        thresholds[m] = {"nok": spec["thr"][0], "ok": spec["thr"][1]}
    for m, (nok, ok) in GLOBAL_THRESHOLDS.items():
        if nok is not None and ok is not None:
            thresholds[m] = {"nok": nok, "ok": ok}
    return thresholds


def _refine_threshold_2d(metric, weights_dict, full_thresholds, df_total,
                          target_timespans, metrics):
    """Locally optimize one metric's (nok, ok) via bounded 2D Nelder-Mead,
    holding weights and every other metric's threshold fixed.

    A well-scoped 2-parameter subproblem, called once per metric per
    coordinate-descent round in optimize_combo. Unlike the abandoned
    60-dimensional joint approach (see removed optimize_stepwise),
    Nelder-Mead is well-suited at this size -- and it replaces the old
    fixed-step grid (_threshold_grid_for_metric) with a real local search,
    so it can land on precise values instead of only fixed multiples of
    0.3*span.

    Bounded to the metric's own observed value range (2nd-98th percentile,
    +50% margin) so the search can't wander to a degenerate threshold that
    trivially passes/fails every company regardless of merit -- the exact
    failure mode behind the threshold-drift bug fixed in
    optimize_weights_and_thresholds (an unbounded search will find these
    looking spuriously better on a small, noisy sample).

    Returns (best_threshold_dict, best_cv) if refinement ran, or
    (current_threshold, None) if there wasn't enough data to bound the
    search (caller should treat None as "no candidate, keep current").
    """
    direction = DIRECTION_OVERRIDES.get(metric, +1)
    if metric in RATIO_SPECS:
        direction = RATIO_SPECS[metric]["dir"]

    cur = full_thresholds[metric]
    vals = pd.to_numeric(df_total.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
    if vals.empty:
        return cur, None
    lo, hi = float(vals.quantile(0.02)), float(vals.quantile(0.98))
    margin = max((hi - lo) * 0.5, 1e-6)
    bound = (lo - margin, hi + margin)

    def objective(x):
        nok, ok = float(x[0]), float(x[1])
        if direction == +1 and nok >= ok:
            ok = nok + THRESHOLD_ORDER_EPS
        elif direction == -1 and nok <= ok:
            nok = ok + THRESHOLD_ORDER_EPS
        trial = dict(full_thresholds)
        trial[metric] = {"nok": round(nok, 4), "ok": round(ok, 4)}
        return -_cv_score(weights_dict, df_total, target_timespans, metrics, trial)

    result = sp_optimize.minimize(
        objective,
        x0=np.array([cur["nok"], cur["ok"]], dtype=float),
        method="Nelder-Mead",
        bounds=[bound, bound],
        options={"maxiter": 25, "maxfev": 25, "xatol": 0.01, "fatol": 1e-4},
    )

    # Clip to bounds first, then fix ordering with an epsilon capped to the
    # available room -- doing this in the other order can itself push a
    # value back outside the bound (e.g. nok pinned at the upper edge, then
    # "ok = nok + THRESHOLD_ORDER_EPS" tips ok just past it). Caught by
    # test_refine_threshold_2d_respects_bounds_even_with_pathological_objective.
    lo_b, hi_b = bound
    nok = min(max(float(result.x[0]), lo_b), hi_b)
    ok = min(max(float(result.x[1]), lo_b), hi_b)
    if (direction == +1 and nok >= ok) or (direction == -1 and nok <= ok):
        mid = (nok + ok) / 2
        eps = min(THRESHOLD_ORDER_EPS, (hi_b - lo_b) / 2)
        nok, ok = (mid - eps, mid + eps) if direction == +1 else (mid + eps, mid - eps)
        nok = min(max(nok, lo_b), hi_b)
        ok = min(max(ok, lo_b), hi_b)
    return {"nok": round(nok, 4), "ok": round(ok, 4)}, -float(result.fun)


def _threshold_grid_for_metric(metric, current_nok, current_ok, n_steps=3):
    """Generate candidate (nok, ok) pairs around current thresholds.

    For direction=+1: nok < ok, so we shift both and ensure nok < ok.
    For direction=-1: nok > ok (e.g. gross margin stability), same logic.

    Returns list of {"nok": x, "ok": y} dicts.
    """
    direction = DIRECTION_OVERRIDES.get(metric, +1)
    if metric in RATIO_SPECS:
        direction = RATIO_SPECS[metric]["dir"]

    # Determine step size: ~20% of the range between nok and ok
    span = abs(current_ok - current_nok)
    if span < 1e-6:
        # nok == ok (e.g. revenue trend where both are 0.0)
        # Use absolute step based on the value magnitude
        step = max(abs(current_ok) * 0.2, 0.02)
    else:
        step = span * 0.3

    candidates = []
    nok_range = [current_nok + i * step for i in range(-n_steps, n_steps + 1)]
    ok_range = [current_ok + i * step for i in range(-n_steps, n_steps + 1)]

    for nok in nok_range:
        for ok in ok_range:
            # Enforce ordering: for dir=+1 nok<ok, for dir=-1 nok>ok
            if direction == +1 and nok >= ok:
                continue
            if direction == -1 and nok <= ok:
                continue
            candidates.append({"nok": round(nok, 4), "ok": round(ok, 4)})

    # Always include the original
    candidates.append({"nok": current_nok, "ok": current_ok})
    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
        key = (c["nok"], c["ok"])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def _score_with_weights(df_ts, metrics, weights_dict, thresholds_dict=None):
    """Score a timespan slice with custom weights and thresholds.

    thresholds_dict: optional {metric: {"nok": x, "ok": y}}
    Returns (scores, returns) aligned Series, or (None, None).
    """
    sm = SummaryManager()
    sm._weight_overrides = weights_dict
    sm.process_historical(df_ts, metrics, thresholds=thresholds_dict)
    calculate_score(sm, metrics_to_score=metrics)

    scored = sm.summary
    if scored is None or (isinstance(scored, pd.DataFrame) and scored.empty):
        scored = sm.summary_investment
    if scored is None or (isinstance(scored, pd.DataFrame) and scored.empty):
        return None, None
    if isinstance(scored, dict):
        scored = pd.DataFrame(scored).T

    s = pd.to_numeric(scored.get("points", pd.Series(dtype=float)), errors="coerce")
    returns = df_ts.set_index("company")["total_return"]
    common = s.index.intersection(returns.index)
    if len(common) < 5:
        return None, None

    sv = s.loc[common].astype(float)
    rv = returns.loc[common].astype(float)
    valid = sv.notna() & rv.notna()
    if valid.sum() < 5:
        return None, None
    return sv[valid], rv[valid]


def _avg_spearman_across_windows(weights_dict, df_total, target_timespans, metrics,
                                  thresholds_dict=None):
    """Compute average Spearman correlation across timespans."""
    corrs = []
    for ts in target_timespans:
        df_ts = df_total[df_total["timespan"] == ts].copy()
        if len(df_ts) < 5:
            continue
        sv, rv = _score_with_weights(df_ts, metrics, weights_dict, thresholds_dict)
        if sv is None:
            continue
        rho, _ = sp_stats.spearmanr(sv, rv)
        if not np.isnan(rho):
            corrs.append(rho)
    return np.mean(corrs) if corrs else 0.0


def _avg_quintile_spread_across_windows(weights_dict, df_total, target_timespans,
                                        metrics, thresholds_dict=None):
    """Average quintile long-short spread across timespans (optimizer objective).

    Mirrors _avg_spearman_across_windows but scores each window by the
    decision-relevant top-minus-bottom spread instead of raw Spearman.
    """
    spreads = []
    for ts in target_timespans:
        df_ts = df_total[df_total["timespan"] == ts].copy()
        if len(df_ts) < 5:
            continue
        sv, rv = _score_with_weights(df_ts, metrics, weights_dict, thresholds_dict)
        if sv is None:
            continue
        spread = _quintile_spread(sv, rv)
        if spread is not None and not np.isnan(spread):
            spreads.append(spread)
    return np.mean(spreads) if spreads else 0.0


def _load_previous_results(variant):
    """Load previous optimization results if they exist.

    variant: "individual" or "combo"
    Returns (weights_dict, thresholds_dict) or (None, None) if not found.
    """
    filenames = {
        "individual": "optimization_results_individual.json",
        "combo": "optimization_results_combo.json",
    }
    filename = filenames.get(variant)
    if not filename:
        return None, None

    path = os.path.join(os.getcwd(), filename)
    if not os.path.exists(path):
        # Also check project root
        path = os.path.join(project_root, filename)
    if not os.path.exists(path):
        return None, None

    try:
        with open(path) as f:
            data = json.load(f)
        weights = data.get("optimized_weights")
        thresholds = data.get("optimized_thresholds")
        if weights:
            print(f"  Loaded previous results from {path}")
            return weights, thresholds
    except Exception as e:
        print(f"  [WARN] Could not load {path}: {e}")
    return None, None


def _get_starting_weights_and_thresholds(csv_path, variant):
    """Get starting weights and thresholds for an optimization run.

    Priority:
    1. Previous results for this variant (warm-start / iterative refinement)
    2. Previous individual results (as base for combo)
    3. Run individual optimization fresh (fallback)
    """
    # 1. Check for previous results of the same method
    prev_w, prev_t = _load_previous_results(variant)
    if prev_w:
        print(f"  → Warm-starting from previous {variant} results")
        thresholds = prev_t if prev_t else _get_default_thresholds()
        return prev_w, thresholds

    # 2. For combo, check for existing individual results
    if variant == "combo":
        ind_w, ind_t = _load_previous_results("individual")
        if ind_w:
            print(f"  → Starting from previous individual results")
            thresholds = ind_t if ind_t else _get_default_thresholds()
            return ind_w, thresholds

    # 3. Run individual optimization fresh
    print(f"  → No previous results found, running individual optimization...")
    result = optimize_weights_and_thresholds(csv_path=csv_path)
    if not result or "optimized_weights" not in result:
        metrics = _all_scored_metrics()
        return {m: 1.0 for m in metrics}, _get_default_thresholds()
    weights = result["optimized_weights"]
    thresholds = result.get("optimized_thresholds", _get_default_thresholds())
    return weights, thresholds


def _prepare_data(csv_path):
    """Load data and determine target timespans."""
    df = _load_timespan_csv(csv_path)
    if df.empty:
        return None, None, None, None
    metrics = _all_scored_metrics()
    all_ts = df["timespan"].unique()
    # Objective windows: 3Y/5Y TOTAL only (see optimize_weights_and_thresholds).
    target_timespans = [
        t for t in all_ts
        if "TOTAL" in str(t) and ("3Y" in str(t) or "5Y" in str(t))
    ]
    if not target_timespans:
        target_timespans = list(all_ts)
    df_total = df[df["timespan"].isin(target_timespans)]
    return df, df_total, target_timespans, metrics


_CV_FOLDS = 5


def _company_folds(companies, k=_CV_FOLDS):
    """Deterministic leave-companies-out partition.

    Sort company names and stride them into k folds (company i -> fold i % k).
    Sorting makes the assignment reproducible across runs; striding keeps the
    folds balanced. A company's full set of (self-overlapping) window rows
    always lands in exactly one fold.
    """
    ordered = sorted(set(companies))
    folds = [[] for _ in range(k)]
    for i, c in enumerate(ordered):
        folds[i % k].append(c)
    return [set(f) for f in folds if f]


def _cv_score(weights_dict, df_total, target_timespans, metrics,
              thresholds_dict=None):
    """Leave-companies-out cross-validation of the quintile-spread objective.

    Partition companies into K=5 deterministic folds and evaluate the
    quintile-spread objective separately on each fold's own rows (across the
    3Y/5Y TOTAL objective windows), then average the fold scores. This keeps
    every company's overlapping windows together in a single fold, so the
    search never sees fragments of one company's overlapping windows as if
    they were independent evidence. Weight/threshold selection remains the
    outer search loop's job -- this only changes which rows are scored.
    """
    companies = df_total["company"].unique()
    folds = _company_folds(companies)
    if len(folds) < 2:
        return _avg_quintile_spread_across_windows(
            weights_dict, df_total, target_timespans, metrics, thresholds_dict
        )

    fold_scores = []
    for fold in folds:
        df_fold = df_total[df_total["company"].isin(fold)]
        if df_fold.empty:
            continue
        fold_scores.append(
            _avg_quintile_spread_across_windows(
                weights_dict, df_fold, target_timespans, metrics, thresholds_dict
            )
        )
    return float(np.mean(fold_scores)) if fold_scores else 0.0


# ======================================================================
# Phase C: Grid sweep + cross-validation (combo)
# ======================================================================

def optimize_combo(csv_path="metrics_by_timespan.csv"):
    """Grid sweep around independent-correlation weights AND thresholds
    with cross-validation.

    1. Get starting weights + thresholds from method 1
    2. Coordinate descent: for each metric, sweep weight and threshold
    3. Evaluate using CV across time windows
    4. Pick the best combination
    """
    print("\n" + "=" * 70)
    print("  COMBO OPTIMIZATION (Grid Sweep + Cross-Validation)")
    print("=" * 70)

    print("\n[Step 1] Getting starting point...")
    start_weights, start_thresholds = _get_starting_weights_and_thresholds(csv_path, "combo")

    df, df_total, target_timespans, metrics = _prepare_data(csv_path)
    if df is None:
        print("[WARN] No data.")
        return {}

    print(f"\n[Step 2] Grid sweep over {len(metrics)} metrics (weights + thresholds), "
          f"{len(target_timespans)} time windows...")

    best_weights = dict(start_weights)
    best_thresholds = dict(start_thresholds)
    best_cv = _cv_score(best_weights, df_total, target_timespans, metrics, best_thresholds)
    print(f"  Starting CV quintile spread: {best_cv:+.4f}")

    # Coordinate descent: sweep weight AND threshold per metric, repeat
    max_rounds = 10
    for round_num in range(1, max_rounds + 1):
        improved = False
        for m in metrics:
            # --- Weight sweep ---
            current_w = best_weights.get(m, 0.0)
            weight_candidates = sorted(set(
                max(0.0, min(2.0, round((current_w + d) * 4) / 4))
                for d in np.arange(-0.5, 0.5, 0.1)
            ))

            for cand_w in weight_candidates:
                if cand_w == best_weights.get(m, 0.0):
                    continue
                trial_w = dict(best_weights)
                trial_w[m] = cand_w
                trial_w = _apply_weight_constraints(trial_w)

                cv = _cv_score(trial_w, df_total, target_timespans, metrics, best_thresholds)
                if cv > best_cv + 1e-6:
                    best_cv = cv
                    best_weights = trial_w
                    improved = True

            # --- Threshold refinement (only for metrics with weight > 0) ---
            # Bounded 2D local search per metric, not a fixed grid -- see
            # _refine_threshold_2d.
            if best_weights.get(m, 0) > 0 and m in best_thresholds:
                refined_thr, refined_cv = _refine_threshold_2d(
                    m, best_weights, best_thresholds, df_total, target_timespans, metrics
                )
                if refined_cv is not None and refined_cv > best_cv + 1e-6:
                    best_cv = refined_cv
                    best_thresholds = dict(best_thresholds)
                    best_thresholds[m] = refined_thr
                    improved = True

        print(f"  Round {round_num}: CV quintile spread = {best_cv:+.4f}")
        if not improved:
            print("  Converged.")
            break

    best_weights = _apply_weight_constraints(best_weights)

    # Compute baseline for comparison
    baseline_cv = _cv_score(
        {m: 1.0 for m in metrics}, df_total, target_timespans, metrics
    )

    # Report
    print("\n" + "-" * 70)
    print("  COMBO OPTIMIZATION RESULTS")
    print("-" * 70)
    print(f"\n  Equal-weight CV quintile spread:    {baseline_cv:+.4f}")
    print(f"  Combo-optimized CV quintile spread: {best_cv:+.4f}")
    print(f"\nOptimized weights:")
    for m in sorted(best_weights, key=lambda x: best_weights[x], reverse=True):
        w = best_weights[m]
        sw = start_weights.get(m, 0.0)
        delta = w - sw
        print(f"  {m:40s}  w={w:.2f}  (indep={sw:.2f}, Δ={delta:+.2f})")

    default_thr = _get_default_thresholds()
    print(f"\nThreshold changes:")
    for m in sorted(best_thresholds):
        old = default_thr.get(m)
        new = best_thresholds[m]
        if old and (old["nok"] != new["nok"] or old["ok"] != new["ok"]):
            print(f"  {m:40s}  ({old['nok']}, {old['ok']}) → ({new['nok']}, {new['ok']})")

    # Reliability
    # Reliability uses only the non-overlapping 5Y_YoY-* windows (3Y_YoY-k and
    # 5Y_YoY-k are the same calendar window; TOTAL windows overlap each other),
    # so each company's sample is genuinely independent (n capped at 5).
    yoy_timespans = [t for t in df["timespan"].unique() if str(t).startswith("5Y_YoY-")]
    reliability = _compute_reliability(df, yoy_timespans)

    # Save
    thr_serializable = {m: {"nok": t["nok"], "ok": t["ok"]}
                        for m, t in best_thresholds.items()}
    result = {
        "method": "combo_grid_cv",
        "optimized_weights": best_weights,
        "optimized_thresholds": thr_serializable,
        "cv_quintile_spread": round(best_cv, 4),
        "baseline_cv_quintile_spread": round(baseline_cv, 4),
        "independent_weights": start_weights,
    }

    out_path = "optimization_results_combo.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    if reliability is not None and not reliability.empty:
        reliability.to_csv("company_reliability_combo.csv", index=False)

    print("=" * 70)
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
    ap.add_argument("--optimize-combo", action="store_true",
                    help="Grid sweep + cross-validation optimization")
    ap.add_argument("--csv", default="metrics_by_timespan.csv",
                    help="Path to metrics_by_timespan.csv")
    args = ap.parse_args()

    if args.optimize_combo:
        optimize_combo(args.csv)
    elif args.baseline:
        baseline_correlation(args.csv)
    elif args.optimize:
        optimize_weights_and_thresholds(csv_path=args.csv)
    else:
        print("Running baseline correlation analysis...")
        print("(Use --optimize or --optimize-combo)")
        baseline_correlation(args.csv)
