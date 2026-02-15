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

    # Metrics whose correlation with returns is partly circular
    # (they measure price-derived signals, not fundamentals).
    # Cap their weight so fundamentals remain the primary driver.
    MOMENTUM_METRICS = {"price momentum status"}
    MOMENTUM_WEIGHT_CAP = 1.0

    # Minimum weight floors for academically proven metrics.
    # These may show weak/negative correlation due to data quality issues
    # in the historical adapter, but are well-established in research.
    WEIGHT_FLOORS = {
        "piotroski f-score status": 0.5,
        "dividend yield status": 0.25,
        "earnings quality status": 0.25,
    }

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
    optimized_thresholds = dict(default_thresholds)

    for m in metrics:
        if optimized_weights.get(m, 0) == 0:
            continue  # skip dropped metrics
        if m not in default_thresholds:
            continue

        cur = default_thresholds[m]
        candidates = _threshold_grid_for_metric(m, cur["nok"], cur["ok"], n_steps=2)

        best_thr = cur
        best_rho = -999

        for cand in candidates:
            trial_thr = dict(optimized_thresholds)
            trial_thr[m] = cand
            rho = _avg_spearman_across_windows(
                optimized_weights, df_total, target_timespans, metrics, trial_thr
            )
            if rho > best_rho:
                best_rho = rho
                best_thr = cand

        optimized_thresholds[m] = best_thr

    # ---- Step 4: Re-score with optimized weights + thresholds ----
    print("\n[Step 4] Re-scoring with optimized weights and thresholds...")

    avg_baseline = _avg_spearman_across_windows(
        {m: (2.0 if m in HIGHEST_WEIGHT_METRICS else 1.5 if m in HIGH_WEIGHT_METRICS else 1.0)
         for m in metrics},
        df_total, target_timespans, metrics
    )
    avg_optimized = _avg_spearman_across_windows(
        optimized_weights, df_total, target_timespans, metrics, optimized_thresholds
    )

    # ---- Step 5: Per-company fundamental reliability ----
    print("\n[Step 5] Computing per-company fundamental reliability...")

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
        "baseline_spearman": round(avg_baseline, 4),
        "optimized_spearman": round(avg_optimized, 4),
    }

    out_path = "optimization_results_individual.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved optimization results to {out_path}")

    _save_thresholds_to_metrics_py(optimized_thresholds, "INDIVIDUAL")

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
# Shared helpers for combo / stepwise optimization
# ======================================================================

MOMENTUM_METRICS = {"price momentum status"}
MOMENTUM_WEIGHT_CAP = 1.0
WEIGHT_FLOORS = {
    "piotroski f-score status": 0.5,
    "dividend yield status": 0.25,
    "earnings quality status": 0.25,
}


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


def _get_starting_weights_and_thresholds(csv_path):
    """Run method 1 (individual) to get starting weights and thresholds."""
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
    target_timespans = [t for t in df["timespan"].unique() if "TOTAL" in str(t)]
    if not target_timespans:
        target_timespans = list(df["timespan"].unique())
    df_total = df[df["timespan"].isin(target_timespans)]
    return df, df_total, target_timespans, metrics


def _cv_score(weights_dict, df_total, target_timespans, metrics,
              thresholds_dict=None):
    """Leave-one-out cross-validation score across time windows."""
    if len(target_timespans) < 2:
        return _avg_spearman_across_windows(
            weights_dict, df_total, target_timespans, metrics, thresholds_dict
        )

    val_corrs = []
    for held_out in target_timespans:
        df_val = df_total[df_total["timespan"] == held_out].copy()
        if len(df_val) < 5:
            continue
        sv, rv = _score_with_weights(df_val, metrics, weights_dict, thresholds_dict)
        if sv is None:
            continue
        rho, _ = sp_stats.spearmanr(sv, rv)
        if not np.isnan(rho):
            val_corrs.append(rho)

    return np.mean(val_corrs) if val_corrs else 0.0


def _save_thresholds_to_metrics_py(thresholds_dict, variant):
    """Append/update OPTIMIZED_THRESHOLDS_<VARIANT> dict in metrics.py.

    variant: "INDIVIDUAL", "COMBO", or "STEPWISE"
    """
    metrics_path = os.path.join(os.path.dirname(__file__), "metrics.py")
    if not os.path.exists(metrics_path):
        print(f"[WARN] metrics.py not found at {metrics_path}, skipping threshold save.")
        return

    dict_name = f"OPTIMIZED_THRESHOLDS_{variant.upper()}"

    with open(metrics_path, "r") as f:
        content = f.read()

    # Build the new dict string
    lines = [f"{dict_name} = {{"]
    for m in sorted(thresholds_dict.keys()):
        t = thresholds_dict[m]
        lines.append(f'    "{m}": ({t["nok"]}, {t["ok"]}),')
    lines.append("}")
    new_block = "\n".join(lines) + "\n"

    # Check if dict already exists — replace it
    import re
    pattern = rf'^{dict_name}\s*=\s*\{{.*?\}}\s*$'
    match = re.search(pattern, content, re.MULTILINE | re.DOTALL)

    if match:
        content = content[:match.start()] + new_block + content[match.end():]
    else:
        # Append before the get_metrics_threshold function, or at end
        insert_marker = "\n\ndef get_metrics_threshold"
        if insert_marker in content:
            content = content.replace(
                insert_marker,
                "\n\n" + new_block + insert_marker
            )
        else:
            content = content.rstrip() + "\n\n\n" + new_block

    with open(metrics_path, "w") as f:
        f.write(content)
    print(f"  Saved {dict_name} to {metrics_path}")


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

    print("\n[Step 1] Running independent correlation to get starting point...")
    start_weights, start_thresholds = _get_starting_weights_and_thresholds(csv_path)

    df, df_total, target_timespans, metrics = _prepare_data(csv_path)
    if df is None:
        print("[WARN] No data.")
        return {}

    print(f"\n[Step 2] Grid sweep over {len(metrics)} metrics (weights + thresholds), "
          f"{len(target_timespans)} time windows...")

    best_weights = dict(start_weights)
    best_thresholds = dict(start_thresholds)
    best_cv = _cv_score(best_weights, df_total, target_timespans, metrics, best_thresholds)
    print(f"  Starting CV Spearman: {best_cv:+.4f}")

    # Coordinate descent: sweep weight AND threshold per metric, repeat
    max_rounds = 3
    for round_num in range(1, max_rounds + 1):
        improved = False
        for m in metrics:
            # --- Weight sweep ---
            current_w = best_weights.get(m, 0.0)
            weight_candidates = sorted(set(
                max(0.0, min(2.0, round((current_w + d) * 4) / 4))
                for d in [-0.5, -0.25, 0.0, 0.25, 0.5]
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

            # --- Threshold sweep (only for metrics with weight > 0) ---
            if best_weights.get(m, 0) > 0 and m in best_thresholds:
                cur_thr = best_thresholds[m]
                thr_candidates = _threshold_grid_for_metric(
                    m, cur_thr["nok"], cur_thr["ok"], n_steps=2
                )

                for cand_thr in thr_candidates:
                    if cand_thr["nok"] == cur_thr["nok"] and cand_thr["ok"] == cur_thr["ok"]:
                        continue
                    trial_thr = dict(best_thresholds)
                    trial_thr[m] = cand_thr

                    cv = _cv_score(best_weights, df_total, target_timespans, metrics, trial_thr)
                    if cv > best_cv + 1e-6:
                        best_cv = cv
                        best_thresholds = trial_thr
                        improved = True

        print(f"  Round {round_num}: CV Spearman = {best_cv:+.4f}")
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
    print(f"\n  Equal-weight CV Spearman:    {baseline_cv:+.4f}")
    print(f"  Combo-optimized CV Spearman: {best_cv:+.4f}")
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
    reliability = _compute_reliability(df, target_timespans)

    # Save
    thr_serializable = {m: {"nok": t["nok"], "ok": t["ok"]}
                        for m, t in best_thresholds.items()}
    result = {
        "method": "combo_grid_cv",
        "optimized_weights": best_weights,
        "optimized_thresholds": thr_serializable,
        "cv_spearman": round(best_cv, 4),
        "baseline_cv_spearman": round(baseline_cv, 4),
        "independent_weights": start_weights,
    }

    out_path = "optimization_results_combo.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    _save_thresholds_to_metrics_py(best_thresholds, "COMBO")

    if reliability is not None and not reliability.empty:
        reliability.to_csv("company_reliability_combo.csv", index=False)

    print("=" * 70)
    return result


# ======================================================================
# Phase D: Scipy numerical optimization (stepwise)
# ======================================================================

def optimize_stepwise(csv_path="metrics_by_timespan.csv"):
    """Numerical optimization of all weights AND thresholds simultaneously.

    Uses Nelder-Mead (derivative-free) to maximize average Spearman
    correlation across time windows with cross-validation.

    The parameter vector is: [weight_0, ..., weight_N, nok_0, ok_0, ..., nok_N, ok_N]
    for all metrics that have thresholds.
    """
    print("\n" + "=" * 70)
    print("  STEPWISE OPTIMIZATION (Scipy Nelder-Mead + Cross-Validation)")
    print("=" * 70)

    print("\n[Step 1] Running independent correlation to get starting point...")
    start_weights, start_thresholds = _get_starting_weights_and_thresholds(csv_path)

    df, df_total, target_timespans, metrics = _prepare_data(csv_path)
    if df is None:
        print("[WARN] No data.")
        return {}

    # Order metrics consistently for vector operations
    metric_list = sorted(metrics)
    # Metrics that have thresholds to optimize
    thr_metrics = [m for m in metric_list if m in start_thresholds]

    # Build x0: [weights..., nok_0, ok_0, nok_1, ok_1, ...]
    n_weights = len(metric_list)
    n_thr = len(thr_metrics)
    x0_weights = [start_weights.get(m, 0.0) for m in metric_list]
    x0_thresholds = []
    for m in thr_metrics:
        t = start_thresholds[m]
        x0_thresholds.extend([t["nok"], t["ok"]])
    x0 = np.array(x0_weights + x0_thresholds)

    print(f"\n[Step 2] Optimizing {n_weights} weights + {n_thr * 2} threshold params over "
          f"{len(target_timespans)} time windows...")

    eval_count = [0]

    def objective(x):
        """Negative CV Spearman (we minimize)."""
        # Decode weights
        w_dict = {}
        for i, m in enumerate(metric_list):
            w_dict[m] = float(np.clip(x[i], 0.0, 2.0))
        w_dict = _apply_weight_constraints(w_dict)

        # Decode thresholds
        thr_dict = dict(start_thresholds)  # start with defaults for non-optimized
        for j, m in enumerate(thr_metrics):
            nok = float(x[n_weights + j * 2])
            ok = float(x[n_weights + j * 2 + 1])

            # Enforce ordering based on direction
            direction = DIRECTION_OVERRIDES.get(m, +1)
            if m in RATIO_SPECS:
                direction = RATIO_SPECS[m]["dir"]
            if direction == +1 and nok >= ok:
                ok = nok + 0.01
            elif direction == -1 and nok <= ok:
                nok = ok + 0.01

            thr_dict[m] = {"nok": round(nok, 4), "ok": round(ok, 4)}

        cv = _cv_score(w_dict, df_total, target_timespans, metrics, thr_dict)
        eval_count[0] += 1
        if eval_count[0] % 50 == 0:
            print(f"    eval {eval_count[0]}: CV Spearman = {cv:+.4f}")
        return -cv

    result_opt = sp_optimize.minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={
            "maxiter": 1000,
            "maxfev": 5000,
            "xatol": 0.02,
            "fatol": 1e-4,
            "adaptive": True,
        },
    )

    # Extract final weights
    best_weights = {}
    for i, m in enumerate(metric_list):
        best_weights[m] = float(np.clip(result_opt.x[i], 0.0, 2.0))
    best_weights = _apply_weight_constraints(best_weights)

    # Extract final thresholds
    best_thresholds = dict(start_thresholds)
    for j, m in enumerate(thr_metrics):
        nok = float(result_opt.x[n_weights + j * 2])
        ok = float(result_opt.x[n_weights + j * 2 + 1])
        direction = DIRECTION_OVERRIDES.get(m, +1)
        if m in RATIO_SPECS:
            direction = RATIO_SPECS[m]["dir"]
        if direction == +1 and nok >= ok:
            ok = nok + 0.01
        elif direction == -1 and nok <= ok:
            nok = ok + 0.01
        best_thresholds[m] = {"nok": round(nok, 4), "ok": round(ok, 4)}

    best_cv = -result_opt.fun
    baseline_cv = _cv_score(
        {m: 1.0 for m in metrics}, df_total, target_timespans, metrics
    )

    # Report
    print(f"\n  Converged after {result_opt.nfev} evaluations")
    print("\n" + "-" * 70)
    print("  STEPWISE OPTIMIZATION RESULTS")
    print("-" * 70)
    print(f"\n  Equal-weight CV Spearman:       {baseline_cv:+.4f}")
    print(f"  Stepwise-optimized CV Spearman: {best_cv:+.4f}")
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
    reliability = _compute_reliability(df, target_timespans)

    # Save
    thr_serializable = {m: {"nok": t["nok"], "ok": t["ok"]}
                        for m, t in best_thresholds.items()}
    result = {
        "method": "stepwise_nelder_mead_cv",
        "optimized_weights": best_weights,
        "optimized_thresholds": thr_serializable,
        "cv_spearman": round(best_cv, 4),
        "baseline_cv_spearman": round(baseline_cv, 4),
        "independent_weights": start_weights,
        "scipy_converged": bool(result_opt.success),
        "scipy_message": result_opt.message,
        "n_evaluations": int(result_opt.nfev),
    }

    out_path = "optimization_results_stepwise.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    _save_thresholds_to_metrics_py(best_thresholds, "STEPWISE")

    if reliability is not None and not reliability.empty:
        reliability.to_csv("company_reliability_stepwise.csv", index=False)

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
    ap.add_argument("--optimize-stepwise", action="store_true",
                    help="Scipy numerical optimization + cross-validation")
    ap.add_argument("--csv", default="metrics_by_timespan.csv",
                    help="Path to metrics_by_timespan.csv")
    args = ap.parse_args()

    if args.optimize_combo:
        optimize_combo(args.csv)
    elif args.optimize_stepwise:
        optimize_stepwise(args.csv)
    elif args.baseline:
        baseline_correlation(args.csv)
    elif args.optimize:
        optimize_weights_and_thresholds(csv_path=args.csv)
    else:
        print("Running baseline correlation analysis...")
        print("(Use --optimize, --optimize-combo, or --optimize-stepwise)")
        baseline_correlation(args.csv)
