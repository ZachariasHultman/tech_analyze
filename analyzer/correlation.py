# ----------------------------------------------------------------------
#  Correlation analysis: score vs forward return
#  Panel pipeline only — the fiscal-year cross-section built by
#  analyzer/panel.py. The older rolling-window (metrics_by_timespan.csv)
#  optimizers were removed: their objective windows (3Y_TOTAL/5Y_TOTAL) were
#  both anchored at the same max_date and fully overlapping, so they carried
#  no time diversification and nothing downstream consumed their output.
# ----------------------------------------------------------------------

import sys
import os
import warnings
import json
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from analyzer.summary_manager import SummaryManager
from analyzer.data_processing import calculate_score
from analyzer.fast_score import cached_score_matrix, points_from_matrix
from analyzer.stats_utils import deflated_sharpe_ratio
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
# Module-level constants (shared across the panel optimizers)
# ======================================================================

# Metrics whose correlation with returns is partly circular (price-derived,
# not fundamentals). Cap their weight so fundamentals remain the primary driver.
MOMENTUM_METRICS = {"price momentum status"}
MOMENTUM_WEIGHT_CAP = 1.0

# Minimum weight floors for academically proven metrics. These may show
# weak/negative correlation due to data-quality issues in the historical
# adapter, but are well-established in research.
WEIGHT_FLOORS = {
    "piotroski f-score status": 0.5,
    "dividend yield status": 0.25,
    # OCF is populated live for Nordic stocks but Avanza exposes no
    # historical OCF series, so the backtest/optimizer can never compute a
    # real correlation for this metric -- its weight comes entirely from
    # this floor. See HIGH_WEIGHT_METRICS in metrics.py for why it's not in
    # HIGHEST (would permanently block the bonus/malus check for non-Nordic
    # holdings, where operatingCashFlow is always None).
    "earnings quality status": 0.5,
}

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
    """Return every metric that has a weight > 0, in a stable order.

    `sorted`, not `list(set(...))`: Python randomizes string hashing per
    process, so the un-sorted version returned a different order on every run.
    Coordinate descent iterates this list, so the order picks the greedy path
    and therefore the final weights -- which live scoring then loads from
    optimization_results_panel.json. See tests/test_metric_order_stable.py.
    """
    return sorted(
        set(HIGHEST_WEIGHT_METRICS)
        | set(HIGH_WEIGHT_METRICS)
        | set(LOW_WEIGHT_METRICS)
    )


def _score_snapshot(df_slice, metrics_to_score=None, thresholds=None,
                    weight_overrides=None):
    """Score a slice of historical data using SummaryManager.

    Returns a DataFrame with company index and 'points' + per-metric scores.

    weight_overrides: optional {metric: weight}. When None (default) the
    production HIGHEST/HIGH/LOW weight tiers apply exactly as before — every
    existing caller passes nothing, so behavior is unchanged. The new panel
    pipeline passes an equal-weight dict here to score without those tiers.
    """
    sm = SummaryManager()
    if weight_overrides is not None:
        sm._weight_overrides = weight_overrides
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
# Shared weight/threshold helpers
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


def _score_with_weights(df_ts, metrics, weights_dict, thresholds_dict=None,
                        return_col="total_return"):
    """Score a timespan slice with custom weights and thresholds.

    thresholds_dict: optional {metric: {"nok": x, "ok": y}}
    return_col: which column holds the target return. The panel pipeline
        passes "fwd_excess_return_1y"; the "total_return" default is the
        legacy signature, kept so callers must name the column they mean.
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
    returns = df_ts.set_index("company")[return_col]
    common = s.index.intersection(returns.index)
    if len(common) < 5:
        return None, None

    sv = s.loc[common].astype(float)
    rv = returns.loc[common].astype(float)
    valid = sv.notna() & rv.notna()
    if valid.sum() < 5:
        return None, None
    return sv[valid], rv[valid]


# ======================================================================
# Phase D: Panel-based gating (challenger to equal weight)
# ======================================================================
#
# The whole optimizer now. Everything operates on the fiscal-year panel
# (grouped by "fiscal_year", target "fwd_excess_return_1y", company key
# "company"); the helpers above are the shared scoring/weight machinery it
# builds on.

PANEL_RETURN_COL = "fwd_excess_return_1y"


def _panel_year_scores(g, fiscal_year, metrics, weights_dict, thresholds_dict,
                       return_col):
    """(scores, returns) for one fiscal year, via the cached score matrix.

    Exactly equivalent to _score_with_weights on the same slice (asserted in
    tests/test_fast_score.py), but the expensive part -- the unit-weight
    scoring pass -- is memoised per (year, metrics, thresholds), so a weight
    sweep reuses it instead of rebuilding the whole SummaryManager chain for
    every candidate.
    """
    G, hw = cached_score_matrix(g, metrics, thresholds_dict, group_key=fiscal_year)
    if G is None or G.empty:
        return None, None
    s = points_from_matrix(G, hw, weights_dict)
    returns = g.set_index("company")[return_col]
    # Duplicate company labels make .loc fan out, so the returns series comes
    # back longer than the scores and the combined mask matches neither.
    # dedupe_fiscal_years removes the known cause upstream; this keeps a future
    # one from silently double-weighting a company instead of erroring.
    if returns.index.has_duplicates:
        returns = returns[~returns.index.duplicated(keep="last")]
    if s.index.has_duplicates:
        s = s[~s.index.duplicated(keep="last")]
    common = s.index.intersection(returns.index)
    if len(common) < 5:
        return None, None
    sv = s.loc[common].astype(float)
    rv = returns.loc[common].astype(float)
    valid = sv.notna() & rv.notna()
    if valid.sum() < 5:
        return None, None
    return sv[valid], rv[valid]


def _panel_avg_quintile_spread(weights_dict, panel_df, metrics,
                               thresholds_dict=None, return_col=PANEL_RETURN_COL):
    """Mean quintile long-short spread across fiscal years (panel objective)."""
    spreads = []
    for fy, g in panel_df.groupby("fiscal_year"):
        if len(g) < 5:
            continue
        sv, rv = _panel_year_scores(
            g, fy, metrics, weights_dict, thresholds_dict, return_col
        )
        if sv is None:
            continue
        spread = _quintile_spread(sv, rv)
        if spread is not None and not np.isnan(spread):
            spreads.append(spread)
    return float(np.mean(spreads)) if spreads else 0.0


def _scale_weights_from_corrs(avg_corrs, metrics):
    """Correlation -> weight scaling: strongest fundamental = 2.0, momentum
    capped, weight floors applied, snapped to 0.25. Metrics absent from
    `avg_corrs` (or correlating at or below +0.02) get exactly their
    WEIGHT_FLOORS entry, nothing else."""
    positive = {m: r for m, r in avg_corrs.items() if r > 0.02}
    if not positive:
        return {m: WEIGHT_FLOORS.get(m, 0.0) for m in metrics}
    fundamental = {m: r for m, r in positive.items() if m not in MOMENTUM_METRICS}
    max_corr = max(fundamental.values()) if fundamental else max(positive.values())
    weights = {}
    for m in metrics:
        if m in positive:
            w = round((positive[m] / max_corr * 2.0) * 4) / 4
            if m in MOMENTUM_METRICS:
                w = min(w, MOMENTUM_WEIGHT_CAP)
            weights[m] = max(w, WEIGHT_FLOORS.get(m, 0.0))
        else:
            weights[m] = WEIGHT_FLOORS.get(m, 0.0)
    return weights


def _panel_per_metric_corrs(panel_df, metrics, return_col=PANEL_RETURN_COL):
    """Per-metric avg Spearman with the target, grouped by fiscal year."""
    metric_corrs = {}
    for _, g in panel_df.groupby("fiscal_year"):
        if len(g) < 5:
            continue
        scored = _score_snapshot(g)
        if scored.empty:
            continue
        returns = g.set_index("company")[return_col]
        for sc in [c for c in scored.columns if c.endswith("_score")]:
            m = sc.replace("_score", "")
            if m not in metrics:
                continue
            ms = pd.to_numeric(scored[sc], errors="coerce")
            common = ms.index.intersection(returns.index)
            if len(common) < 5:
                continue
            mv = ms.loc[common].astype(float)
            rv = returns.loc[common].astype(float)
            valid = mv.notna() & rv.notna()
            if valid.sum() < 5:
                continue
            rho, _ = sp_stats.spearmanr(mv[valid], rv[valid])
            if not np.isnan(rho):
                metric_corrs.setdefault(m, []).append(rho)
    return {m: float(np.mean(v)) for m, v in metric_corrs.items()}


def optimize_panel_weights_and_thresholds(panel_df, metrics):
    """The "individual" method: per-metric Spearman -> weight scaling, then a
    per-metric threshold grid search maximizing the panel quintile spread.
    A single fast pass, no coordinate descent. Logs every
    (candidate -> objective) evaluation into trial_objectives (the real N-trials
    count for the Deflated Sharpe Ratio)."""
    trial_objectives = []
    avg_corrs = _panel_per_metric_corrs(panel_df, metrics)
    weights = _scale_weights_from_corrs(avg_corrs, metrics)

    default_thresholds = _get_default_thresholds()
    thresholds = dict(default_thresholds)
    for m in metrics:
        if weights.get(m, 0) == 0 or m not in thresholds:
            continue
        cur = thresholds[m]
        best_thr, best_spread = cur, -np.inf
        for cand in _threshold_grid_for_metric(m, cur["nok"], cur["ok"], n_steps=2):
            trial = dict(thresholds)
            trial[m] = cand
            spread = _panel_avg_quintile_spread(weights, panel_df, metrics, trial)
            trial_objectives.append(spread)
            if spread > best_spread:
                best_spread, best_thr = spread, cand
        thresholds[m] = best_thr

    return {
        "optimized_weights": weights,
        "optimized_thresholds": thresholds,
        "per_metric_correlations": avg_corrs,
        "trial_objectives": trial_objectives,
    }


def optimize_panel_combo(panel_df, metrics):
    """Coordinate descent over weight and threshold per metric (up to 10
    rounds, stops when nothing improves), objective = panel quintile spread.
    Starts from optimize_panel_weights_and_thresholds and logs every trial
    objective for the Deflated Sharpe Ratio's N-trials count."""
    trial_objectives = []
    start = optimize_panel_weights_and_thresholds(panel_df, metrics)
    trial_objectives.extend(start["trial_objectives"])
    best_weights = dict(start["optimized_weights"])
    best_thresholds = dict(start["optimized_thresholds"])
    best = _panel_avg_quintile_spread(best_weights, panel_df, metrics, best_thresholds)
    trial_objectives.append(best)

    for _ in range(10):
        improved = False
        for m in metrics:
            current_w = best_weights.get(m, 0.0)
            for cand_w in sorted(set(
                max(0.0, min(2.0, round((current_w + d) * 4) / 4))
                for d in np.arange(-0.5, 0.5, 0.1)
            )):
                if cand_w == best_weights.get(m, 0.0):
                    continue
                trial_w = _apply_weight_constraints(dict(best_weights, **{m: cand_w}))
                spread = _panel_avg_quintile_spread(
                    trial_w, panel_df, metrics, best_thresholds
                )
                trial_objectives.append(spread)
                if spread > best + 1e-6:
                    best, best_weights, improved = spread, trial_w, True

            if best_weights.get(m, 0) > 0 and m in best_thresholds:
                cur = best_thresholds[m]
                for cand in _threshold_grid_for_metric(m, cur["nok"], cur["ok"], n_steps=2):
                    trial_t = dict(best_thresholds)
                    trial_t[m] = cand
                    spread = _panel_avg_quintile_spread(
                        best_weights, panel_df, metrics, trial_t
                    )
                    trial_objectives.append(spread)
                    if spread > best + 1e-6:
                        best, best_thresholds, improved = spread, trial_t, True
        if not improved:
            break

    return {
        "optimized_weights": _apply_weight_constraints(best_weights),
        "optimized_thresholds": best_thresholds,
        "trial_objectives": trial_objectives,
    }


def permutation_benchmark(panel_df, metrics, optimizer_fn, n_permutations=200,
                          seed=0, return_col=PANEL_RETURN_COL, progress_every=25):
    """Null distribution of the best objective this search finds on pure noise.

    Shuffles the target **within each fiscal year** (destroying the
    score-to-return link while preserving every cross-section's own return
    distribution and the panel's shape), refits the optimizer, and records the
    best objective it managed to reach. Repeated ``n_permutations`` times, that
    is the distribution of "how good a result this exact search produces when
    there is nothing to find."

    It replaces the Euler-Mascheroni expected-max approximation with a measured
    one, and unlike that approximation it is immune to how the grid was sized
    or how many duplicate candidates it evaluated.

    Shuffling only the target leaves the fundamentals untouched, so the cached
    score matrices stay valid across permutations -- which is what makes this
    affordable. Do not clear the cache in the loop.

    Returns ``{"null_best": [...], "sigma": float, "mean": float,
    "p95": float, "n_permutations": int}``.
    """
    skipped = {"null_best": [], "sigma": None, "mean": None, "p95": None,
               "n_permutations": 0}
    if n_permutations < 1:
        return skipped
    if (panel_df is None or panel_df.empty
            or "fiscal_year" not in panel_df.columns
            or return_col not in panel_df.columns):
        # Nothing to shuffle. Callers that stub out the walk-forward (tests,
        # and any future caller holding a pre-target frame) must not be forced
        # to supply a full panel just to reach the verdict.
        return skipped

    rng = np.random.default_rng(seed)
    groups = list(panel_df.groupby("fiscal_year").indices.values())
    values = panel_df[return_col].to_numpy(dtype=float)
    shuffled = panel_df.copy()
    null_best = []

    print(f"  [permutation] running {n_permutations} refits on shuffled targets...")
    for i in range(n_permutations):
        permuted = values.copy()
        for idx in groups:
            block = permuted[idx]
            rng.shuffle(block)
            permuted[idx] = block
        shuffled[return_col] = permuted

        fit = optimizer_fn(shuffled, metrics)
        objectives = [o for o in fit.get("trial_objectives", [])
                      if o is not None and not np.isnan(o)]
        if objectives:
            null_best.append(float(max(objectives)))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  [permutation] {i + 1}/{n_permutations}")

    if not null_best:
        return {"null_best": [], "sigma": None, "mean": None, "p95": None,
                "n_permutations": n_permutations}
    arr = np.asarray(null_best, dtype=float)
    return {
        "null_best": null_best,
        "sigma": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "mean": float(arr.mean()),
        "p95": float(np.quantile(arr, 0.95)),
        "n_permutations": int(arr.size),
    }


def permutation_p_value(null_best, observed):
    """P(search reaches `observed` on noise), with the standard +1 correction.

    The +1 in numerator and denominator keeps the estimate from ever being
    exactly zero on a finite number of permutations -- claiming p=0 from 200
    draws would overstate what was actually measured.
    """
    arr = np.asarray([v for v in null_best if v is not None and not np.isnan(v)],
                     dtype=float)
    if arr.size == 0 or observed is None or np.isnan(observed):
        return float("nan")
    return float((np.sum(arr >= observed) + 1) / (arr.size + 1))


def _panel_fold_eval(weights, thresholds, panel_year, metrics, fiscal_year=None):
    """Out-of-sample (quintile spread, IC) on one held-out fiscal year."""
    sv, rv = _panel_year_scores(
        panel_year, fiscal_year, metrics, weights, thresholds, PANEL_RETURN_COL
    )
    if sv is None:
        return None, None
    spread = _quintile_spread(sv, rv)
    rho, _ = sp_stats.spearmanr(sv, rv)
    return (
        spread if spread is not None and not np.isnan(spread) else None,
        rho if not np.isnan(rho) else None,
    )


def leave_one_fiscal_year_out(panel_df, metrics, optimizer_fn=optimize_panel_combo):
    """Genuine walk-forward: for each fiscal year Y, refit on every *other*
    year and evaluate both the optimized weights and the equal-weight baseline
    on year Y's own cross-section. Cheap enough to refit per fold at this scale
    (~127 companies, 5-8 folds)."""
    default_thresholds = _get_default_thresholds()
    equal_weights = {m: 1.0 for m in metrics}
    folds = []
    for Y in sorted(panel_df["fiscal_year"].unique()):
        train = panel_df[panel_df["fiscal_year"] != Y]
        test = panel_df[panel_df["fiscal_year"] == Y]
        if train.empty or len(test) < 5:
            continue
        fit = optimizer_fn(train, metrics)
        opt_spread, opt_ic = _panel_fold_eval(
            fit["optimized_weights"], fit["optimized_thresholds"], test, metrics,
            fiscal_year=Y,
        )
        eq_spread, eq_ic = _panel_fold_eval(
            equal_weights, default_thresholds, test, metrics, fiscal_year=Y
        )
        folds.append({
            "fiscal_year": int(Y),
            "optimized_spread": opt_spread, "optimized_ic": opt_ic,
            "equal_spread": eq_spread, "equal_ic": eq_ic,
        })
    return folds


def gate_optimized_weights(panel_df, metrics, optimizer_fn=optimize_panel_combo,
                           confidence=0.95, n_permutations=200,
                           permutation_seed=0):
    """Accept/reject the optimized weights as a challenger to equal weight.

    Accepts iff BOTH (a) mean optimized quintile spread beats mean equal-weight
    spread out-of-sample (walk-forward) AND (b) the Deflated Sharpe Ratio
    exceeds `confidence` (default 0.95 — correcting for the size of the grid
    sweep). Otherwise rejects and falls back to equal weight + default
    thresholds, stating which condition failed.

    `confidence` is compared directly against the raw DSR probability
    (`deflated_sharpe_ratio`'s own `significant_at_95` field is fixed at 0.95
    and intentionally left untouched as the primitive's default reporting —
    this function makes its own accept/reject call against whatever bar the
    caller actually wants).
    """
    print("\n" + "=" * 72)
    print("  PANEL CHALLENGER VERDICT (new pipeline)")
    print("  Optimized weights vs equal weight, out-of-sample walk-forward")
    print("=" * 72)

    folds = leave_one_fiscal_year_out(panel_df, metrics, optimizer_fn)
    opt_spreads = [f["optimized_spread"] for f in folds if f["optimized_spread"] is not None]
    eq_spreads = [f["equal_spread"] for f in folds if f["equal_spread"] is not None]
    for f in folds:
        print(f"  {f['fiscal_year']}: optimized spread={_fmt(f['optimized_spread'])} "
              f"(IC={_fmt(f['optimized_ic'])})  "
              f"equal spread={_fmt(f['equal_spread'])} (IC={_fmt(f['equal_ic'])})")

    mean_opt = float(np.mean(opt_spreads)) if opt_spreads else float("nan")
    mean_eq = float(np.mean(eq_spreads)) if eq_spreads else float("nan")

    # How often the challenger actually won its held-out year. At n=4 periods
    # no statistic has real power, so this blunt count is printed next to the
    # DSR: it is the number a human can read and judge.
    paired = [f for f in folds
              if f["optimized_spread"] is not None and f["equal_spread"] is not None]
    n_beat = sum(1 for f in paired if f["optimized_spread"] > f["equal_spread"])

    # trial_objectives from a full-data fit = the real N-trials count for DSR.
    full_fit = optimizer_fn(panel_df, metrics)
    trial_objectives = full_fit.get("trial_objectives", [])
    finite_trials = [o for o in trial_objectives
                     if o is not None and not np.isnan(o)]
    observed_best = max(finite_trials) if finite_trials else float("nan")

    # Measure the null instead of approximating it, when there is budget.
    perm = permutation_benchmark(
        panel_df, metrics, optimizer_fn,
        n_permutations=n_permutations, seed=permutation_seed,
    )
    p_value = permutation_p_value(perm["null_best"], observed_best)
    if perm["n_permutations"] and perm["sigma"] is not None:
        dsr = deflated_sharpe_ratio(
            trial_objectives, opt_spreads,
            sigma_sr_override=perm["sigma"],
            sr_benchmark_override=perm["p95"],
            n_trials_override=len(set(np.round(finite_trials, 10))),
        )
    else:
        dsr = deflated_sharpe_ratio(trial_objectives, opt_spreads)

    beats = (not np.isnan(mean_opt) and not np.isnan(mean_eq) and mean_opt > mean_eq)
    dsr_value = dsr.get("dsr", float("nan"))
    significant = bool(dsr_value > confidence) if not np.isnan(dsr_value) else False
    accept = beats and significant

    print("-" * 72)
    print(f"  mean optimized spread (OOS): {_fmt(mean_opt)}")
    print(f"  mean equal-weight spread (OOS): {_fmt(mean_eq)}")
    print(f"  beat equal weight in {n_beat} of {len(paired)} held-out year(s)")
    if perm["n_permutations"]:
        print(f"  permutation null ({perm['n_permutations']} refits on shuffled "
              f"targets): mean={_fmt(perm['mean'])} p95={_fmt(perm['p95'])} "
              f"sigma={_fmt(perm['sigma'])}")
        print(f"  observed best in-sample objective: {_fmt(observed_best)}  "
              f"-> permutation p={p_value:.3f}")
    else:
        print("  permutation null: SKIPPED (--permutations 0) — DSR falls back "
              "to the Euler-Mascheroni approximation, whose sigma is grid "
              "dispersion, not sampling noise. Treat it as decorative.")
    print(f"  DSR: {_fmt(dsr.get('dsr'))}  (n_trials={dsr.get('n_trials')}, "
          f"benchmark={_fmt(dsr.get('sr_benchmark'))}, "
          f"confidence_bar={confidence:.3f}, significant={significant})")

    if accept:
        print("  DECISION: ACCEPT optimized weights (beats equal weight AND DSR-significant).")
        chosen_weights = full_fit["optimized_weights"]
        chosen_thresholds = full_fit["optimized_thresholds"]
    else:
        reasons = []
        if not beats:
            reasons.append("did not beat equal weight out-of-sample")
        if not significant:
            reasons.append(f"Deflated Sharpe Ratio not significant at {confidence:.1%}")
        print(f"  DECISION: REJECT — {', '.join(reasons)}. "
              f"Falling back to equal weight + default thresholds.")
        chosen_weights = {m: 1.0 for m in metrics}
        chosen_thresholds = _get_default_thresholds()
    print("=" * 72)

    return {
        "accept": accept,
        "beats_equal_weight": beats,
        "dsr_significant": significant,
        "confidence": confidence,
        "mean_optimized_spread": mean_opt,
        "mean_equal_spread": mean_eq,
        "dsr": dsr,
        "folds": folds,
        "n_folds_beating_equal": n_beat,
        "n_folds": len(paired),
        "n_companies": (
            int(panel_df["company_id"].nunique())
            if panel_df is not None and "company_id" in getattr(panel_df, "columns", [])
            else None
        ),
        "permutation": {k: v for k, v in perm.items() if k != "null_best"},
        "permutation_p_value": p_value,
        "observed_best_objective": observed_best,
        "chosen_weights": chosen_weights,
        "chosen_thresholds": chosen_thresholds,
    }


def build_validation_summary(gate_result):
    """Condense the gate's verdict into what a reader needs to judge it.

    Reports the out-of-sample numbers for the weights **actually chosen** --
    the optimized fold results on accept, the equal-weight ones on reject --
    rather than always the challenger's, which would describe something the
    system isn't running.
    """
    accepted = bool(gate_result.get("accept"))
    ic_key = "optimized_ic" if accepted else "equal_ic"
    spread_key = "optimized_spread" if accepted else "equal_spread"

    def _clean(v):
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else v

    per_year = [
        {
            "fiscal_year": f.get("fiscal_year"),
            "ic": _clean(f.get(ic_key)),
            "spread": _clean(f.get(spread_key)),
        }
        for f in gate_result.get("folds", [])
    ]
    per_year.sort(key=lambda r: (r["fiscal_year"] is None, r["fiscal_year"]))

    ics = [r["ic"] for r in per_year if r["ic"] is not None]
    spreads = [r["spread"] for r in per_year if r["spread"] is not None]
    if len(ics) > 1:
        t_stat, p_value = sp_stats.ttest_1samp(ics, popmean=0.0)
        t_stat, p_value = float(t_stat), float(p_value)
    else:
        t_stat = p_value = None

    return {
        "fitted_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "n_periods": len(per_year),
        "n_companies": gate_result.get("n_companies"),
        "per_year": per_year,
        "mean_ic": float(np.mean(ics)) if ics else None,
        "mean_spread": float(np.mean(spreads)) if spreads else None,
        "t_stat": t_stat,
        "p_value": p_value,
        "n_folds_beating_equal": gate_result.get("n_folds_beating_equal"),
        "n_folds": gate_result.get("n_folds"),
        "permutation_p_value": _clean(gate_result.get("permutation_p_value")),
        "n_permutations": (gate_result.get("permutation") or {}).get("n_permutations"),
        "return_basis": "total return (price + dividends), demeaned within year",
    }


def save_panel_optimization_results(gate_result, out_path="optimization_results_panel.json"):
    """Persist gate_optimized_weights' verdict so live scoring can load it via
    _load_optimized_params("panel") (analyzer/main.py). Always writes, on
    accept AND reject -- on reject, chosen_weights/chosen_thresholds are
    already the gate's own equal-weight/default-threshold fallback, so this
    file always reflects a defensible recommendation, never a broken one.

    The `validation` block travels in the same file as the weights, so the one
    artefact copied to the Pi carries both what to score with and the evidence
    for trusting it -- the email renders that block verbatim.
    """
    result = {
        "optimized_weights": gate_result["chosen_weights"],
        "optimized_thresholds": gate_result["chosen_thresholds"],
        "accepted": gate_result["accept"],
        "confidence": gate_result["confidence"],
        "dsr": gate_result["dsr"].get("dsr"),
        "mean_optimized_spread": gate_result["mean_optimized_spread"],
        "mean_equal_spread": gate_result["mean_equal_spread"],
        "validation": build_validation_summary(gate_result),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved panel optimization results to {out_path}")
    return out_path


def _fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x:+.4f}"
