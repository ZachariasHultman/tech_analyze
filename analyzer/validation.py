"""Step 4 validation battery for the fiscal-year panel, wired to --validate.

Standard cross-sectional quant-equity validation on the panel built by
panel.py: quintile portfolio sorts, an Information Coefficient time series, and
Fama-MacBeth regression. Every number is computed live from the loaded panel;
nothing is hardcoded. Power is intentionally low (n = fiscal years, ~5), so the
whole report is framed as directional, not proof.
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from analyzer.correlation import _quintile_spread
from analyzer.stats_utils import fama_macbeth

SCORE_COL = "composite_score_equal"
RETURN_COL = "fwd_excess_return_1y"


def _metric_score_cols(panel_df):
    """Per-metric '<name>_score' columns only.

    Excludes 'combined_score' (a composite quality*value column that happens to
    end in '_score', not a per-metric score) and the all-NaN earnings-quality
    column (no historical OCF), which would otherwise be mistaken for factors.
    """
    return [
        c for c in panel_df.columns
        if c.endswith("_score") and c != "combined_score"
        and not panel_df[c].isna().all()
    ]


def _usable_metric_cols(panel_df, return_col=RETURN_COL, min_coverage=0.9):
    """Metric score columns dense enough for complete-case OLS.

    Multi-factor Fama-MacBeth needs rows non-null across every regressor at
    once; sparse metrics would empty every period. Keep only columns populated
    in >= min_coverage of the return-available rows.
    """
    ret = panel_df[panel_df[return_col].notna()]
    if ret.empty:
        return []
    return [
        c for c in _metric_score_cols(panel_df)
        if ret[c].notna().mean() >= min_coverage
    ]


# ---------------------------------------------------------------- caveats
def print_caveat_header(panel_df) -> dict:
    """Print (and return) the standing caveats + this panel's real dimensions."""
    ret = panel_df[panel_df[RETURN_COL].notna()]
    n_periods = int(ret["fiscal_year"].nunique())
    n_company_years = int(len(ret))
    n_companies = int(ret["company_id"].nunique())

    print("=" * 72)
    print("  PANEL VALIDATION BATTERY")
    print("=" * 72)
    print(f"  periods (fiscal years w/ returns): {n_periods}")
    print(f"  company-years with forward return: {n_company_years}")
    print(f"  distinct companies:                {n_companies}")
    print("  CAVEATS:")
    print("   - Survivorship bias: today's watchlist/preset membership is applied")
    print("     retroactively; Avanza exposes no point-in-time constituents.")
    print("   - Fundamentals appear as-reported/frozen (checked across 3 snapshot")
    print("     dates, bit-identical) — but no fiscal-year rollover was observed in")
    print("     that window; re-check after the next one.")
    print(f"   - Standard errors are unreliable at n={n_periods} periods. Treat every")
    print("     result below as DIRECTIONAL, not proof.")
    print("=" * 72)
    return {
        "n_periods": n_periods,
        "n_company_years": n_company_years,
        "n_companies": n_companies,
    }


# ---------------------------------------------------------------- sanity
def run_sanity_checks(panel_df, fundamentals_df) -> dict:
    """Three cheap misalignment/direction checks. Both an eyeball diagnostic on
    every --validate run and (via test_panel_sanity_checks) a regression test of
    the check logic itself. Consumes precomputed panels — does not re-score.
    """
    print("\n--- Sanity checks ---")
    out = {}

    # 1. bottom-decile ROE company-years should mostly score negative.
    merged = panel_df.merge(
        fundamentals_df[["company_id", "fiscal_year", "roe"]],
        on=["company_id", "fiscal_year"], how="left",
    )
    neg_fracs = []
    for fy, g in merged.groupby("fiscal_year"):
        roe = pd.to_numeric(g["roe"], errors="coerce")
        if roe.notna().sum() < 10:
            continue
        cutoff = roe.quantile(0.10)
        bottom = g[roe <= cutoff]
        comp = pd.to_numeric(bottom[SCORE_COL], errors="coerce").dropna()
        if comp.empty:
            continue
        neg_fracs.append(float((comp < 0).mean()))
    mean_neg = float(np.mean(neg_fracs)) if neg_fracs else float("nan")
    out["bottom_roe_negative_score_fraction"] = mean_neg
    print(f"  [1] bottom-decile-ROE years scoring negative: {mean_neg:.0%} "
          f"(expect a majority)")
    if not np.isnan(mean_neg) and mean_neg < 0.5:
        print("      [WARN] fewer than half score negative — scoring/ROE misalignment?")

    # 2. excess-return direction in the single best broad-rally year.
    ret = panel_df[panel_df[RETURN_COL].notna()]
    rally_neg_frac = float("nan")
    if not ret.empty:
        by_year = ret.groupby("fiscal_year")["universe_mean_return_that_year"].first()
        best_year = by_year.idxmax()
        rally = ret[ret["fiscal_year"] == best_year]
        rally_neg_frac = float((rally[RETURN_COL] < 0).mean())
        out["best_rally_year"] = int(best_year)
        out["best_rally_negative_excess_fraction"] = rally_neg_frac
        print(f"  [2] best rally year {int(best_year)} "
              f"(univ mean {by_year.max():+.1%}): "
              f"{rally_neg_frac:.0%} of firms had NEGATIVE excess return "
              f"(expect ~half)")
        if rally_neg_frac < 0.15:
            print("      [WARN] suspiciously few negatives — demeaning/alignment bug?")

    # 3. within-year percentile ranks span ~[0,1] independently.
    pct_col = "roe_pe ratio status_pct"
    spans_ok = True
    if pct_col in panel_df.columns:
        for fy, g in panel_df.groupby("fiscal_year"):
            v = pd.to_numeric(g[pct_col], errors="coerce").dropna()
            # skip thin early years: with only ~3 firms ranks can't reach 0/1.
            if len(v) < 10:
                continue
            if v.min() > 0.15 or v.max() < 0.85:
                spans_ok = False
                print(f"      [WARN] {fy}: {pct_col} spans only "
                      f"[{v.min():.2f}, {v.max():.2f}] — not ranked within-year?")
    out["within_year_ranks_span_full"] = spans_ok
    print(f"  [3] within-year percentile ranks span ~[0,1]: {spans_ok}")
    return out


# ---------------------------------------------------------------- quintiles
def _bucket_means(scores, returns, n_buckets):
    """Mean return per score bucket, ordered lowest-score -> highest-score."""
    df = pd.DataFrame({"s": scores, "r": returns}).dropna()
    ranks = df["s"].rank(method="first")
    buckets = pd.qcut(ranks, n_buckets, labels=False)
    means = df["r"].groupby(buckets).mean()
    return [float(means.get(i, np.nan)) for i in range(n_buckets)]


def quintile_sorts_by_fiscal_year(panel_df, score_col=SCORE_COL, return_col=RETURN_COL):
    """Per fiscal year: every bucket's mean return, Q(top)-Q(bottom) spread
    (reusing _quintile_spread), and an exact monotonicity check. Plus a pooled
    mean-spread summary."""
    print("\n--- Quintile sorts by fiscal year ---")
    rows = []
    for fy, g in panel_df.groupby("fiscal_year"):
        sub = g[[score_col, return_col]].dropna()
        n = len(sub)
        if n < 6:
            continue
        n_buckets = 5 if n >= 25 else 3
        means = _bucket_means(sub[score_col], sub[return_col], n_buckets)
        spread = _quintile_spread(sub[score_col], sub[return_col])
        monotonic = all(means[i] < means[i + 1] for i in range(len(means) - 1))
        rows.append({
            "fiscal_year": int(fy), "n": n, "n_buckets": n_buckets,
            "bucket_means": means, "spread": spread, "monotonic": monotonic,
        })
        pretty = "  ".join(f"Q{i+1}={m:+.1%}" for i, m in enumerate(means))
        flag = "monotonic" if monotonic else "NON-monotonic"
        print(f"  {int(fy)} (n={n}, {n_buckets}b): {pretty}  "
              f"spread={spread:+.1%}  [{flag}]")

    spreads = [r["spread"] for r in rows if r["spread"] is not None and not np.isnan(r["spread"])]
    mean_spread = float(np.mean(spreads)) if spreads else float("nan")
    print(f"  pooled: mean top-bottom spread across years = {mean_spread:+.1%} "
          f"({len(spreads)} years)")
    return {"per_year": rows, "mean_spread": mean_spread}


# ---------------------------------------------------------------- IC series
def ic_time_series(panel_df, score_col=SCORE_COL, return_col=RETURN_COL):
    """Per-fiscal-year Spearman IC (score vs excess return), + mean/std/ICIR."""
    print("\n--- Information Coefficient (IC) time series ---")
    ics = []
    for fy, g in panel_df.groupby("fiscal_year"):
        sub = g[[score_col, return_col]].dropna()
        if len(sub) < 5:
            continue
        rho, _ = sp_stats.spearmanr(sub[score_col], sub[return_col])
        if not np.isnan(rho):
            ics.append((int(fy), float(rho), len(sub)))
            print(f"  {int(fy)}: IC={rho:+.3f} (n={len(sub)})")
    vals = [ic for _, ic, _ in ics]
    mean_ic = float(np.mean(vals)) if vals else float("nan")
    std_ic = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
    icir = mean_ic / std_ic if std_ic and not np.isnan(std_ic) and std_ic != 0 else float("nan")
    print(f"  mean IC={mean_ic:+.3f}  std={std_ic:.3f}  ICIR={icir:+.3f}  "
          f"(low power: n={len(vals)} periods)")
    return {"per_year": ics, "mean_ic": mean_ic, "std_ic": std_ic, "icir": icir}


# ------------------------------------------------------- currency-neutral IC
# A currency cohort needs at least this many rows in a year to be demeaned.
# A 1-company cohort demeans to exactly zero, which is a fake perfect
# neutralisation rather than a measurement.
_MIN_COHORT = 5


def _load_currency_map():
    """{company_id: currency} from the Yahoo symbol map, or {} when absent."""
    try:
        from analyzer.fx import currency_for_symbol
        from analyzer.yahoo_prices import load_symbol_map
    except Exception:
        return {}
    try:
        return {
            company: currency_for_symbol(symbol)
            for company, symbol in load_symbol_map().items()
            if currency_for_symbol(symbol)
        }
    except Exception:
        return {}


def currency_neutral_ic(panel_df, currency_map=None, score_col=SCORE_COL,
                        total_return_col="fwd_total_return_1y"):
    """IC against a target demeaned within (fiscal_year, currency).

    The SEK target is correct -- it is what a SEK investor earns -- but it
    gives every USD-listed name the same USD/SEK move, so a currency cohort
    sits inside the target as a shared factor that within-year demeaning does
    not remove. On the real panel, currency-within-year explains 5.3% of return
    variance (year itself explains 6.7%), and neutralising it drops the pooled
    quintile spread from +0.97% to +0.23%: most of the headline spread was
    currency exposure, not company selection.

    Reported *alongside* the headline IC, never instead of it. The raw number
    is what you earn; this one is what you are skilled at. A score that merely
    knows which currency a stock trades in scores well on the former and zero
    on the latter.

    Returns None when the currency map or the total-return column is missing,
    so a fresh clone with no Yahoo backfill still runs --validate.
    """
    if panel_df is None or panel_df.empty or not currency_map:
        return None
    if total_return_col not in panel_df.columns:
        return None

    df = panel_df.copy()
    df["_ccy"] = df["company_id"].map(currency_map)
    df = df.dropna(subset=["_ccy", total_return_col, score_col])
    if df.empty or df["_ccy"].nunique() < 1:
        return None

    # Variance attributable to the currency cohort, over and above the year.
    total = df[total_return_col]
    grand = total.mean()
    ss_tot = float(((total - grand) ** 2).sum())
    year_mean = df.groupby("fiscal_year")[total_return_col].transform("mean")
    ccy_mean = df.groupby(["fiscal_year", "_ccy"])[total_return_col].transform("mean")
    ccy_share = (float(((ccy_mean - year_mean) ** 2).sum()) / ss_tot
                 if ss_tot > 0 else 0.0)

    sizes = df.groupby(["fiscal_year", "_ccy"])[total_return_col].transform("size")
    keep = df[sizes >= _MIN_COHORT].copy()
    if keep.empty:
        return None
    keep["_neutral"] = keep[total_return_col] - keep.groupby(
        ["fiscal_year", "_ccy"])[total_return_col].transform("mean")
    keep["_raw"] = keep[total_return_col] - keep.groupby(
        "fiscal_year")[total_return_col].transform("mean")

    per_year, ics, ics_raw = [], [], []
    for fy, g in keep.groupby("fiscal_year"):
        if len(g) < 25:
            continue
        rho, _ = sp_stats.spearmanr(g[score_col], g["_neutral"])
        rho_raw, _ = sp_stats.spearmanr(g[score_col], g["_raw"])
        if np.isnan(rho) or np.isnan(rho_raw):
            continue
        per_year.append((int(fy), float(rho), float(rho_raw), len(g)))
        ics.append(float(rho))
        ics_raw.append(float(rho_raw))
    if not ics:
        return None

    t_stat, p_value = (sp_stats.ttest_1samp(ics, 0.0) if len(ics) > 1
                       else (np.nan, np.nan))
    return {
        "per_year": per_year,
        "mean_ic": float(np.mean(ics)),
        "mean_ic_raw": float(np.mean(ics_raw)),
        "std_ic": float(np.std(ics, ddof=1)) if len(ics) > 1 else float("nan"),
        "t_stat": float(t_stat), "p_value": float(p_value),
        "currency_variance_share": ccy_share,
        "cohorts_used": sorted(keep["_ccy"].unique()),
    }


def report_currency_neutral_ic(panel_df, currency_map=None):
    """Print the currency-neutral IC beside the headline one."""
    print("\n--- Currency-neutral IC (stock picking vs currency exposure) ---")
    res = currency_neutral_ic(panel_df, currency_map=currency_map)
    if res is None:
        print("  skipped — no currency map (run --backfill-prices) or no "
              "total-return column.")
        return None
    print(f"  currency-within-year explains {res['currency_variance_share']:.1%} "
          f"of forward-return variance  (cohorts: {', '.join(res['cohorts_used'])})")
    for fy, rho, rho_raw, n in res["per_year"]:
        print(f"  {fy}: neutral IC={rho:+.3f}  (raw {rho_raw:+.3f})  n={n}")
    print(f"  mean neutral IC={res['mean_ic']:+.3f}  vs raw {res['mean_ic_raw']:+.3f}"
          f"   t={res['t_stat']:+.2f} p={res['p_value']:.3f}")
    print("  Raw is what a SEK investor earns; neutral is what the score picks.")
    if res["mean_ic_raw"] != 0 and res["mean_ic"] < 0.5 * res["mean_ic_raw"]:
        print("  [WARN] over half the raw edge disappears once currency is "
              "neutralised — the score is partly a currency bet.")
    return res


def per_metric_ic(panel_df, return_col=RETURN_COL):
    """Per-metric IC grouped by fiscal year — which individual metrics carry
    signal."""
    print("\n--- Per-metric IC (mean Spearman across fiscal years) ---")
    score_cols = _metric_score_cols(panel_df)
    results = {}
    for sc in score_cols:
        rhos = []
        for fy, g in panel_df.groupby("fiscal_year"):
            sub = g[[sc, return_col]].dropna()
            if len(sub) < 5:
                continue
            rho, _ = sp_stats.spearmanr(sub[sc], sub[return_col])
            if not np.isnan(rho):
                rhos.append(rho)
        if rhos:
            results[sc.replace("_score", "")] = float(np.mean(rhos))
    for m, r in sorted(results.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {m:40s} mean IC={r:+.3f}")
    return results


# ---------------------------------------------------------------- Fama-MacBeth
def run_fama_macbeth_report(panel_df, x_cols, label, return_col=RETURN_COL):
    """Thin wrapper around stats_utils.fama_macbeth (per-year cross-sectional
    OLS + coefficient t-test). Power is n=periods, NOT n=firm-years."""
    print(f"\n--- Fama-MacBeth ({label}) ---")
    x_cols = list(x_cols)
    if not x_cols:
        print("  no sufficiently-dense regressors — skipped.")
        return {"per_factor": {}, "n_periods_used": 0, "n_periods_skipped": 0}
    sub = panel_df[["fiscal_year", return_col] + list(x_cols)].dropna(
        subset=[return_col] + list(x_cols)
    )
    res = fama_macbeth(sub, x_cols, return_col, "fiscal_year", standardize=True)
    print(f"  periods used={res['n_periods_used']} "
          f"skipped={res['n_periods_skipped']} (power is n=periods, NOT firm-years)")
    for c, stats in res["per_factor"].items():
        print(f"  {c:40s} mean coef={stats['mean']:+.4f} "
              f"t={stats['t_stat']:+.2f} p={stats['p_value']:.3f} "
              f"(n_periods={stats['n_periods']})")
    return res


# ---------------------------------------------------------------- battery
def run_validation_battery(panel_scores_path="data/panel_scores.csv",
                           panel_fundamentals_path="data/panel_fundamentals.csv") -> dict:
    """Top-level entry point for --validate. Runs the caveat header, sanity
    checks, quintile sorts (4a), IC time series (4b), and Fama-MacBeth (4c) in
    order; notes that the factor-attribution step (4d) was cut. Returns every
    sub-result in a dict so tests can assert on structured output."""
    from analyzer.panel import drop_thin_years

    panel = pd.read_csv(panel_scores_path)
    panel.columns = panel.columns.str.strip()
    # A year too small to rank is not evidence -- see analyzer/config.py's
    # MIN_CROSS_SECTION. Applied before anything reads the panel so the
    # header's period count, the IC series and the quintile sorts all agree.
    panel = drop_thin_years(panel, RETURN_COL, label="validate")
    try:
        fundamentals = pd.read_csv(panel_fundamentals_path)
        fundamentals.columns = fundamentals.columns.str.strip()
    except FileNotFoundError:
        fundamentals = pd.DataFrame(columns=["company_id", "fiscal_year", "roe"])

    header = print_caveat_header(panel)
    sanity = run_sanity_checks(panel, fundamentals)
    quintiles = quintile_sorts_by_fiscal_year(panel)
    ic = ic_time_series(panel)
    ccy_ic = report_currency_neutral_ic(panel, currency_map=_load_currency_map())
    metric_ic = per_metric_ic(panel)
    fm_single = run_fama_macbeth_report(panel, [SCORE_COL], "single-factor composite")
    metric_cols = _usable_metric_cols(panel)
    fm_multi = run_fama_macbeth_report(panel, metric_cols, "multi-factor per-metric")

    print("\n--- Step 4d (factor regression) ---")
    print("  CUT: can't separate real alpha from a repackaged value/quality")
    print("  factor without external Fama-French/AQR data — out of scope.")
    print("=" * 72)

    return {
        "header": header,
        "sanity": sanity,
        "quintiles": quintiles,
        "ic": ic,
        "currency_neutral_ic": ccy_ic,
        "per_metric_ic": metric_ic,
        "fama_macbeth_single": fm_single,
        "fama_macbeth_multi": fm_multi,
    }
