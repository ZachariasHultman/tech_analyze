"""Is the score useful, in which variant, and does weighting help? (--evaluate)

Replaces the weight optimizer. That searched ~34 parameters (weight and
threshold per metric) on 7 fiscal years and was rejected by its own gate on
every run -- the last one beat equal weight in 3 of 7 held-out years,
permutation p=0.149 -- so live scoring had been running equal metric weights
all along. The question worth answering on this data is not "which weights"
but whether the ranking earns its keep at all. Three layers, each against a
different baseline, because each failure points at a different fix:

1. **Score variants** (``score_usefulness``). For the live key and its two
   halves: IC per year, and the top-25 basket after the sleeve gate, with
   equal *stock* weights, against the universe mean. Basket losing to the
   universe means the score adds nothing.
2. **Benchmarks.** The same baskets, and the whole universe, against
   OMXS30 GI and MSCI World over each row's own window. A universe that loses
   to the index while the basket beats the universe means the score works
   and the universe is the problem.
3. **Weighting** (``weighting_backtest``). The live basket as equal weight,
   min-variance and max-Sharpe (analyzer/frontier.py, the same functions the
   live email uses). Covariance is estimated strictly before the basket's
   earliest report date, so nothing from any holding window enters it.

Everything reads the already-built panel; no Avanza session. Each fiscal
year's windows are staggered (each row runs report_date -> +1y), so a
"basket" is the panel's convention, not a portfolio held on one date, and the
index comparison is matched per row over that row's own window.

The panel's scores use equal metric weights (panel.build_scores_panel), which
is exactly what live runs now -- the backtest is of the system as deployed.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from analyzer.benchmarks import load_benchmarks, load_core, window_return
from analyzer.config import CORE_MIX, SATELLITE_SHARE, SLEEVE_GATE_MIN
from analyzer.summary_manager import one_per_company
from analyzer.frontier import (
    CORE_KEY, WEIGHT_CAP, _solve, common_weekly_returns, grinold_alpha, ledoit_wolf,
)

TARGET = "fwd_excess_return_1y"   # total return, SEK, demeaned within year
TOTAL = "fwd_total_return_1y"
LIVE_KEY = "blend_score"
# Same as --push-top's default: the basket the backtest validates is the one
# that gets pushed.
TOP_N = 25
# Covariance lookback before each decision date, as in the live email.
LOOKBACK_YEARS = 5
WEEKS_PER_YEAR = 52
RESULTS_PATH = "evaluation_results.json"

VARIANTS = [
    ("score (live)", LIVE_KEY),
    ("points only", "fundamental_pct"),
    ("momentum only", "momentum_pct"),
]
PORTFOLIOS = [("equal", "equal stock weight"), ("minvar", "A min-var"),
              ("maxsharpe", "B max-Sharpe"),
              ("minvar_core", "A min-var, core-aware"),
              ("maxsharpe_core", "B max-Sharpe, core-aware")]


def load_panel(scores_path="data/panel_scores.csv"):
    """Scored panel rows with a forward return, thin/partial years dropped
    (the same MIN_CROSS_SECTION / MIN_YEAR_COVERAGE guards --validate uses)."""
    from analyzer.panel import drop_thin_years
    df = pd.read_csv(scores_path)
    df.columns = df.columns.str.strip()
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df = drop_thin_years(df, TARGET, label="evaluate")
    return df[df[TARGET].notna()].copy()


def basket(g, key, top_n=TOP_N):
    """The rows live would push: sleeve gate, then the top N on ``key``."""
    gated = g[(g["quality_pct"] >= SLEEVE_GATE_MIN)
              & (g["value_pct"] >= SLEEVE_GATE_MIN)]
    ranked = gated.dropna(subset=[key]).sort_values(key, ascending=False)
    # One listing per company, exactly as main._gate_and_rank does live.
    ranked = one_per_company(ranked.set_index("company_id", drop=False))
    return ranked.head(top_n).reset_index(drop=True)


def _row_vs_bench(rows, bench):
    """Per-row total return minus the benchmark over that row's own window.
    None unless the benchmark covers every row -- a partly covered basket
    would be compared on a subsample."""
    out = []
    for _, r in rows.iterrows():
        start = r["report_date"]
        b = window_return(bench, start, start + pd.DateOffset(years=1))
        if b is None or pd.isna(r[TOTAL]):
            return None
        out.append(r[TOTAL] - b)
    return np.array(out) if out else None


def _ttest(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return None
    return float(sp_stats.ttest_1samp(vals, 0.0).pvalue)


def _vs_summary(per_year, bench_labels):
    out = {}
    for label in bench_labels:
        vals = [y["vs"][label] for y in per_year if y["vs"].get(label) is not None]
        out[label] = {
            "mean": float(np.mean(vals)) if vals else None,
            "years_beating": int(sum(v > 0 for v in vals)),
            "n_years": len(vals),
        }
    return out


def score_usefulness(df, benches, top_n=TOP_N):
    """Layers 1 and 2: each score variant's IC and basket, and the universe,
    against the universe mean and each benchmark."""
    labels = list(benches)
    variants = []
    for label, key in VARIANTS:
        if key not in df.columns:
            continue
        per_year = []
        for year, g in df.groupby("fiscal_year"):
            ranked = g.dropna(subset=[key])
            if ranked[key].nunique() < 3:
                continue
            ic = sp_stats.spearmanr(ranked[key], ranked[TARGET]).correlation
            b = basket(g, key, top_n)
            if b.empty or not np.isfinite(ic):
                continue
            vs = {}
            for bl, series in benches.items():
                d = _row_vs_bench(b, series)
                vs[bl] = float(d.mean()) if d is not None else None
            per_year.append({"fiscal_year": int(year), "ic": float(ic),
                             "excess": float(b[TARGET].mean()),
                             "n_basket": int(len(b)), "vs": vs})
        if not per_year:
            continue
        ics = [y["ic"] for y in per_year]
        exc = [y["excess"] for y in per_year]
        variants.append({
            "label": label, "key": key, "per_year": per_year,
            "n_periods": len(per_year),
            "mean_ic": float(np.mean(ics)), "ic_p_value": _ttest(ics),
            "years_ic_positive": int(sum(v > 0 for v in ics)),
            "mean_excess": float(np.mean(exc)), "excess_p_value": _ttest(exc),
            "years_excess_positive": int(sum(v > 0 for v in exc)),
            "vs": _vs_summary(per_year, labels),
        })

    universe = []
    for year, g in df.groupby("fiscal_year"):
        vs = {}
        for bl, series in benches.items():
            d = _row_vs_bench(g, series)
            vs[bl] = float(d.mean()) if d is not None else None
        universe.append({"fiscal_year": int(year), "vs": vs})
    return {"variants": variants,
            "universe": {"per_year": universe, "vs": _vs_summary(universe, labels)}}


def _path_stats(weekly, w):
    """Annualised vol and max drawdown of a fixed-weight weekly path."""
    if weekly.empty:
        return None, None
    port = weekly.to_numpy() @ w
    vol = float(np.std(port, ddof=1) * np.sqrt(WEEKS_PER_YEAR)) if len(port) > 1 else None
    wealth = np.cumprod(1 + port)
    dd = float(np.min(wealth / np.maximum.accumulate(wealth) - 1))
    return vol, dd


def _core_aware_weights(hist, kept, core_close, decision, lookback, alpha_src, cap):
    """A and B optimised for core + satellite, on the same ``kept`` names, or
    {} when the core's history does not cover the window. Kept separate from
    the satellite-alone fit so adding the core can never change which names
    the equal / A / B comparison runs on."""
    if core_close is None:
        return {}
    c = core_close[(core_close.index >= lookback) & (core_close.index < decision)]
    rets, _ = common_weekly_returns({**{k: hist[k] for k in kept}, CORE_KEY: c})
    if CORE_KEY not in rets.columns or any(k not in rets.columns for k in kept):
        return {}
    S_full, _ = ledoit_wolf(rets[kept + [CORE_KEY]].to_numpy())
    n = len(kept)
    S = S_full[:n, :n]
    core = (S_full[:n, n], float(S_full[n, n]), 1.0 - SATELLITE_SHARE, SATELLITE_SHARE)
    out = {"minvar_core": _solve(S, cap, core=core)}
    alpha = grinold_alpha(alpha_src, np.sqrt(np.diag(S) * WEEKS_PER_YEAR), kept)
    if alpha is not None and np.any(alpha > 0):
        out["maxsharpe_core"] = _solve(S, cap, alpha, core=core)
    return out


def weighting_backtest(df, closes_sek, benches, top_n=TOP_N, cap=WEIGHT_CAP,
                       core_close=None):
    """Layer 3: the live basket as equal / min-variance / max-Sharpe.

    All three run on the same subset -- basket names with enough verified
    price history for a covariance -- so the comparison is like for like.
    """
    labels = list(benches)
    years = []
    for year, g in df.groupby("fiscal_year"):
        b = basket(g, LIVE_KEY, top_n).set_index("company_id")
        if b.empty:
            continue
        decision = b["report_date"].min()
        lookback = decision - pd.DateOffset(years=LOOKBACK_YEARS)
        hist = {}
        for c in b.index:
            s = closes_sek.get(c)
            if s is not None:
                s = s[(s.index >= lookback) & (s.index < decision)]
                if not s.empty:
                    hist[c] = s
        rets, _ = common_weekly_returns(hist)
        if rets.shape[1] < 2:
            continue
        kept = list(rets.columns)
        S, delta = ledoit_wolf(rets.to_numpy())
        sigma = np.sqrt(np.diag(S) * WEEKS_PER_YEAR)
        alpha = grinold_alpha(g.set_index("company_id")[LIVE_KEY], sigma, kept)
        weights = {"equal": np.full(len(kept), 1.0 / len(kept)),
                   "minvar": _solve(S, cap)}
        if alpha is not None and np.any(alpha > 0):
            weights["maxsharpe"] = _solve(S, cap, alpha)
        weights.update(_core_aware_weights(
            hist, kept, core_close, decision, lookback,
            g.set_index("company_id")[LIVE_KEY], cap))

        # Realised path over the median row's window -- an approximation,
        # since each row's own window is staggered around it.
        mid = b.loc[kept, "report_date"].median()
        fwd = {c: closes_sek[c][(closes_sek[c].index >= mid)
                                & (closes_sek[c].index <= mid + pd.DateOffset(years=1))]
               for c in kept}
        weekly = pd.DataFrame({c: s.resample("W-FRI").last().pct_change(fill_method=None)
                               for c, s in fwd.items()}).iloc[1:].dropna()

        rec = {"fiscal_year": int(year), "n": len(kept), "n_basket": int(len(b)),
               "weeks": len(rets), "shrinkage": float(delta)}
        for key, w in weights.items():
            excess = float(b.loc[kept, TARGET].to_numpy() @ w)
            vs = {}
            for bl, series in benches.items():
                d = _row_vs_bench(b.loc[kept].reset_index(), series)
                vs[bl] = float(d @ w) if d is not None else None
            vol, dd = _path_stats(weekly, w)
            rec[key] = {"excess": excess, "vol": vol, "max_dd": dd,
                        "eff_n": float(1.0 / np.sum(w ** 2)), "vs": vs,
                        "weights": {str(c): float(x) for c, x in zip(kept, w)}}
        years.append(rec)

    summary = {}
    for key, _ in PORTFOLIOS:
        rows = [y[key] for y in years if key in y]
        if not rows:
            continue
        exc = [r["excess"] for r in rows]
        vols = [r["vol"] for r in rows if r["vol"] is not None]
        dds = [r["max_dd"] for r in rows if r["max_dd"] is not None]
        beat_eq = [y[key]["excess"] > y["equal"]["excess"]
                   for y in years if key in y and key != "equal"]
        summary[key] = {
            "mean_excess": float(np.mean(exc)),
            "years_excess_positive": int(sum(v > 0 for v in exc)),
            "mean_vol": float(np.mean(vols)) if vols else None,
            "worst_dd": float(np.min(dds)) if dds else None,
            "years_beating_equal": int(sum(beat_eq)) if key != "equal" else None,
            "n_years": len(rows),
            "vs": _vs_summary(rows, labels),
        }
    return {"per_year": years, "summary": summary}


def run_evaluation(scores_path="data/panel_scores.csv", out_path=RESULTS_PATH,
                   data_dir="data", top_n=TOP_N):
    """The whole of --evaluate: compute, print, and save the JSON the email's
    status block renders (copy it to the Pi alongside nothing else)."""
    from analyzer.panel import convert_closes_to_sek, load_verified_yahoo_closes

    df = load_panel(scores_path)
    if df.empty:
        raise RuntimeError(f"no usable fiscal years in {scores_path}")
    benches = load_benchmarks()
    closes, symbols = load_verified_yahoo_closes(data_dir)
    closes_sek, _ = convert_closes_to_sek(closes, symbols)
    core = load_core(CORE_MIX)

    result = {
        "evaluated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "fiscal_years": sorted(int(y) for y in df["fiscal_year"].unique()),
        "n_companies": int(df["company_id"].nunique()),
        "top_n": top_n,
        "benchmarks": {k: [str(v.index.min().date()), str(v.index.max().date())]
                       for k, v in benches.items()},
        "return_basis": "total return in SEK; excess = vs universe mean that year",
        **score_usefulness(df, benches, top_n),
        "weighting": weighting_backtest(df, closes_sek, benches, top_n,
                                        core_close=core),
    }
    print("\n".join(format_report(result)))
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved {out_path} -- copy it to the Pi for the email status block.")
    return result


# ----------------------------------------------------------------------
# Rendering -- the terminal report and the email status block
# ----------------------------------------------------------------------

def _p(x, signed=True):
    if x is None:
        return "n/a"
    return f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%"


def _vs_cell(v):
    if not v or v.get("mean") is None:
        return "n/a"
    return f"{_p(v['mean'])} {v['years_beating']}/{v['n_years']}"


def _bench_head(result):
    return list(result.get("benchmarks") or {})


def format_report(result):
    """Full terminal report for --evaluate."""
    years = result["fiscal_years"]
    benches = _bench_head(result)
    span = f"FY{years[0]}-{years[-1]}, {len(years)} years" if years else "no years"
    out = ["", "=" * 78,
           f"  IS THE SCORE USEFUL?  top {result['top_n']} after the sleeve gate, "
           "equal stock weights",
           f"  {span}, {result['n_companies']} companies. {result['return_basis']}.",
           "=" * 78]
    head = f"  {'variant':<16}{'IC':>7}{'p':>6}{'IC>0':>6}{'basket/yr':>11}{'p':>6}{'>univ':>7}"
    head += "".join(f"{b:>18}" for b in benches)
    out.append(head)
    for v in result["variants"]:
        n = v["n_periods"]
        line = (f"  {v['label']:<16}{v['mean_ic']:>+7.3f}"
                f"{_fmt_p(v['ic_p_value']):>6}{v['years_ic_positive']:>4}/{n}"
                f"{_p(v['mean_excess']):>11}{_fmt_p(v['excess_p_value']):>6}"
                f"{v['years_excess_positive']:>5}/{n}")
        line += "".join(f"{_vs_cell(v['vs'].get(b)):>18}" for b in benches)
        out.append(line)
    uni = result["universe"]["vs"]
    out.append(f"  {'universe (all)':<16}{'':>48}"
               + "".join(f"{_vs_cell(uni.get(b)):>18}" for b in benches))
    out.append("  vs index cells: mean per-row excess over the index's own window, "
               "years beating / years covered.")

    w = result["weighting"]
    out += ["", f"  WEIGHTING  (live score's basket, names with >=2y verified prices)"]
    out.append(f"  {'portfolio':<26}{'excess/yr':>10}{'>0':>6}{'>equal':>8}"
               f"{'vol':>8}{'worst DD':>10}" + "".join(f"{b:>18}" for b in benches))
    for key, label in PORTFOLIOS:
        s = w["summary"].get(key)
        if not s:
            continue
        beq = "-" if s["years_beating_equal"] is None else \
            f"{s['years_beating_equal']}/{s['n_years']}"
        out.append(f"  {label:<26}{_p(s['mean_excess']):>10}"
                   f"{s['years_excess_positive']:>4}/{s['n_years']}{beq:>8}"
                   f"{_p(s['mean_vol'], False):>8}{_p(s['worst_dd']):>10}"
                   + "".join(f"{_vs_cell(s['vs'].get(b)):>18}" for b in benches))
    out.append("  vol / DD: fixed weights over the median row's year -- approximate.")
    out += [""] + confidence_lines(result)
    return out


def _fmt_p(p):
    return "n/a" if p is None else f"{p:.2f}"


def confidence_lines(result):
    live = next((v for v in result.get("variants", []) if v["key"] == LIVE_KEY), None)
    n = len(result.get("fiscal_years") or [])
    if live and live["ic_p_value"] is not None and live["ic_p_value"] < 0.05:
        verdict = (f"CONFIDENCE: MODERATE — the live score's IC beats chance "
                   f"across {n} years (p={live['ic_p_value']:.2f}).")
    else:
        p = f", IC p={live['ic_p_value']:.2f}" if live and live["ic_p_value"] is not None else ""
        verdict = f"CONFIDENCE: LOW — {n} years only{p}. Directional, not proof."
    return [verdict,
            "  Survivorship bias is not corrected (today's universe applied "
            "backwards) and inflates every number above, most of all vs index."]


# Results older than this are flagged in the email: the panel gains a fiscal
# year roughly once a year, after a --save / --backfill-panel.
STALE_AFTER_DAYS = 120


def load_status(path):
    """The saved evaluation, or None when absent/unreadable (a fresh clone,
    or a Pi that has not been sent the file)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) and "variants" in data else None


def format_status(result):
    """The compact block at the top of the email and the watchlist report."""
    if not result:
        return ["SYSTEM STATUS: unknown — no evaluation found.",
                "  Run --evaluate on the Mac and copy evaluation_results.json across."]
    when = str(result.get("evaluated_at") or "unknown")[:10]
    stale = ""
    try:
        age = (pd.Timestamp.now() - pd.Timestamp(result["evaluated_at"])).days
        if age >= STALE_AFTER_DAYS:
            stale = f"  [STALE — {age} days old, re-run --evaluate]"
    except Exception:
        pass
    years = result.get("fiscal_years") or []
    span = f"FY{years[0]}-{years[-1]}" if years else "?"
    benches = _bench_head(result)
    lines = [f"SYSTEM STATUS  (evaluated {when}, equal metric weights){stale}",
             f"  Is the score useful?  {span}, top {result.get('top_n', TOP_N)} "
             "vs universe mean (total return, SEK):"]
    for v in result.get("variants", []):
        n = v["n_periods"]
        lines.append(f"    {v['label']:<15} {_p(v['mean_excess'])}/yr, "
                     f"beat universe {v['years_excess_positive']}/{n} yrs, "
                     f"IC {v['mean_ic']:+.3f} (p={_fmt_p(v['ic_p_value'])})")
    live = next((v for v in result.get("variants", []) if v["key"] == LIVE_KEY), None)
    if live and benches:
        bits = [f"{b} {_vs_cell(live['vs'].get(b))}" for b in benches]
        lines.append("  Live basket vs index (yrs beating/covered): " + ", ".join(bits))
        uni = result.get("universe", {}).get("vs", {})
        lines.append("  Whole universe vs index:                   " + ", ".join(
            f"{b} {_vs_cell(uni.get(b))}" for b in benches))
    s = (result.get("weighting") or {}).get("summary") or {}
    if s:
        cells = []
        for key, label in PORTFOLIOS:
            if key in s:
                beq = (f", beat equal {s[key]['years_beating_equal']}/{s[key]['n_years']}"
                       if s[key]["years_beating_equal"] is not None else "")
                cells.append(f"{label} {_p(s[key]['mean_excess'])}/yr "
                             f"vol {_p(s[key]['mean_vol'], False)}{beq}")
        lines.append("  Weighting: " + cells[0])
        lines += [f"             {c}" for c in cells[1:]]
    lines += confidence_lines(result)
    return lines
