"""Fiscal-year-anchored cross-sectional panel (new validation pipeline).

This module builds a panel with one row per ``(company_id, fiscal_year)`` from
the same ``data/*.csv`` snapshots the rolling-window backtest reads. It is
purely additive: it reuses ``historical_calc.py``'s helpers directly and does
not modify the existing rolling-window pipeline in any way. The old pipeline
(``calculate_metrics_given_hist`` / ``metrics_by_timespan.csv``) stays a
working fallback for weight optimization until this one is trusted. Per-company
reliability (an earlier use of this panel) was tried at both company and
sector granularity and dropped entirely -- the ~5yr live OHLC ceiling leaves
too little forward-return depth for either to detect real signal, old
mechanism or new (see git history for `_compute_panel_reliability`/
`_compute_reliability`, both removed).

Known limitations (documented, not fixed here):

* **Survivorship bias.** Today's preset/watchlist membership is applied
  retroactively to every fiscal year. Avanza exposes no point-in-time index
  constituents endpoint (confirmed by a full audit of every read method in the
  ``avanza-api`` library — nothing resembling historical index membership
  exists), so this is not fixable from this data source, only documentable.

* **Restated-vs-as-reported.** Empirically the annual fundamentals appear
  frozen/as-reported: comparing the same fiscal year's net_profit / roe /
  total_assets across 3 real snapshot dates spanning ~5 months (Feb -> Jun ->
  Jul) was bit-for-bit identical (4 metrics x 3 companies, zero deviations).
  Caveat: that window never observed an actual fiscal-year rollover — worth
  re-checking after the next one.

* **``fiscal_year = report_date.year`` approximation.** Companies report at
  different times of year (ABB always Feb, others always May), so a single
  calendar year bundles reports published at slightly different points. This is
  acceptable because each company's own report-month is stable year over year,
  and cross-sectional comparison only needs "roughly the same point in time,"
  not exact fiscal-period alignment.

* **Investment companies are absent entirely.** ``_iter_fiscal_years`` derives
  a company's report dates from ``revenue_year``, and the investment-company
  branch of ``get_data`` never records one (its snapshots carry no
  ``revenue_year``/``de_ratio``/``netDebtEbitdaRatio``/quarterly series). So
  Investor, Industrivärden, Öresund, Latour, Kinnevik and Ratos contribute
  zero rows here and every panel statistic excludes them -- while they remain
  live, scored, and eligible for the watchlist. Nothing in the validation
  covers that part of the universe.

* **Dividend timing is prorated, not exact.** The forward target is a total
  return (Avanza's close is not dividend-adjusted -- see
  ``forward_dividend_yield``), but ``dividend_per_share`` carries calendar-year
  *payment*-year labels rather than ex-dividend dates, so a 12-month window
  straddling two label years is split by elapsed fraction. Exact when the
  payout is flat year over year, approximate when it steps. The most recent
  label year can also be a partial/forecast figure (e.g. Coca-Cola's 2026 entry
  shows 1.06 against a 2.04 full year), which understates rather than inflates.

* **``earnings quality status``** is permanently all-NaN historically
  (operatingCashFlow is Nordic-only live, never in the CSV snapshot schema).
  The column is kept (not dropped) so the schema matches production 1:1 and the
  existing coverage-fallback logic in ``calculate_score`` handles it
  automatically.
"""

import numpy as np
import pandas as pd

from analyzer.config import MIN_CROSS_SECTION, MIN_YEAR_COVERAGE
from analyzer.metrics import RATIO_SPECS
from analyzer.historical_calc import (
    get_hist_data,
    slice_df_upto,
    price_cagr_window,
    _safe_last,
    _safe_div,
    _to_pct,
    _build_ticker_dicts,
)
from analyzer.correlation import _all_scored_metrics, _score_snapshot
from analyzer.financial_metrics import (
    calculate_revenue_y_cagr,
    calculate_eps_y_cagr,
    calculate_revenue_yoy_hit_rate,
    calculate_eps_yoy_hit_rate,
    calculate_net_margin_vs_avg,
    calculate_roe_vs_avg,
    calculate_gross_margin_stability,
    calculate_piotroski_f_score,
    calculate_revenue_trend,
    calculate_earnings_quality,
    calculate_dividend_growth,
    calculate_dividend_yield,
)


# metric name lists mirror calculate_metrics_given_hist exactly
_OTHER_KEYS = [
    "revenue trend year status",
    "net debt - ebitda status",
    "net margin vs avg status",
    "roe vs avg status",
    "revenue yoy hit-rate status",
    "eps yoy hit-rate status",
    "eps y cagr status",
    "revenue y cagr status",
    "gross margin stability status",
    "piotroski f-score status",
    "price momentum status",
    "dividend yield status",
    "earnings quality status",
    "dividend growth status",
]

# columns not treated as as-of metric time series (mirrors excl_cols)
_EXCL_COLS = {"company", "name", "sector", "ohlc", "market_cap", "currency"}


def _iter_fiscal_years(row):
    """Return the sorted unique report dates for a company.

    Every ``companyFinancialsByYear``-sourced column shares the same
    report-date index per company, so the ``revenue_year`` date column is a
    faithful list of that company's annual report dates (e.g. ABB's 8 dates
    2019-02-28 -> 2026-01-29).
    """
    ry = row.get("revenue_year")
    if not isinstance(ry, pd.DataFrame) or "date" not in ry.columns:
        return []
    dates = pd.to_datetime(ry["date"], errors="coerce").dropna()
    return sorted(dates.unique())


def build_fundamentals_panel(data_dir="data") -> pd.DataFrame:
    """One row per ``(company_id, fiscal_year)`` of as-of-report-date metrics.

    Near-verbatim reuse of ``calculate_metrics_given_hist``'s inner body, with
    the rolling-window loop replaced by a per-fiscal-year loop. The as-of cut is
    ``<= report_date`` (inclusive): this fiscal year's own value is meant to be
    the last element, unlike the old pipeline's ``start_d`` cut which was
    deliberately *before* the window. There is no window concept here, so
    ``ohlc_win`` / ``total_return`` / ``fund_forward_score`` are dropped. Forward
    returns are attached later, in ``build_scores_panel`` (step 3).
    """
    df = get_hist_data(data_dir)
    pre_metrics = [c for c in df.columns if c not in _EXCL_COLS]

    results = []
    for company, row in df.iterrows():
        sector = row.get("sector", "Unknown")
        ohlc_df = row["ohlc"]

        for report_date in _iter_fiscal_years(row):
            report_date = pd.Timestamp(report_date)

            # as-of slices: only data known by report_date (inclusive of this
            # fiscal year's own report).
            asof = {
                k: slice_df_upto(row[k], report_date)
                for k in pre_metrics
                if k in row
            }

            # ---- Base fields for ratios ----
            pe_val = _safe_last(asof.get("pe"))
            de_val = _safe_last(asof.get("de_ratio")) or _safe_last(asof.get("de"))
            roe_val = _safe_last(asof.get("roe"))
            fcfy_val = _safe_last(asof.get("free_cashflow_yield")) or _safe_last(
                asof.get("fcfy")
            )

            # CAGR proxy as-of report_date (price-based YoY branch): last close
            # <= report_date over last close < report_date-1yr.
            price_cagr = None
            if isinstance(ohlc_df, pd.DataFrame) and "close" in ohlc_df.columns:
                price_cagr = price_cagr_window(
                    ohlc_df["close"],
                    report_date - pd.DateOffset(years=1),
                    report_date,
                    1,
                )
            if isinstance(price_cagr, np.floating):
                price_cagr = float(price_cagr)

            # ---- Build adapter dicts for financial_metrics functions ----
            ticker_analysis, ticker_info = _build_ticker_dicts(asof)

            entry = {
                "company": company,
                "company_id": company,
                "fiscal_year": report_date.year,
                "report_date": str(report_date.date()),
                "sector": sector,
                "pe": pe_val,
                "de": de_val,
                "roe": roe_val,
                "fcfy": fcfy_val,
                "cagr": price_cagr,
            }

            # ---- Use financial_metrics functions (same as live flow) ----
            try:
                rev_trend_y, _, _, _ = calculate_revenue_trend(ticker_analysis)
                entry["revenue trend year status"] = rev_trend_y
            except Exception:
                entry["revenue trend year status"] = None

            nde_val = _safe_last(asof.get("netDebtEbitdaRatio"))
            entry["net debt - ebitda status"] = nde_val

            try:
                rev_cagr, _ = calculate_revenue_y_cagr(ticker_analysis)
                entry["revenue y cagr status"] = rev_cagr
            except Exception:
                entry["revenue y cagr status"] = None

            try:
                eps_cagr, _ = calculate_eps_y_cagr(ticker_analysis)
                entry["eps y cagr status"] = eps_cagr
            except Exception:
                entry["eps y cagr status"] = None

            try:
                rev_hit, _ = calculate_revenue_yoy_hit_rate(
                    ticker_analysis, lookback_quarters=12
                )
                entry["revenue yoy hit-rate status"] = rev_hit
            except Exception:
                entry["revenue yoy hit-rate status"] = None

            try:
                eps_hit, _ = calculate_eps_yoy_hit_rate(
                    ticker_analysis, lookback_quarters=12
                )
                entry["eps yoy hit-rate status"] = eps_hit
            except Exception:
                entry["eps yoy hit-rate status"] = None

            try:
                nm_vs, _ = calculate_net_margin_vs_avg(
                    ticker_info, ticker_analysis, years=5
                )
                entry["net margin vs avg status"] = nm_vs
            except Exception:
                entry["net margin vs avg status"] = None

            try:
                roe_vs, _ = calculate_roe_vs_avg(ticker_info, ticker_analysis, years=5)
                entry["roe vs avg status"] = roe_vs
            except Exception:
                entry["roe vs avg status"] = None

            try:
                gm_stab = calculate_gross_margin_stability(ticker_analysis)
                entry["gross margin stability status"] = gm_stab
            except Exception:
                entry["gross margin stability status"] = None

            try:
                f_score = calculate_piotroski_f_score(
                    ticker_analysis, ticker_info, fcfy_val, de_val, roe_val
                )
                entry["piotroski f-score status"] = f_score
            except Exception:
                entry["piotroski f-score status"] = None

            try:
                entry["dividend yield status"] = calculate_dividend_yield(ticker_info)
            except Exception:
                entry["dividend yield status"] = None

            try:
                eq = calculate_earnings_quality(ticker_info, ticker_analysis)
                entry["earnings quality status"] = eq
            except Exception:
                entry["earnings quality status"] = None

            try:
                dg = calculate_dividend_growth(ticker_analysis, years=3)
                entry["dividend growth status"] = dg
            except Exception:
                entry["dividend growth status"] = None

            # price momentum: price / SMA200 as-of report_date
            try:
                close = ohlc_df["close"]
                close_to_date = close[close.index <= report_date]
                if len(close_to_date) >= 200:
                    sma200 = close_to_date.iloc[-200:].mean()
                    last_price = close_to_date.iloc[-1]
                    entry["price momentum status"] = float(last_price / sma200) - 1.0
                else:
                    entry["price momentum status"] = None
            except Exception:
                entry["price momentum status"] = None

            # ---- Ratios (sector-agnostic, using RATIO_SPECS) ----
            for rk, spec in RATIO_SPECS.items():
                if rk in entry and entry[rk] is not None:
                    continue
                num_name = spec["num"]
                den_name = spec["den"]
                num_is_rate = spec.get("num_is_rate", False)

                if num_name == "cagr":
                    num_val = price_cagr
                elif num_name == "roe":
                    num_val = roe_val
                elif num_name == "fcfy":
                    num_val = fcfy_val
                else:
                    num_val = _safe_last(asof.get(num_name))

                den_val = (
                    pe_val
                    if den_name == "pe"
                    else (
                        de_val
                        if den_name == "de"
                        else _safe_last(asof.get(den_name))
                    )
                )

                if num_is_rate:
                    num_val = _to_pct(num_val, force_convert=True)

                den_floor = spec.get("den_floor")
                if den_floor is not None and den_val is not None:
                    try:
                        den_val = float(den_val)
                        if abs(den_val) < den_floor:
                            den_val = den_floor if den_val >= 0 else -den_floor
                    except (TypeError, ValueError):
                        pass

                entry[rk] = _safe_div(num_val, den_val)

            results.append(entry)

    panel = dedupe_fiscal_years(pd.DataFrame(results))
    _print_coverage_summary(panel)
    return panel


def dedupe_fiscal_years(panel):
    """One row per (company_id, fiscal_year), keeping the latest report.

    ``fiscal_year`` is ``report_date.year``, so a company that files twice in
    one calendar year yields two rows sharing a key. Observed in real data:
    Munchener Ruck filed 2019-02-06 and 2019-09-26; Siemens filed 2019-11-07
    and 2019-11-08.

    Latent until the Yahoo backfill pulled 2019 into the panel, at which point
    it crashed the challenger gate -- duplicate index labels make the return
    series longer than the score series, and the resulting boolean mask no
    longer matches. Short of crashing it would have quietly double-weighted
    those companies in that year's cross-section.

    The later report wins: it is the most recent information available as of
    that fiscal year, consistent with the as-of cut everywhere else.
    """
    if panel is None or panel.empty:
        return panel
    if not {"company_id", "fiscal_year", "report_date"} <= set(panel.columns):
        return panel
    before = len(panel)
    out = (panel.sort_values(["company_id", "fiscal_year", "report_date"])
                .drop_duplicates(["company_id", "fiscal_year"], keep="last")
                .reset_index(drop=True))
    if len(out) < before:
        print(f"[panel] collapsed {before - len(out)} duplicate "
              f"(company, fiscal_year) row(s) — multiple reports in one "
              f"calendar year; kept the latest")
    return out


def drop_thin_years(df, return_col, min_n=MIN_CROSS_SECTION,
                    min_coverage=MIN_YEAR_COVERAGE, label="panel"):
    """Drop fiscal years that cannot serve as evidence.

    Two independent tests, because each catches a failure the other misses:

    * **Size** (``min_n``) -- a year needs enough companies for a top-vs-bottom
      sort to mean anything. FY2021 had 9, scored as 3-vs-3, and reported
      IC +0.867.
    * **Coverage** (``min_coverage``) -- a year also needs to be *the universe*,
      not a subsample of it. A partial Yahoo backfill gives the earliest years
      a forward return only for the backfilled companies: FY2019 arrived with
      27 companies, comfortably over the size floor, every one of them a large
      cap from a 30-symbol partial run, against 87 with fundamentals that year.
      Size alone would have admitted it.

    Both are judged on rows with a *usable target*, not raw row count -- FY2026
    rows exist as the current cross-section but have no forward return yet and
    must not prop a year up. ``df`` should therefore still contain its
    target-less rows, which form the coverage denominator.

    Passes the frame through untouched when it is empty or has no
    ``return_col``, so callers holding pre-target frames aren't silently
    emptied. Prints what it dropped and why.
    """
    if df is None or df.empty or return_col not in df.columns:
        return df
    usable = df[df[return_col].notna()]
    if usable.empty:
        return df.iloc[0:0]

    sizes = usable.groupby("fiscal_year").size()
    totals = df.groupby("fiscal_year").size()

    keep, dropped = set(), []
    for fy, n in sizes.items():
        total = int(totals.get(fy, n))
        coverage = n / total if total else 0.0
        if n < min_n:
            dropped.append(f"{int(fy)} (n={int(n)} < {min_n})")
        elif coverage < min_coverage:
            dropped.append(
                f"{int(fy)} (only {coverage:.0%} of {total} companies have a "
                f"return — biased subsample)"
            )
        else:
            keep.add(fy)
    if dropped:
        print(f"[{label}] dropped fiscal year(s): {', '.join(dropped)}")
    return df[df["fiscal_year"].isin(keep)].copy()


def load_gate_panel(scores_path="data/panel_scores.csv",
                    fundamentals_path="data/panel_fundamentals.csv"):
    """Merge the raw-metric fundamentals panel with the scored panel's forward
    excess return, keyed on (company_id, fiscal_year), keeping only rows with a
    real 1-year forward excess return.

    The result carries the raw metric/base columns (so the optimizer can
    re-score with candidate weights via _score_with_weights) plus
    fwd_excess_return_1y (the target) and a "company" column for the scorer's
    index. Used only by the panel challenger gate in correlation.py.
    """
    fundamentals = pd.read_csv(fundamentals_path)
    fundamentals.columns = fundamentals.columns.str.strip()
    scores = pd.read_csv(scores_path)
    scores.columns = scores.columns.str.strip()
    ret = scores[["company_id", "fiscal_year", "fwd_excess_return_1y"]]
    merged = fundamentals.merge(ret, on=["company_id", "fiscal_year"], how="left")
    if "company" not in merged.columns:
        merged["company"] = merged["company_id"]
    # Year filtering runs BEFORE dropping target-less rows: those rows are the
    # denominator for the coverage test. Filtering here covers the challenger
    # gate and both panel optimizers at once -- all three source their panel
    # through this function.
    merged = drop_thin_years(merged, "fwd_excess_return_1y", label="gate")
    return merged[merged["fwd_excess_return_1y"].notna()].copy()


def forward_dividend_yield(dps_df, report_date, price_at_report):
    """Dividends paid in ``(report_date, report_date + 1y]``, as a yield.

    Avanza's OHLC close is not dividend-adjusted (Handelsbanken A drops ~12.5%
    on its ex-div day in the raw snapshot; Yahoo's adjusted close moves ~1.2%
    for the same session), so a price-only forward return systematically
    understates high-yield stocks -- exactly the ones `dividend yield status`
    rewards. This adds the missing leg back.

    ``dividend_per_share`` dates are calendar-year labels for the **payment**
    year, not ex-dividend dates (verified against a real payment:
    Handelsbanken's ``2025-12-31: 15.0`` is the 15.00 SEK paid 2025-03-27). So
    the 12-month window starting at ``report_date`` straddles two label years,
    and we prorate by how much of the window falls in each. When DPS is flat
    year over year this is exact; it only approximates when the payout changes,
    and it degrades gracefully (a missing label year falls back to the other).

    Returns 0.0 for a genuine non-payer, NaN when the price is unusable.
    """
    if price_at_report is None:
        return float("nan")
    try:
        price = float(price_at_report)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(price) or price <= 0:
        return float("nan")

    if not isinstance(dps_df, pd.DataFrame) or dps_df.empty:
        return 0.0
    if "date" not in dps_df.columns or "value" not in dps_df.columns:
        return 0.0

    by_year = {}
    for _, e in dps_df.iterrows():
        ts = pd.to_datetime(e["date"], errors="coerce")
        val = pd.to_numeric(e["value"], errors="coerce")
        if pd.notna(ts) and pd.notna(val):
            by_year[ts.year] = float(val)
    if not by_year:
        return 0.0

    report_date = pd.Timestamp(report_date)
    year = report_date.year
    # Fraction of the next 12 months still inside the report's own label year.
    w = (365.0 - report_date.dayofyear) / 365.0
    w = min(max(w, 0.0), 1.0)

    this_year, next_year = by_year.get(year), by_year.get(year + 1)
    if this_year is None and next_year is None:
        return 0.0
    if this_year is None:
        this_year = next_year
    if next_year is None:
        next_year = this_year

    return (w * this_year + (1.0 - w) * next_year) / price


def load_verified_yahoo_closes(data_dir="data"):
    """Verified Yahoo adjusted-close series, or {} when the backfill is absent.

    Kept deliberately quiet and optional: the backfill is a manual Mac-only
    step, so every consumer must work unchanged without it.
    """
    try:
        from analyzer.yahoo_prices import load_symbol_map, load_verified_closes
    except Exception:
        return {}
    symbol_map = load_symbol_map()
    if not symbol_map:
        return {}
    try:
        closes, _ = load_verified_closes(symbol_map, data_dir=data_dir)
        return closes
    except Exception as exc:
        print(f"[panel] Yahoo closes unavailable ({exc}); using Avanza prices")
        return {}


def _close_series(ohlc_df):
    """Sorted close-price Series indexed by date, or empty."""
    if not isinstance(ohlc_df, pd.DataFrame) or "close" not in ohlc_df.columns:
        return pd.Series(dtype=float)
    ser = ohlc_df["close"].copy()
    ser.index = pd.to_datetime(ser.index, errors="coerce")
    return ser[ser.index.notna()].sort_index()


# representative raw-percentile column exposed purely for the within-year
# ranking-span sanity check (item 5, check 3).
_PCT_CHECK_METRIC = "roe_pe ratio status"


def build_scores_panel(fundamentals_df, data_dir="data") -> pd.DataFrame:
    """Score each fiscal year's cross-section and attach 1-year forward returns.

    The single most important correctness property (called out by the source
    spec as "the most likely place for a subtle bug"): every fiscal year is
    scored and ranked **independently** — years are never pooled before
    ranking. We ``groupby("fiscal_year")`` and score each slice on its own.

    Per fiscal year:
      1. Score the cross-section twice — equal-weight (the new default) and
         with today's production HIGHEST/HIGH/LOW tiers (reference column), so
         ``--validate`` can compare live weighting vs equal-weight out of sample.
      2. Attach 1-year forward return per company (price at report -> price one
         year later). Rows whose OHLC doesn't reach the +1y anchor get NaN
         returns (the real OHLC-depth ceiling), but are still scored/ranked.
      3. Demean within the year: subtract that year's universe mean forward
         return. Rank statistics (IC / quintile spread / regression slopes) are
         invariant to this within-year constant shift; it only affects
         absolute-level reporting.
    """
    metrics = _all_scored_metrics()
    df_hist = get_hist_data(data_dir)
    close_map = {
        company: _close_series(df_hist.loc[company, "ohlc"])
        for company in df_hist.index
    }
    dps_map = {
        company: (df_hist.loc[company, "dividend_per_share"]
                  if "dividend_per_share" in df_hist.columns else None)
        for company in df_hist.index
    }

    # Yahoo's adjusted close already contains dividends and reaches back much
    # further than Avanza's rolling ~5y window. Where a symbol is *verified*
    # against the Avanza series (see yahoo_prices.verify_symbol) we use it and
    # must NOT also add dividends back -- double-counting them would inflate
    # exactly the stocks the price-only bug used to penalise. `return_basis`
    # records which leg each row took so this can never be guessed later.
    yahoo_closes = load_verified_yahoo_closes(data_dir)
    if yahoo_closes:
        share = len(yahoo_closes) / max(len(close_map), 1)
        print(f"[panel] using Yahoo adjusted closes for {len(yahoo_closes)} "
              f"of {len(close_map)} companies (dividends already included)")
        if share < 0.9:
            # A partial backfill is worse than none for the early years: only
            # the backfilled companies reach back before Avanza's window, so
            # those cross-sections are a biased subsample rather than the
            # universe, and MIN_CROSS_SECTION would happily admit them.
            print(f"  [WARN] only {share:.0%} of the universe has Yahoo prices. "
                  "Fiscal years older than Avanza's window will contain just "
                  "that subset -- a biased cross-section, not the universe. "
                  "Finish --backfill-prices before trusting the extra years.")

    out_rows = []
    skipped_no_return = 0
    for fiscal_year, df_fy in fundamentals_df.groupby("fiscal_year"):
        df_fy = df_fy.reset_index(drop=True)

        # --- Step 1: score this year's cross-section, twice (per-year only) ---
        scored_eq = _score_snapshot(
            df_fy, metrics_to_score=metrics,
            weight_overrides={m: 1.0 for m in metrics},
        )
        scored_tier = _score_snapshot(df_fy, metrics_to_score=metrics)
        eq_score_cols = [c for c in scored_eq.columns if c.endswith("_score")]

        # within-year percentile rank of the representative raw metric value
        pct_rank = None
        if _PCT_CHECK_METRIC in df_fy.columns:
            pct_rank = (
                pd.to_numeric(df_fy[_PCT_CHECK_METRIC], errors="coerce")
                .rank(pct=True, na_option="keep")
            )

        # --- Step 2: forward returns for this year's rows ---
        fy_rows = []
        for i, r in df_fy.iterrows():
            company = r["company_id"]
            report_date = pd.Timestamp(r["report_date"])
            on_yahoo = company in yahoo_closes
            close = (yahoo_closes[company] if on_yahoo
                     else close_map.get(company, pd.Series(dtype=float)))

            fwd_return = np.nan
            price_at_report = np.nan
            price_at_anchor = np.nan
            anchor_date = report_date + pd.DateOffset(years=1)
            before = close[close.index <= report_date]
            if not before.empty and close.index.max() >= anchor_date:
                price_at_report = float(before.iloc[-1])
                at_anchor = close[close.index <= anchor_date]
                if not at_anchor.empty and price_at_report > 0:
                    price_at_anchor = float(at_anchor.iloc[-1])
                    fwd_return = price_at_anchor / price_at_report - 1.0
            if np.isnan(fwd_return):
                skipped_no_return += 1

            # Dividends over the same window. Only meaningful where a price
            # return exists -- otherwise there is nothing to add it to.
            # On the Yahoo leg the adjusted close already includes them, so the
            # add-back is deliberately zero rather than skipped: the column
            # stays populated and the totals stay comparable across legs.
            if np.isnan(fwd_return):
                fwd_dy = np.nan
            elif on_yahoo:
                fwd_dy = 0.0
            else:
                fwd_dy = forward_dividend_yield(
                    dps_map.get(company), report_date, price_at_report
                )
            fwd_total = fwd_return + fwd_dy if not np.isnan(fwd_dy) else np.nan

            row = {
                "company_id": company,
                "fiscal_year": fiscal_year,
                "report_date": r["report_date"],
                "sector": r.get("sector"),
                "composite_score_equal": _lookup(scored_eq, company, "points"),
                "composite_score_tiered": _lookup(scored_tier, company, "points"),
                "quality_pct": _lookup(scored_eq, company, "quality_pct"),
                "value_pct": _lookup(scored_eq, company, "value_pct"),
                "combined_score": _lookup(scored_eq, company, "combined_score"),
                f"{_PCT_CHECK_METRIC}_pct": (
                    float(pct_rank.iloc[i]) if pct_rank is not None
                    and pd.notna(pct_rank.iloc[i]) else np.nan
                ),
                "price_at_report": price_at_report,
                "price_at_anchor": price_at_anchor,
                "fwd_return_1y": fwd_return,
                "fwd_dividend_yield_1y": fwd_dy,
                "fwd_total_return_1y": fwd_total,
                "return_basis": ("yahoo_adjusted" if on_yahoo
                                 else "avanza_price_plus_dps"),
            }
            for sc in eq_score_cols:
                row[sc] = _lookup(scored_eq, company, sc)
            fy_rows.append(row)

        # --- Step 3: demean within this fiscal year ---
        # fwd_excess_return_1y is the target every downstream consumer reads
        # (the optimizers, the challenger gate, validation), and it is now
        # TOTAL return. The price-only version is kept alongside under an
        # explicit name so nothing can read a stale meaning by accident.
        def _demean(key):
            vals = np.array([r[key] for r in fy_rows], dtype=float)
            mask = ~np.isnan(vals)
            mean = float(vals[mask].mean()) if mask.any() else np.nan
            return mean

        price_mean = _demean("fwd_return_1y")
        total_mean = _demean("fwd_total_return_1y")
        for r in fy_rows:
            r["universe_mean_return_that_year"] = price_mean
            r["universe_mean_total_return_that_year"] = total_mean
            r["fwd_excess_price_return_1y"] = (
                r["fwd_return_1y"] - price_mean
                if not np.isnan(r["fwd_return_1y"]) and not np.isnan(price_mean)
                else np.nan
            )
            r["fwd_excess_return_1y"] = (
                r["fwd_total_return_1y"] - total_mean
                if not np.isnan(r["fwd_total_return_1y"]) and not np.isnan(total_mean)
                else np.nan
            )
        out_rows.extend(fy_rows)

    panel = pd.DataFrame(out_rows)
    _print_scores_summary(panel, skipped_no_return)
    return panel


def _lookup(scored_df, company, col):
    """Safe scalar lookup from a company-indexed scored DataFrame."""
    try:
        if company in scored_df.index and col in scored_df.columns:
            v = scored_df.loc[company, col]
            if isinstance(v, pd.Series):
                v = v.iloc[0]
            return float(v) if pd.notna(v) else np.nan
    except Exception:
        pass
    return np.nan


def _print_scores_summary(panel: pd.DataFrame, skipped_no_return: int) -> None:
    if panel.empty:
        print("[panel] no scored rows produced")
        return
    with_ret = panel["fwd_return_1y"].notna()
    per_company = panel[with_ret].groupby("company_id")["fiscal_year"].nunique()
    print("[panel] scores + forward-return coverage")
    print(f"  scored company-years:       {len(panel)}")
    print(f"  with 1y forward return:     {int(with_ret.sum())}")
    print(f"  skipped (no +1y anchor):    {skipped_no_return}")
    print(f"  distinct fiscal years (ret):{panel[with_ret]['fiscal_year'].nunique()}")
    if not per_company.empty:
        print(f"  companies with returns:     {per_company.size}")
        print(f"  mean usable years/company:  {per_company.mean():.2f}")
        print(f"  min/median/max per co.:     {per_company.min()}/"
              f"{int(per_company.median())}/{per_company.max()}")
        thin = per_company[per_company <= 1]
        for name in thin.index:
            print(f"  [WARN] {name}: only {int(per_company[name])} usable return-year(s)")


def _print_coverage_summary(panel: pd.DataFrame) -> None:
    """Real, live-computed coverage stats (never hardcoded)."""
    if panel.empty:
        print("[panel] no rows produced")
        return
    per_company = panel.groupby("company_id")["fiscal_year"].nunique()
    print("[panel] fundamentals coverage")
    print(f"  companies:                {per_company.size}")
    print(f"  distinct fiscal years:    {panel['fiscal_year'].nunique()} "
          f"({panel['fiscal_year'].min()}-{panel['fiscal_year'].max()})")
    print(f"  total company-years:      {len(panel)}")
    print(f"  mean fiscal years/company:{per_company.mean():.2f}")
    print(f"  min/median/max per co.:   {per_company.min()}/"
          f"{int(per_company.median())}/{per_company.max()}")
    thin = per_company[per_company <= 1]
    for name in thin.index:
        print(f"  [WARN] {name}: only {int(per_company[name])} usable fiscal year(s)")
