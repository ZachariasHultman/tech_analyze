import numpy as np
import pandas as pd
from analyzer.metrics import HIGHEST_WEIGHT_METRICS, RATIO_SPECS, DIRECTION_OVERRIDES
from analyzer.config import QUALITY_METRICS, VALUE_METRICS
from analyzer.helper import *
from analyzer.financial_metrics import *


def _extract_yearly_series(ticker_analysis, section, key):
    """Extract a yearly time-series from the Avanza API response."""
    try:
        raw = ticker_analysis.get(section, {}).get(key, [])
        hist = [
            {"date": e["date"], "value": e["value"]}
            for e in raw
            if e.get("reportType") == "FULL_YEAR" and "date" in e and "value" in e
        ]
        return hist if hist else None
    except Exception:
        return None


def _extract_dividend_series(ticker_analysis):
    """Extract dividend per share history from dividendsByYear."""
    try:
        raw = ticker_analysis.get("dividendsByYear", {}).get("dividendPerShare", [])
        hist = [
            {"date": e["date"], "value": e["value"]}
            for e in raw
            if e.get("reportType") == "FULL_YEAR" and "date" in e and "value" in e
        ]
        return hist if hist else None
    except Exception:
        return None


def _unwrap(v):
    return v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v


def _to_pct(x, force_convert=False):
    """Convert a value to percent if `force_convert` is True.

    The old heuristic (multiply by 100 when 0 < |x| < 1) is dangerous
    because legitimate values like PE=0.5, ROE=0.8 get corrupted.
    Now only converts when the caller explicitly says the field is a rate.
    """
    x = _unwrap(x)
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    return x * 100.0 if force_convert else x


def _safe_div(a, b):
    a = _unwrap(a)
    b = _unwrap(b)
    try:
        if a is None or b is None:
            return None
        a = float(a)
        b = float(b)
        if b == 0 or np.isnan(a) or np.isnan(b):
            return None
        return a / b
    except Exception:
        return None


def enrich_ratios(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for out_col, spec in RATIO_SPECS.items():
        if out_col == "net debt - ebitda status":
            continue
        num, den = spec["num"], spec["den"]
        num_is_rate = spec.get("num_is_rate", False)
        den_floor = spec.get("den_floor")
        if num in out.columns and den in out.columns:
            vals = []
            for i in out.index:
                n = _to_pct(out.at[i, num], force_convert=True) if num_is_rate else _unwrap(out.at[i, num])
                d = _unwrap(out.at[i, den])
                # Clamp denominator to floor to prevent blow-up (e.g. ROE/DE when DE≈0)
                if den_floor is not None and d is not None:
                    try:
                        d = float(d)
                        if abs(d) < den_floor:
                            d = den_floor if d >= 0 else -den_floor
                    except (TypeError, ValueError):
                        pass
                vals.append(_safe_div(n, d))
            out[out_col] = vals
        else:
            out[out_col] = None
    return out


def get_data(
    ticker_id,
    ticker_info,
    manager,
    avanza,
    yahoo_ticker,
    get_hist=False,
):
    ticker_analysis = avanza.get_analysis(ticker_id)

    investment = any(
        sector["sectorName"] == "Investmentbolag" for sector in ticker_info["sectors"]
    )

    if get_hist:
        hist = {}

    ticker_name = f'{ticker_info["name"]} {ticker_info["orderbookId"]}'

    if not investment:
        sector = [sector for sector in ticker_info["sectors"]]
        manager._initialize_template(ticker_name, sector)

        # --- OHLC data (needed for CAGR, FCFY) ---
        sma200, weekly_average_close, sma200_slope, closing_hist_data = (
            calculate_sma200(avanza, ticker_id)
        )

        # --- revenue trend (year only, quarterly removed as too noisy) ---
        rev_trend_year, rev_trend_quarter, rev_year_hist, rev_quarter_hist = (
            calculate_revenue_trend(ticker_analysis)
        )
        manager._update(
            ticker_name, sector, "revenue trend year status", rev_trend_year
        )

        # --- valuation/growth/cashflow base fields (sector-agnostic ratios use these) ---
        pe, pe_hist = calculate_PE(ticker_analysis)
        cagr = calculate_closing_CAGR(avanza, ticker_id)
        (fcfy, free_cashflow, fcfy_hist, free_cashflow_hist) = (
            calculate_free_cashflow_yield(yahoo_ticker, ticker_info, closing_hist_data)
        )
        de_ratio, de_ratio_hist = calculate_de(ticker_analysis)
        roe, roe_hist = calculate_roe(ticker_analysis)
        nd_ebitda_ratio, nd_ebitda_hist = extract_netdebt_ebitda_ratio(ticker_analysis)

        # write base inputs (SummaryManager accepts these even if not in template)
        manager._update(ticker_name, sector, "pe", pe[-1] if pe else None)
        manager._update(ticker_name, sector, "cagr", cagr[-1] if cagr else None)
        manager._update(ticker_name, sector, "fcfy", fcfy)
        manager._update(ticker_name, sector, "de", de_ratio)
        manager._update(ticker_name, sector, "roe", roe)
        manager._update(
            ticker_name, sector, "net debt - ebitda status", nd_ebitda_ratio
        )
        # multi-year growth
        rev_cagr_y, _ = calculate_revenue_y_cagr(ticker_analysis)
        eps_cagr_y, _ = calculate_eps_y_cagr(ticker_analysis)
        manager._update(ticker_name, sector, "revenue y cagr status", rev_cagr_y)
        manager._update(ticker_name, sector, "eps y cagr status", eps_cagr_y)

        # consistency
        rev_hit, _ = calculate_revenue_yoy_hit_rate(
            ticker_analysis, lookback_quarters=12
        )
        eps_hit, _ = calculate_eps_yoy_hit_rate(ticker_analysis, lookback_quarters=12)
        manager._update(ticker_name, sector, "revenue yoy hit-rate status", rev_hit)
        manager._update(ticker_name, sector, "eps yoy hit-rate status", eps_hit)

        # quality vs own history
        nm_vs_avg, _ = calculate_net_margin_vs_avg(
            ticker_info, ticker_analysis, years=5
        )
        roe_vs_avg, _ = calculate_roe_vs_avg(ticker_info, ticker_analysis, years=5)
        manager._update(ticker_name, sector, "net margin vs avg status", nm_vs_avg)
        manager._update(ticker_name, sector, "roe vs avg status", roe_vs_avg)

        # --- : gross margin stability ---
        gm_stability = calculate_gross_margin_stability(ticker_analysis)
        manager._update(ticker_name, sector, "gross margin stability status", gm_stability)

        # --- : dividend yield ---
        div_yield = calculate_dividend_yield(ticker_info)
        manager._update(ticker_name, sector, "dividend yield status", div_yield)

        # --- : Piotroski F-Score ---
        f_score = calculate_piotroski_f_score(
            ticker_analysis, ticker_info, fcfy, de_ratio, roe
        )
        manager._update(ticker_name, sector, "piotroski f-score status", f_score)

        # --- : earnings quality (OCF / net income) ---
        eq = calculate_earnings_quality(ticker_info, ticker_analysis)
        manager._update(ticker_name, sector, "earnings quality status", eq)

        # --- : dividend growth ---
        div_growth = calculate_dividend_growth(ticker_analysis, years=3)
        manager._update(ticker_name, sector, "dividend growth status", div_growth)

        # --- Price momentum: price / SMA200 - 1 ---
        if sma200 is not None and weekly_average_close is not None and sma200 > 0:
            momentum = (weekly_average_close / sma200) - 1.0
        else:
            momentum = None
        manager._update(ticker_name, sector, "price momentum status", momentum)

        if get_hist:
            hist["sector"] = sector
            hist["ohlc"] = closing_hist_data
            hist["pe"] = pe_hist
            hist["roe"] = roe_hist
            hist["revenue_year"] = rev_year_hist
            hist["revenue_quarter"] = rev_quarter_hist
            hist["de_ratio"] = de_ratio_hist
            hist["free_cashflow_yield"] = fcfy_hist
            hist["free_cashflow"] = free_cashflow_hist
            hist["netDebtEbitdaRatio"] = nd_ebitda_hist
            hist["profit_margin"] = extract_profit_margin_hist(ticker_analysis)
            hist["profit_per_share"] = extract_eps_hist(ticker_analysis)
            hist["dividend_yield"] = div_yield
            # Full financial data for Piotroski and new metrics
            hist["net_profit"] = _extract_yearly_series(ticker_analysis, "companyFinancialsByYear", "netProfit")
            hist["total_assets"] = _extract_yearly_series(ticker_analysis, "companyFinancialsByYear", "totalAssets")
            hist["total_liabilities"] = _extract_yearly_series(ticker_analysis, "companyFinancialsByYear", "totalLiabilities")
            hist["equity_per_share"] = _extract_yearly_series(ticker_analysis, "companyKeyRatiosByYear", "equityPerShare")
            hist["ev_ebit"] = _extract_yearly_series(ticker_analysis, "stockKeyRatiosByYear", "evEbitRatio")
            hist["dividend_per_share"] = _extract_dividend_series(ticker_analysis)

    else:
        sector = [{"sectorId": "51", "sectorName": "Investmentbolag"}]
        manager._initialize_template(ticker_name, sector)

        # --- OHLC data ---
        sma200, weekly_average_close, sma200_slope, closing_hist_data = (
            calculate_sma200(avanza, ticker_id)
        )

        # --- base fields for ratios ---
        pe, pe_hist = calculate_PE(ticker_analysis)
        cagr = calculate_closing_CAGR(avanza, ticker_id)
        (fcfy, free_cashflow, fcfy_hist, free_cashflow_hist) = (
            calculate_free_cashflow_yield(yahoo_ticker, ticker_info)
        )
        roe, roe_hist = calculate_roe(ticker_analysis)
        nd_ebitda_ratio, nd_ebitda_hist = extract_netdebt_ebitda_ratio(ticker_analysis)

        manager._update(ticker_name, sector, "pe", pe[-1] if pe else None)
        manager._update(ticker_name, sector, "cagr", cagr[-1] if cagr else None)
        manager._update(ticker_name, sector, "fcfy", fcfy)
        manager._update(ticker_name, sector, "roe", roe)
        manager._update(
            ticker_name, sector, "net debt - ebitda status", nd_ebitda_ratio
        )

        # --- NAV fields kept in investment template ---
        (
            nav_discount,
            calculated_nav_discount,
            nav_discount_trend,
            nav_discount_hist,
            calculated_nav_discount_hist,
        ) = calculate_NAV_discount(ticker_info["listing"]["tickerSymbol"])
        manager._update(ticker_name, sector, "nav discount status", nav_discount)
        manager._update(
            ticker_name,
            sector,
            "calculated nav discount status",
            calculated_nav_discount,
        )
        manager._update(
            ticker_name, sector, "nav discount trend status", nav_discount_trend
        )

        # --- : dividend yield ---
        div_yield = calculate_dividend_yield(ticker_info)
        manager._update(ticker_name, sector, "dividend yield status", div_yield)

        if get_hist:
            hist["sector"] = sector
            hist["ohlc"] = closing_hist_data
            hist["pe"] = pe_hist
            hist["roe"] = roe_hist
            hist["nav_discount"] = nav_discount_hist
            hist["calculated_nav_discount"] = calculated_nav_discount_hist
            hist["free_cashflow_yield"] = fcfy_hist
            hist["free_cashflow"] = free_cashflow_hist
            hist["profit_margin"] = extract_profit_margin_hist(ticker_analysis)
            hist["profit_per_share"] = extract_eps_hist(ticker_analysis)
            hist["dividend_yield"] = div_yield
            hist["net_profit"] = _extract_yearly_series(ticker_analysis, "companyFinancialsByYear", "netProfit")
            hist["total_assets"] = _extract_yearly_series(ticker_analysis, "companyFinancialsByYear", "totalAssets")
            hist["total_liabilities"] = _extract_yearly_series(ticker_analysis, "companyFinancialsByYear", "totalLiabilities")
            hist["equity_per_share"] = _extract_yearly_series(ticker_analysis, "companyKeyRatiosByYear", "equityPerShare")
            hist["ev_ebit"] = _extract_yearly_series(ticker_analysis, "stockKeyRatiosByYear", "evEbitRatio")
            hist["dividend_per_share"] = _extract_dividend_series(ticker_analysis)

    if get_hist:
        return ticker_name, hist
    else:
        return ticker_name, None


def calculate_score(manager, metrics_to_score=None, use_cross_sectional_ranks=True):
    """Score all companies held in manager.

    use_cross_sectional_ranks (default True):
        Score each metric by where the company ranks within the peer group
        (cross-sectional percentile, 0–1) rather than against fixed absolute
        thresholds.  This makes every metric sector-agnostic automatically:
        a utility with decent-for-utilities ROE ranks in the top half and
        scores positively without needing a separate utility threshold.
        Falls back to absolute-threshold scoring when fewer than 5 companies
        are present in the group (ranks are unstable at very small n).
    """

    def apply_scores(summary, template, manager, metrics_to_score=None):
        excluded_columns = {"sector", "points"}

        if isinstance(summary, dict):
            summary = pd.DataFrame(summary).T
        if summary.empty:
            return pd.DataFrame()

        # Derive composite ratios (ROE/PE, CAGR/PE, etc.) before scoring
        summary = enrich_ratios(summary)

        # ── Cross-sectional percentile ranks (1 = best in group) ─────────
        # For direction +1 (higher raw value = better): rank 1.0 → highest value.
        # For direction -1 (lower raw value = better): rank 1.0 → lowest value
        # (achieved by inverting the percentile).
        # This encodes direction into the rank so _assign_points_rank can
        # always treat rank 1.0 as "best" uniformly.
        cross_ranks: dict = {}
        skipped_metrics: list = []
        if use_cross_sectional_ranks and len(summary) >= 5:
            # Require a metric to be populated for at least 60% of the peer
            # group before trusting its cross-sectional ranking; sparser
            # metrics fall through to the absolute-threshold path below.
            min_samples = max(3, int(np.ceil(0.6 * len(summary))))
            for col in template:
                if col in excluded_columns or col not in summary.columns:
                    continue
                if metrics_to_score is not None and col not in metrics_to_score:
                    continue
                direction = DIRECTION_OVERRIDES.get(col, +1)
                if col in RATIO_SPECS:
                    direction = RATIO_SPECS[col]["dir"]
                # Historical re-scoring (process_historical) stores values as
                # single-element lists/tuples, e.g. [0.82] — pd.to_numeric
                # can't parse those and would read every row as missing.
                # _assign_points already unwraps this shape; mirror it here
                # so the coverage check sees the real values.
                unwrapped = summary[col].map(
                    lambda v: v[0] if isinstance(v, (list, tuple)) and len(v) >= 1 else v
                )
                vals = pd.to_numeric(unwrapped, errors="coerce")
                # Winsorize cross-sectionally at the 2nd/98th percentile so a
                # single extreme outlier can't dominate the percentile ranks.
                lo, hi = vals.quantile(0.02), vals.quantile(0.98)
                vals = vals.clip(lower=lo, upper=hi)
                if vals.notna().sum() >= min_samples:
                    pct = vals.rank(pct=True, na_option="keep")
                    if direction == -1:
                        pct = 1.0 - pct
                    cross_ranks[col] = pct
                else:
                    skipped_metrics.append(col)

            if skipped_metrics:
                print(
                    "[cross-sectional] too few samples "
                    f"(<{min_samples}/{len(summary)}), using absolute thresholds: "
                    + ", ".join(skipped_metrics)
                )

        score_data = {}

        for col in template:
            if col in excluded_columns or col not in summary.columns:
                continue
            if metrics_to_score is not None and col not in metrics_to_score:
                continue

            if col in cross_ranks:
                _ranks = cross_ranks[col]
                def assign_rank(row, _col=col, _ranks=_ranks):
                    rank_val = _ranks.get(row.name)
                    if rank_val is None or pd.isna(rank_val):
                        return 0.0
                    return manager._assign_points_rank(row, _col, float(rank_val))
                score_data[col + "_score"] = summary.apply(assign_rank, axis=1)
            else:
                def assign(row, _col=col):
                    return manager._assign_points(row, _col)
                score_data[col + "_score"] = summary.apply(assign, axis=1)

        for key, val in score_data.items():
            summary[key] = val

        score_cols = [c for c in score_data if c.endswith("_score")]
        summary["points"] = summary[score_cols].sum(axis=1)

        # Symmetric bonus/penalty: +1 when ALL highest-weight metrics score
        # positively, -1 when ALL score negatively. No bonus otherwise.
        bonus_metrics = [
            col for col in HIGHEST_WEIGHT_METRICS if col in summary.columns
        ]
        existing_bonus_score_cols = [col + "_score" for col in bonus_metrics]

        if metrics_to_score is None or set(HIGHEST_WEIGHT_METRICS).issubset(
            metrics_to_score
        ):
            if all(col in summary.columns for col in existing_bonus_score_cols):
                all_positive = (
                    (summary[existing_bonus_score_cols] > 0).all(axis=1).astype(int)
                )
                all_negative = (
                    (summary[existing_bonus_score_cols] < 0).all(axis=1).astype(int)
                )
                summary["points"] += all_positive - all_negative

        # ── Two-sleeve scoring ───────────────────────────────────────────
        # Split the flat point total into a quality sub-sum and a value
        # sub-sum, percentile-rank each cross-sectionally (same primitive as
        # the metric ranks above), and multiply the two percentiles so a
        # stock scores well only when it is good on BOTH dimensions.
        # `points` (above) is kept unchanged for continuity / legacy consumers.
        quality_score_cols = [
            m + "_score" for m in QUALITY_METRICS if m + "_score" in summary.columns
        ]
        value_score_cols = [
            m + "_score" for m in VALUE_METRICS if m + "_score" in summary.columns
        ]
        quality_raw = (
            summary[quality_score_cols].sum(axis=1)
            if quality_score_cols
            else pd.Series(0.0, index=summary.index)
        )
        value_raw = (
            summary[value_score_cols].sum(axis=1)
            if value_score_cols
            else pd.Series(0.0, index=summary.index)
        )
        summary["quality_pct"] = quality_raw.rank(pct=True, na_option="keep")
        summary["value_pct"] = value_raw.rank(pct=True, na_option="keep")
        summary["combined_score"] = summary["quality_pct"] * summary["value_pct"]

        return summary

    manager.summary = apply_scores(
        manager.summary, manager.template, manager, metrics_to_score
    )
    manager.summary_investment = apply_scores(
        manager.summary_investment,
        manager.template_investment,
        manager,
        metrics_to_score,
    )
