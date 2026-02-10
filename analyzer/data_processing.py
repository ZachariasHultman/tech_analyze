import numpy as np
import pandas as pd
from analyzer.metrics import HIGHEST_WEIGHT_METRICS, RATIO_SPECS
from analyzer.helper import *
from analyzer.financial_metrics import *


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
        if num in out.columns and den in out.columns:
            vals = []
            for i in out.index:
                n = _to_pct(out.at[i, num], force_convert=True) if num_is_rate else _unwrap(out.at[i, num])
                d = out.at[i, den]
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
        _, _, _, closing_hist_data = (
            calculate_sma200(avanza, ticker_id)
        )

        # --- revenue trend (year only, quarterly removed as too noisy) ---
        rev_trend_year, _, rev_year_hist, rev_quarter_hist = (
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

        gm_stability = calculate_gross_margin_stability(ticker_analysis)
        manager._update(ticker_name, sector, "gross margin stability status", gm_stability)

        div_yield = calculate_dividend_yield(ticker_info)
        manager._update(ticker_name, sector, "dividend yield status", div_yield)

        f_score = calculate_piotroski_f_score(
            ticker_analysis, ticker_info, fcfy, de_ratio, roe
        )
        manager._update(ticker_name, sector, "piotroski f-score status", f_score)

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

        # --- NEW: dividend yield ---
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

    if get_hist:
        return ticker_name, hist
    else:
        return ticker_name, None


def calculate_score(manager, metrics_to_score=None):

    def apply_scores(summary, template, manager, metrics_to_score=None):
        excluded_columns = {"sector", "points"}

        if isinstance(summary, dict):
            summary = pd.DataFrame(summary).T
        if summary.empty:
            return pd.DataFrame()

        # --- derive sector-agnostic ratios before scoring ---
        summary = enrich_ratios(summary)

        score_data = {}

        for col in template:
            if col in excluded_columns or col not in summary.columns:
                continue
            if metrics_to_score is not None and col not in metrics_to_score:
                continue

            def assign(row):
                return manager._assign_points(row, col)

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
