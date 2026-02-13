# ----------------------------------------------------------------------
#  Historical metrics generator
#  – Produces one row per <company , window-label>
# ----------------------------------------------------------------------

from pathlib import Path
import json
import numpy as np
import pandas as pd

from metrics import extract_sector
from analyzer.metrics import RATIO_SPECS  # single source of truth
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
)


# ----------------------------------------------------------------------
# Metric ↔ datapoint map
# ----------------------------------------------------------------------
METRIC_TO_DATAPOINT = {
    "roe status": "roe",
    "de status": "de_ratio",
    "profit margin status": "profit_margin",
    "net debt - ebitda status": "netDebtEbitdaRatio",
    "nav discount status": "nav_discount",
    "nav discount trend status": "nav_discount",
    "calculated nav discount status": "calculated_nav_discount",
    "fcf status": "free_cashflow",
    "fcfy status": "free_cashflow_yield",
    "revenue trend year status": "revenue_year",
    "net debt - ebit status": "evEbit",
}


# ------------------------------------------------------------------ helpers
def _series_from_df(obj):
    """Return numeric Series indexed by datetime, or empty Series."""
    if obj is None:
        return pd.Series(dtype=float)

    if isinstance(obj, pd.DataFrame):
        if obj.empty or "value" not in obj.columns:
            return pd.Series(dtype=float)
        df = obj.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        return df.sort_values("date").set_index("date")["value"].astype(float)

    if isinstance(obj, pd.Series):
        return obj.dropna().astype(float).sort_index()

    return pd.Series([obj], dtype=float)


def _trend_metric(obj):
    ser = _series_from_df(obj)
    if len(ser) < 2:
        return None
    return (ser.iloc[-1] - ser.iloc[0]) / abs(ser.iloc[0])


# ------------------------------------------------------------------
# Reduce a datapoint to one value inside the current window
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# reduce a datapoint to one number inside *filtered* window
# ------------------------------------------------------------------
def _aggregate_for_timespan(metric_name: str, obj):
    """
    • If obj is scalar → float(obj)
    • If DataFrame/Series → median of numeric values
    • If obj is *stringified* dict of dates → convert → median
    """
    # -------------------------------------------------- scalar or NaN
    if obj is None or (isinstance(obj, float) and np.isnan(obj)):
        return None
    if isinstance(obj, (int, float)):
        return float(obj)

    # -------------------------------------------------- stringified dict
    if isinstance(obj, str) and obj.strip().startswith("{"):
        try:
            parsed = json.loads(obj)
            # parsed is dict {date: value}
            ser = pd.Series(parsed, dtype=float).dropna().sort_index()
        except (json.JSONDecodeError, ValueError):
            return None
    # -------------------------------------------------- DataFrame
    elif isinstance(obj, pd.DataFrame):
        if obj.empty or "value" not in obj.columns:
            return None
        ser = obj["value"].dropna().astype(float)
    # -------------------------------------------------- Series
    elif isinstance(obj, pd.Series):
        ser = obj.dropna().astype(float)
    else:
        return None

    return ser.median() if not ser.empty else None


def _trend_metric_yoy(obj, window_start):
    ser = _series_from_df(obj)
    if ser.empty:
        return None
    last = ser[ser.index >= window_start]
    prev = ser[ser.index < window_start]
    if last.empty or prev.empty or prev.iloc[-1] == 0:
        return None
    return (last.iloc[-1] - prev.iloc[-1]) / abs(prev.iloc[-1])


def price_cagr_window(close_ser, start, end, yrs_in_span):
    """CAGR for TOTAL windows (first→last in span) or YoY (prev→last)."""
    if close_ser is None or close_ser.empty:
        return None
    in_win = close_ser[(close_ser.index >= start) & (close_ser.index <= end)]
    if in_win.empty:
        return None
    last_val = in_win.iloc[-1]

    if yrs_in_span == 1:  # YoY
        prev = close_ser[close_ser.index < start]
        if prev.empty or prev.iloc[-1] == 0:
            return None
        start_val = prev.iloc[-1]
    else:  # TOTAL
        start_val = in_win.iloc[0]
        if len(in_win) < 2 or start_val == 0:
            return None

    years = (end - start).days / 365.25
    return (last_val / start_val) ** (1 / years) - 1


def slice_df_between(df, start, end):
    if not isinstance(df, pd.DataFrame):
        return df
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        mask = (df["date"] >= start) & (df["date"] <= end)
        return df.loc[mask]
    idx = pd.to_datetime(df.index, errors="coerce")
    return df.loc[(idx >= start) & (idx <= end)]


def make_windows(max_date, span_years):
    """
    Build a list of rolling-window definitions for a given high-level span.

    Window labels follow this scheme
    ─────────────────────────────────────────────────────────────────────
        <N>Y_TOTAL
            • “Cumulative” window: from <max_date  N years>  (inclusive)
                up to <max_date> (inclusive).


        <N>Y_YoY-k
            • “Year-over-year” one-year window.
            • k tells you how many complete years back the window ends:
                    k = 1  →  (max_date − 1 year)  → max_date
                    k = 2  →  (max_date − 2 years) → (max_date − 1 year)
                    …
                    k = N  →  (max_date − N years) → (max_date − N + 1 years)
            • Every YoY window is exactly 1 year long, so yrs_in_span = 1.

    The function returns
        (label, start_date, end_date, yrs_in_span)
    where:
        label          string as described above
        start_date     pd.Timestamp (inclusive)
        end_date       pd.Timestamp (inclusive)
        yrs_in_span    int
                        • N      for the *_TOTAL window
                        • 1      for all *_YoY-k windows
    """

    out = []
    total_start = max_date - pd.DateOffset(years=span_years)
    out.append((f"{span_years}Y_TOTAL", total_start, max_date, span_years))
    for k in range(1, span_years + 1):
        if span_years == 1 and k == 1:
            continue  # skip redundant 1Y_YoY-1
        end = max_date - pd.DateOffset(years=k - 1)
        start = end - pd.DateOffset(years=1)
        out.append((f"{span_years}Y_YoY-{k}", start, end, 1))
    return out


# ------------------------------------------------------------------ IO helpers
def parse_ohlc_series(s):
    df = pd.DataFrame(json.loads(s))
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")[["close"]]


def convert_cell(cell, col):
    if isinstance(cell, str):
        txt = cell.strip()
        if txt.lower() in ("", "nan", "null"):
            return np.nan
        if txt.startswith("[") and txt.endswith(("]", "}")):
            try:
                return convert_cell(json.loads(txt), col)
            except json.JSONDecodeError:
                pass
        try:
            return float(txt)
        except ValueError:
            return txt
    if isinstance(cell, list):
        return pd.DataFrame(cell)
    if isinstance(cell, dict):
        return pd.DataFrame([cell])
    return cell


def get_hist_data():
    frames = []
    for csv in Path("data").glob("*.csv"):
        key = csv.stem.split("_")[0]
        tmp = pd.read_csv(csv)
        tmp.insert(0, "company", key)
        frames.append(tmp)

    df = pd.concat(frames, ignore_index=True).set_index("company", drop=False)
    if "asof" in df.columns:
        df = df.drop(columns=["asof"])

    for col in df.columns:
        if col == "ohlc":
            df[col] = df[col].apply(parse_ohlc_series)
        else:
            df[col] = df[col].apply(lambda c: convert_cell(c, col))
            if col == "sector":
                df[col] = df[col].apply(extract_sector)
    return df.groupby(level=0).first()  # 1 row per company


# ------------------------------------------------------------------ main


# helpers (window calc already exists in your file)
def _unwrap1(x):
    return x[0] if isinstance(x, (list, tuple)) and len(x) == 1 else x


def _to_pct(x, force_convert=False):
    """Convert to percent only when explicitly requested (force_convert=True)."""
    x = _unwrap1(x)
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    return x * 100.0 if force_convert else x


def _safe_last(series):
    try:
        if series is None:
            return None
        # Handle stringified dict like '{"2024-12-31": 0.034, ...}'
        if isinstance(series, str) and series.strip().startswith("{"):
            parsed = json.loads(series)
            vals = [(k, v) for k, v in sorted(parsed.items()) if v is not None]
            return float(vals[-1][1]) if vals else None
        s = _series_from_df(series)
        if s is None or s.empty:
            return None
        v = s.dropna()
        return float(v.iloc[-1]) if not v.empty else None
    except Exception:
        return None


def _safe_div(a, b):
    try:
        if a is None or b is None:
            return None
        a = float(a)
        b = float(b)
        if b == 0:
            return None
        return a / b
    except Exception:
        return None


def _df_to_dict_list(df_or_obj, start=None, end=None):
    """Convert a CSV DataFrame (date, value cols) to list-of-dicts
    that financial_metrics functions expect: [{"value": x, "date": "...", "reportType": "FULL_YEAR"}, ...]
    Optionally filter to [start, end] window.
    """
    if df_or_obj is None:
        return []
    if isinstance(df_or_obj, (int, float)):
        return [{"value": float(df_or_obj), "reportType": "FULL_YEAR"}]
    if isinstance(df_or_obj, str):
        # stringified dict like '{"2024-12-31": 0.034, ...}'
        try:
            parsed = json.loads(df_or_obj)
            out = []
            for d, v in sorted(parsed.items()):
                if v is not None:
                    out.append({"date": d, "value": float(v), "reportType": "FULL_YEAR"})
            return out
        except (json.JSONDecodeError, ValueError):
            return []
    if isinstance(df_or_obj, pd.DataFrame):
        if df_or_obj.empty or "value" not in df_or_obj.columns:
            return []
        df = df_or_obj.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")
            if start is not None:
                df = df[df["date"] >= start]
            if end is not None:
                df = df[df["date"] <= end]
        out = []
        for _, r in df.iterrows():
            entry = {"value": r["value"], "reportType": "FULL_YEAR"}
            if "date" in r:
                entry["date"] = str(r["date"].date()) if hasattr(r["date"], "date") else str(r["date"])
            out.append(entry)
        return out
    return []


def _build_ticker_dicts(row, filtered, start_d, end_d):
    """Build fake ticker_analysis and ticker_info dicts from CSV data
    so we can reuse the same financial_metrics functions as the live flow.

    Uses all data up to end_d (not just within window) because quality
    metrics like CAGR, hit-rate, stability need multi-year history.
    The window only constrains price-based return calculations.
    """
    # Use full history up to end_d (not start_d) so functions have enough data
    revenue_year = _df_to_dict_list(row.get("revenue_year"), end=end_d)
    revenue_quarter = _df_to_dict_list(row.get("revenue_quarter"), end=end_d)
    profit_margin = _df_to_dict_list(row.get("profit_margin"), end=end_d)
    profit_per_share = _df_to_dict_list(row.get("profit_per_share"), end=end_d)
    roe_series = _df_to_dict_list(row.get("roe"), end=end_d)
    de_series = _df_to_dict_list(row.get("de_ratio"), end=end_d)
    net_profit = _df_to_dict_list(row.get("net_profit"), end=end_d)
    total_assets = _df_to_dict_list(row.get("total_assets"), end=end_d)
    total_liab = _df_to_dict_list(row.get("total_liabilities"), end=end_d)
    equity_ps = _df_to_dict_list(row.get("equity_per_share"), end=end_d)
    ev_ebit = _df_to_dict_list(row.get("ev_ebit"), end=end_d)
    div_ps = _df_to_dict_list(row.get("dividend_per_share"), end=end_d)

    ticker_analysis = {
        "companyFinancialsByYear": {
            "sales": revenue_year,
            "profitMargin": profit_margin,
            "debtToEquityRatio": de_series,
            "totalAssets": total_assets,
            "totalLiabilities": total_liab,
            "netProfit": net_profit,
        },
        "companyFinancialsByQuarter": {
            "sales": revenue_quarter,
        },
        "companyKeyRatiosByYear": {
            "earningsPerShare": profit_per_share,
            "returnOnEquityRatio": roe_series,
            "equityPerShare": equity_ps,
        },
        "companyKeyRatiosByQuarterQuarter": {
            "earningsPerShare": [],  # quarterly EPS not stored separately in CSV
        },
        "stockKeyRatiosByYear": {
            "evEbitRatio": ev_ebit,
        },
        "dividendsByYear": {
            "dividendPerShare": div_ps,
        },
    }

    # Latest values for ticker_info (used by net_margin_vs_avg, roe_vs_avg, etc.)
    latest_margin = profit_margin[-1]["value"] if profit_margin else None
    latest_roe = roe_series[-1]["value"] if roe_series else None
    latest_pe = None
    pe_list = _df_to_dict_list(row.get("pe"), end=end_d)
    if pe_list:
        latest_pe = pe_list[-1]["value"]

    # Dividend yield: scalar in CSV (not time-series)
    div_yield_raw = row.get("dividend_yield")
    div_yield = None
    if div_yield_raw is not None:
        try:
            dv = float(div_yield_raw)
            if not np.isnan(dv):
                div_yield = dv
        except (TypeError, ValueError):
            pass

    # Operating cash flow not in CSV, but net_profit + total_assets lets Piotroski work
    ticker_info = {
        "keyIndicators": {
            "netMargin": latest_margin,
            "returnOnEquity": latest_roe,
            "priceEarningsRatio": latest_pe,
            "directYield": div_yield,
        },
    }

    return ticker_analysis, ticker_info


def calculate_metrics_given_hist() -> None:
    df = get_hist_data()

    # metrics to compute (sector-agnostic)
    ratio_keys = list(RATIO_SPECS.keys())
    other_keys = [
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
    metrics = ratio_keys + other_keys

    excl_cols = {"name", "sector", "ohlc", "market_cap", "currency"}
    pre_metrics = [c for c in df.columns if c not in excl_cols]

    results = []

    for company, row in df.iterrows():
        sector = row.get("sector", "Unknown")
        ohlc_df = row["ohlc"]
        max_d = (
            ohlc_df.index.max()
            if isinstance(ohlc_df, pd.DataFrame)
            else pd.Timestamp.today()
        )

        for span in (1, 3, 5):
            for label, start_d, end_d, yrs_span in make_windows(max_d, span):

                # window slices for all available datapoints
                filtered = {
                    k: slice_df_between(row[k], start_d, end_d)
                    for k in pre_metrics
                    if k in row
                }
                ohlc_win = slice_df_between(ohlc_df, start_d, end_d)

                # ---- Total return ----
                try:
                    price_start = ohlc_win["close"].iloc[0]
                    price_end = ohlc_win["close"].iloc[-1]
                    total_return = (
                        ((price_end / price_start) - 1) if price_start > 0 else None
                    )
                except Exception:
                    total_return = None

                # ---- Base fields for ratios ----
                pe_val = _safe_last(filtered.get("pe"))
                de_val = _safe_last(filtered.get("de_ratio")) or _safe_last(filtered.get("de"))
                roe_val = _safe_last(filtered.get("roe"))
                fcfy_val = _safe_last(filtered.get("free_cashflow_yield")) or _safe_last(filtered.get("fcfy"))

                # CAGR proxy inside window (price-based)
                price_cagr = price_cagr_window(
                    ohlc_df["close"], start_d, end_d, yrs_span
                )
                if isinstance(price_cagr, np.floating):
                    price_cagr = float(price_cagr)

                # ---- Build adapter dicts for financial_metrics functions ----
                ticker_analysis, ticker_info = _build_ticker_dicts(
                    row, filtered, start_d, end_d
                )

                # ---- Build row ----
                entry = {
                    "company": company,
                    "sector": sector,
                    "timespan": label,
                    "total_return": total_return,
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

                # net debt / ebitda
                nde_val = _safe_last(filtered.get("netDebtEbitdaRatio"))
                entry["net debt - ebitda status"] = nde_val

                # revenue y cagr
                try:
                    rev_cagr, _ = calculate_revenue_y_cagr(ticker_analysis)
                    entry["revenue y cagr status"] = rev_cagr
                except Exception:
                    entry["revenue y cagr status"] = None

                # eps y cagr
                try:
                    eps_cagr, _ = calculate_eps_y_cagr(ticker_analysis)
                    entry["eps y cagr status"] = eps_cagr
                except Exception:
                    entry["eps y cagr status"] = None

                # revenue yoy hit-rate
                try:
                    rev_hit, _ = calculate_revenue_yoy_hit_rate(ticker_analysis, lookback_quarters=12)
                    entry["revenue yoy hit-rate status"] = rev_hit
                except Exception:
                    entry["revenue yoy hit-rate status"] = None

                # eps yoy hit-rate
                try:
                    eps_hit, _ = calculate_eps_yoy_hit_rate(ticker_analysis, lookback_quarters=12)
                    entry["eps yoy hit-rate status"] = eps_hit
                except Exception:
                    entry["eps yoy hit-rate status"] = None

                # net margin vs avg
                try:
                    nm_vs, _ = calculate_net_margin_vs_avg(ticker_info, ticker_analysis, years=5)
                    entry["net margin vs avg status"] = nm_vs
                except Exception:
                    entry["net margin vs avg status"] = None

                # roe vs avg
                try:
                    roe_vs, _ = calculate_roe_vs_avg(ticker_info, ticker_analysis, years=5)
                    entry["roe vs avg status"] = roe_vs
                except Exception:
                    entry["roe vs avg status"] = None

                # gross margin stability
                try:
                    gm_stab = calculate_gross_margin_stability(ticker_analysis)
                    entry["gross margin stability status"] = gm_stab
                except Exception:
                    entry["gross margin stability status"] = None

                # piotroski f-score
                try:
                    f_score = calculate_piotroski_f_score(
                        ticker_analysis, ticker_info, fcfy_val, de_val, roe_val
                    )
                    entry["piotroski f-score status"] = f_score
                except Exception:
                    entry["piotroski f-score status"] = None

                # dividend yield
                try:
                    from analyzer.financial_metrics import calculate_dividend_yield
                    entry["dividend yield status"] = calculate_dividend_yield(ticker_info)
                except Exception:
                    entry["dividend yield status"] = None

                # earnings quality (OCF / net income)
                try:
                    eq = calculate_earnings_quality(ticker_info, ticker_analysis)
                    entry["earnings quality status"] = eq
                except Exception:
                    entry["earnings quality status"] = None

                # dividend growth (CAGR of dividend per share)
                try:
                    dg = calculate_dividend_growth(ticker_analysis, years=3)
                    entry["dividend growth status"] = dg
                except Exception:
                    entry["dividend growth status"] = None

                # price momentum: price / SMA200 at end of window
                try:
                    close = ohlc_df["close"]
                    # Use data up to end_d for SMA200
                    close_to_end = close[close.index <= end_d]
                    if len(close_to_end) >= 200:
                        sma200 = close_to_end.iloc[-200:].mean()
                        last_price = close_to_end.iloc[-1]
                        entry["price momentum status"] = float(last_price / sma200) - 1.0
                    else:
                        entry["price momentum status"] = None
                except Exception:
                    entry["price momentum status"] = None

                # ---- Ratios (sector-agnostic, using RATIO_SPECS) ----
                for rk, spec in RATIO_SPECS.items():
                    # Skip if we already computed a valid value (e.g. net debt-ebitda from direct extraction)
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
                        num_val = _safe_last(filtered.get(num_name))

                    den_val = (
                        pe_val
                        if den_name == "pe"
                        else (
                            de_val
                            if den_name == "de"
                            else _safe_last(filtered.get(den_name))
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

    pd.DataFrame(results).set_index(["company", "timespan"]).to_csv(
        "metrics_by_timespan.csv"
    )


if __name__ == "__main__":
    calculate_metrics_given_hist()
