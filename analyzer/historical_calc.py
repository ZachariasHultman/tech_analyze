# ----------------------------------------------------------------------
#  Snapshot reader + as-of adapters
#  – Loads data/*.csv snapshots and reshapes them into the Avanza-API-shaped
#    dicts financial_metrics.py expects, cut to a given as-of date.
#  – Consumed by analyzer/panel.py, which builds the fiscal-year panel.
# ----------------------------------------------------------------------

from pathlib import Path
import json
import numpy as np
import pandas as pd

from analyzer.metrics import extract_sector


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


def slice_df_upto(df, end):
    """Return only data known as-of `end` (date <= end) — no look-ahead.

    Handles DataFrame-with-date-column, date-indexed DataFrame, the
    stringified-JSON-dict shape used by free_cashflow/free_cashflow_yield
    (never converted to a DataFrame by convert_cell because it doesn't
    start with "[" — parsed here to a sorted pd.Series instead), and
    scalar passthrough.
    """
    if isinstance(df, pd.DataFrame):
        if "date" in df.columns:
            out = df.copy()
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            return out.loc[out["date"] <= end]
        idx = pd.to_datetime(df.index, errors="coerce")
        return df.loc[idx <= end]

    if isinstance(df, str) and df.strip().startswith("{"):
        try:
            parsed = json.loads(df)
            ser = pd.Series(parsed, dtype=float).dropna()
        except (json.JSONDecodeError, ValueError, TypeError):
            return pd.Series(dtype=float)
        ser.index = pd.to_datetime(ser.index, errors="coerce")
        ser = ser[ser.index.notna()]
        return ser[ser.index <= end].sort_index()

    return df


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


# Files that live in data/ but are not per-company snapshots.
_NON_SNAPSHOT_FILES = {"fx_sek.csv"}


def get_hist_data(data_dir="data"):
    frames = []
    for csv in Path(data_dir).glob("*.csv"):
        # Skip the fiscal-year panel outputs (panel_fundamentals.csv /
        # panel_scores.csv). They are written into data/ by the new panel
        # pipeline but are NOT per-company snapshots — they already carry a
        # "company" column and no snapshot-date suffix, so treating them as
        # snapshots would crash this reader. No real snapshot is ever named
        # panel_*, so this preserves behavior exactly for every legitimate
        # input.
        # The FX rate cache (fx_sek.csv) is skipped for the same reason: it is
        # a date-indexed rate table, and reading it as a snapshot crashes
        # parse_ohlc_series.
        if csv.name.startswith("panel_") or csv.name in _NON_SNAPSHOT_FILES:
            continue
        key = csv.stem.split("_")[0]
        # filename is "<key>_<YYYY-MM-DD>.csv" — last underscore-separated
        # token is the snapshot date, used below to keep the most recent
        # snapshot per company instead of an arbitrary filesystem-order pick.
        snap_date = csv.stem.rsplit("_", 1)[-1]
        tmp = pd.read_csv(csv)
        tmp.insert(0, "company", key)
        tmp.insert(1, "_snapshot_date", snap_date)
        frames.append(tmp)

    df = pd.concat(frames, ignore_index=True).set_index("company", drop=False)
    if "asof" in df.columns:
        df = df.drop(columns=["asof"])

    for col in df.columns:
        if col == "ohlc":
            df[col] = df[col].apply(parse_ohlc_series)
        elif col != "_snapshot_date":
            df[col] = df[col].apply(lambda c: convert_cell(c, col))
            if col == "sector":
                df[col] = df[col].apply(extract_sector)

    # Keep the most recent snapshot per company (by filename date), not
    # whichever file pd.concat/glob happened to see first.
    df["_snapshot_date"] = pd.to_datetime(df["_snapshot_date"], errors="coerce")
    df = df.sort_values("_snapshot_date", ascending=False)
    return df.groupby(level=0).first().drop(columns=["_snapshot_date"])  # 1 row per company


# ------------------------------------------------------------------ value helpers
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


def _build_ticker_dicts(asof):
    """Build fake ticker_analysis and ticker_info dicts from CSV data
    so we can reuse the same financial_metrics functions as the live flow.

    Consumes `asof` — data already cut to <= the as-of date by slice_df_upto
    in the caller (analyzer/panel.py), so every field here is as-of that date
    with no look-ahead into or past the window being predicted.
    """
    revenue_year = _df_to_dict_list(asof.get("revenue_year"))
    revenue_quarter = _df_to_dict_list(asof.get("revenue_quarter"))
    eps_quarter = _df_to_dict_list(asof.get("eps_quarter"))
    profit_margin = _df_to_dict_list(asof.get("profit_margin"))
    profit_per_share = _df_to_dict_list(asof.get("profit_per_share"))
    roe_series = _df_to_dict_list(asof.get("roe"))
    de_series = _df_to_dict_list(asof.get("de_ratio"))
    net_profit = _df_to_dict_list(asof.get("net_profit"))
    total_assets = _df_to_dict_list(asof.get("total_assets"))
    total_liab = _df_to_dict_list(asof.get("total_liabilities"))
    equity_ps = _df_to_dict_list(asof.get("equity_per_share"))
    ev_ebit = _df_to_dict_list(asof.get("ev_ebit"))
    div_ps = _df_to_dict_list(asof.get("dividend_per_share"))

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
            "earningsPerShare": eps_quarter,
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
    pe_list = _df_to_dict_list(asof.get("pe"))
    if pe_list:
        latest_pe = pe_list[-1]["value"]

    # Dividend yield: scalar in CSV (not time-series)
    div_yield_raw = asof.get("dividend_yield")
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
