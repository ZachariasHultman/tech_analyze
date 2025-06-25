# ----------------------------------------------------------------------
#  Historical metrics generator
#  – Produces one row per <company , window-label>
# ----------------------------------------------------------------------

from pathlib import Path
import json
from datetime import datetime
import numpy as np
import pandas as pd

from metrics import extract_sector, sector_thresholds_berkshire
from helper import calculate_slope  # your util
from financial_metrics import (  # your util
    calc_sma200_metrics,
    calculate_closing_CAGR,
    calculate_PEG,
    calculate_CAGR_helper,
)

# ----------------------------------------------------------------------
# Metric ↔ datapoint map
# ----------------------------------------------------------------------
METRIC_TO_DATAPOINT = {
    "roe status": "roe",
    "de status": "de_ratio",
    "profit margin status": "profit_margin",
    "profit margin trend status": "profit_margin",
    "net debt - ebitda status": "netDebtEbitdaRatio",
    "nav discount status": "nav_discount",
    "nav discount trend status": "nav_discount",
    "calculated nav discount status": "calculated_nav_discount",
    "fcf status": "free_cashflow",
    "fcfy status": "free_cashflow_yield",
    "revenue trend year status": "revenue_year",
    "revenue trend quarter status": "revenue_quarter",
    "profit per share trend status": "profit_per_share",
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
def calculate_metrics_given_hist() -> None:
    df = get_hist_data()
    metrics = list(sector_thresholds_berkshire.keys())
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

        # ----- iterate windows (1Y / 3Y / 5Y TOTAL + YoY-k) ------------
        for span in (1, 3, 5):
            for label, start_d, end_d, yrs_span in make_windows(max_d, span):

                filtered = {
                    k: slice_df_between(row[k], start_d, end_d) for k in pre_metrics
                }
                ohlc_win = slice_df_between(ohlc_df, start_d, end_d)

                sma_val, _, sma_slope = calc_sma200_metrics(ohlc_win)
                sma_status = (
                    (ohlc_win["close"].iloc[-1] - sma_val) / sma_val
                    if sma_val is not None and not ohlc_win.empty
                    else None
                )

                # PE & PEG
                pe_ser = _series_from_df(filtered.get("pe"))
                pe_val = float(pe_ser.iloc[-1]) if not pe_ser.empty else None
                peg_val = calculate_PEG(
                    pe_ser.dropna().tolist()[-3:],
                    calculate_closing_CAGR(
                        None,
                        company,
                        use_hist=True,
                        hist_row=pd.Series({"ohlc": ohlc_win[["close"]]}),
                        years_tuple=(3, 2, 1),
                    ),
                )

                # window-aware price CAGR
                price_cagr = price_cagr_window(
                    ohlc_df["close"], start_d, end_d, yrs_span
                )
                if isinstance(price_cagr, np.floating):
                    price_cagr = float(price_cagr)

                entry = {"company": company, "sector": sector, "timespan": label}

                for m in metrics:
                    if "trend" in m:
                        col = METRIC_TO_DATAPOINT[m]
                        entry[m] = (
                            _trend_metric_yoy(row[col], start_d)
                            if yrs_span == 1 and "_YoY-" in label
                            else _trend_metric(filtered[col])
                        )
                    elif m == "sma200 status":
                        entry[m] = sma_status
                    elif m == "sma200 slope status":
                        entry[m] = sma_slope
                    elif m == "peg status":
                        entry[m] = peg_val
                    elif m == "cagr-pe compare status":
                        entry[m] = [price_cagr, pe_val]
                    else:
                        col = METRIC_TO_DATAPOINT.get(m)
                        entry[m] = (
                            _aggregate_for_timespan(m, filtered.get(col))
                            if col
                            else None
                        )

                results.append(entry)

    pd.DataFrame(results).set_index(["company", "timespan"]).to_csv(
        "metrics_by_timespan.csv"
    )


if __name__ == "__main__":
    calculate_metrics_given_hist()
