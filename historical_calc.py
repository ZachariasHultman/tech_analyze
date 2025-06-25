# ----------------------------------------------------------------------
# Semi-annual price metrics – starter implementation
# ----------------------------------------------------------------------

# TODO: NAV discount could be saved longer than 300 days, so it can be used for 3Y/5Y windows. Or not...


from pathlib import Path
import json
from datetime import datetime
import pandas as pd
from metrics import get_metrics_threshold, extract_sector, sector_thresholds_berkshire
import numpy as np
from helper import calculate_slope
from financial_metrics import (
    calc_sma200_metrics,
    calculate_closing_CAGR,
    calculate_PEG,
    calculate_CAGR_helper,
)


# Map normalized metric names to DataFrame column names
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


def _aggregate_for_timespan(metric_name: str, obj):
    """
    Collapse a DataFrame/Series/scalar to a single number **within
    the current timespan** (i.e. using *filtered_row*).

    ▸ “… trend …”  → pct-change between first & last value
    ▸ everything else → median of the values in the span
    """
    if obj is None or (isinstance(obj, float) and np.isnan(obj)):
        return np.nan

    # ── normalise input to a numeric Series ──────────────────
    if isinstance(obj, pd.DataFrame):
        if obj.empty or "value" not in obj.columns:
            return np.nan
        ser = obj.sort_values("date")["value"].dropna().astype(float)
    elif isinstance(obj, pd.Series):
        ser = obj.dropna().astype(float)
    else:  # scalar
        return float(obj)

    if ser.empty:
        return np.nan

    # ── choose aggregation strategy ──────────────────────────
    if "trend" in metric_name:
        first, last = ser.iloc[0], ser.iloc[-1]
        return np.nan if first == 0 else (last - first) / abs(first)

    # “status” (or anything else) → central tendency
    return ser.median()


def slice_df_between(obj, start, end):
    """
    Keep rows whose 'date' column lies in [start , end] (inclusive).
    Returns the object unchanged if it isn't a date-framed DataFrame.
    """
    if not isinstance(obj, pd.DataFrame) or "date" not in obj.columns:
        return obj

    df = obj.copy()

    # 1) ensure proper datetime dtype (strings → NaT)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 2) build the boolean mask
    mask = (df["date"] >= start) & (df["date"] <= end)

    return df.loc[mask]


def make_windows(max_date, years):
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
    windows = []

    # TOTAL             -------------------------------------------------
    start_total = max_date - pd.DateOffset(years=years)
    windows.append((f"{years}Y_TOTAL", start_total, max_date, years))

    # YEAR-OVER-YEAR    -------------------------------------------------
    for k in range(1, years + 1):
        if years == 1 and k == 1:
            continue  # drop the redundant 1Y_YoY-1 window
        end = max_date - pd.DateOffset(years=k - 1)
        start = end - pd.DateOffset(years=1)
        windows.append((f"{years}Y_YoY-{k}", start, end, 1))  # 1-year span

    return windows


def _series_from_df(obj: pd.DataFrame | pd.Series | float | int):
    """
    Return a clean numeric Series sorted by time (oldest → newest).

    • When 'obj' is a DataFrame it must contain 'date' and 'value' columns.
      The result's index will be that 'date' column, properly coerced to
      datetime64 so date-based comparisons work.
    """
    if obj is None:
        return pd.Series(dtype=float)

    # -- DataFrame -------------------------------------------------------
    if isinstance(obj, pd.DataFrame):
        if obj.empty or "value" not in obj.columns:
            return pd.Series(dtype=float)

        df = obj.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])  # drop NaT rows
        ser = (
            df.sort_values("date")
            .set_index("date")["value"]  # ← index = date
            .astype(float)
        )
        return ser

    # -- already a Series -----------------------------------------------
    if isinstance(obj, pd.Series):
        return obj.dropna().astype(float).sort_index()

    # -- scalar ----------------------------------------------------------
    return pd.Series([obj], dtype=float)


def _trend_metric(obj):
    ser = _series_from_df(obj)
    if len(ser) < 2:
        return None
    first, last = ser.iloc[0], ser.iloc[-1]
    return (last - first) / abs(first)  # fraction


# ------------------------------------------------------------------
# YoY change helper – for 1-year windows
# ------------------------------------------------------------------
def _trend_metric_yoy(obj, window_start):
    """
    Return (last - prev) / |prev|
       • 'last' = last value ON/AFTER window_start
       • 'prev' = last value BEFORE window_start (≥ 1y earlier)

    If prev is 0 or missing → None.
    """
    full_ser = _series_from_df(obj)
    if full_ser.empty:
        return None

    recent = full_ser[full_ser.index >= window_start]
    if recent.empty:
        return None  # nothing in window

    last_val = recent.iloc[-1]

    prev = full_ser[full_ser.index < window_start]
    if prev.empty or prev.iloc[-1] == 0:
        return None

    prev_val = prev.iloc[-1]
    return (last_val - prev_val) / abs(prev_val)


def print_df_info(df):
    print(f"DataFrame shape: {df.shape}")
    for idx, row in df.iterrows():
        for col in df.columns:
            cell = row[col]
            if hasattr(cell, "shape"):
                print(f"Cell[{idx}, {col}] shape: {cell.shape}")
            else:
                print(f"type of Cell[{idx}, {col}]: {type(cell)}, value: {cell}")
            if col == "pe":
                print(f"PE value: {cell}, type: {type(cell)}")

        return


def parse_ohlc_series(ohlc_str: str) -> pd.DataFrame:
    """
    Convert the JSON-encoded OHLC list in the csv to a tidy DataFrame.
    """
    ohlc = json.loads(ohlc_str)
    df = pd.DataFrame(ohlc)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df[["close"]]  # keep only what we need for speed


def convert_cell(cell, col):
    """
    • str → try JSON-decode → recurse
    • list/dict → DataFrame
    • numeric str → float
    • blank / 'nan' → NaN
    """
    # --- 1. decode JSON-like strings ---------------------------------
    if isinstance(cell, str):
        s = cell.strip()
        if s.lower() in ("", "nan", "null"):
            return np.nan
        if (s.startswith("[") and s.endswith("]")) or (
            s.startswith("{") and s.endswith("}")
        ):
            try:
                parsed = json.loads(s)
                return convert_cell(parsed, col)  # recurse
            except json.JSONDecodeError:
                pass
        # plain numeric string
        try:
            return float(s)
        except ValueError:
            return s

    # --- 2. lists / dicts --------------------------------------------
    if isinstance(cell, list):
        return pd.DataFrame(cell)
    if isinstance(cell, dict):
        return pd.DataFrame([cell])

    # --- 3. already numeric / NaN ------------------------------------
    return cell


def get_hist_data():
    data_dir = Path("data")
    frames = []

    for csv_file in data_dir.glob("*.csv"):
        company_key = csv_file.stem.split("_")[0]  # "ericsson", …
        tmp = pd.read_csv(csv_file)
        tmp.insert(0, "company", company_key)  # new column → first col
        frames.append(tmp)

    df = pd.concat(frames, ignore_index=True)

    # optional: drop columns you don’t want *before* setting the index
    if "asof" in df.columns:
        df = df.drop(columns=["asof"])

    # now make the file‐name key the one and only index
    df = df.set_index("company", drop=False)  # keep the column too, if handy

    for col in df.columns:
        if col == "ohlc":
            df[col] = df[col].apply(parse_ohlc_series)
        else:
            df[col] = df[col].apply(lambda cell: convert_cell(cell, col))
            if col == "sector":
                df[col] = df[col].apply(extract_sector)

    return df


def filter_df_by_timespan(df, days):
    if isinstance(df, pd.DataFrame) and "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        end_date = df["date"].max()
        start_date = end_date - pd.Timedelta(days=days)
        return df[df["date"] >= start_date]
    return df


def calculate_metrics_given_hist() -> int:
    df = get_hist_data()  # <- many rows per company
    df = df.groupby(
        level=0
    ).first()  # group by index (= company key)  # keep the first (= latest) row  # now 1-row-per-company, no dups

    metrics = sector_thresholds_berkshire.keys()
    timespans = {"1Y": 365, "3Y": 365 * 3, "5Y": 365 * 5}

    # columns that are **not** raw datapoints
    exclude_cols = {"name", "sector", "ohlc", "market_cap", "currency"}
    PRECOMPUTED_METRICS = [c for c in df.columns if c not in exclude_cols]

    results = []

    for company_name, row in df.iterrows():
        sector = row.get("sector", "Unknown")

        # ──────────────────────────────────────────────────────────────
        # 1. melt any “wide” one-row frames inside this row (once)
        # ──────────────────────────────────────────────────────────────
        for col in df.columns:
            if col == "sector":
                continue
            cell = row[col]
            if (
                isinstance(cell, pd.DataFrame)
                and cell.shape[0] == 1
                and cell.shape[1] > 1
            ):
                melted = cell.melt(var_name="date", value_name="value")
                try:
                    melted["date"] = pd.to_datetime(melted["date"])
                except Exception:
                    pass
                row[col] = melted
        ref_df = row["ohlc"] if isinstance(row["ohlc"], pd.DataFrame) else None
        max_date = (
            ref_df.reset_index()["date"].max()
            if ref_df is not None
            else pd.Timestamp.today()
        )
        # ──────────────────────────────────────────────────────────────
        # 2. iterate over the three time-windows
        # ──────────────────────────────────────────────────────────────
        for span_years in (1, 3, 5):
            for label, start_d, end_d, years_in_span in make_windows(
                max_date, span_years
            ):

                filtered_row = row.copy()
                for key in PRECOMPUTED_METRICS:
                    filtered_row[key] = slice_df_between(row.get(key), start_d, end_d)

                result_entry = {
                    "company": company_name,
                    "sector": sector,
                    "timespan": label,
                }

                # ---------- SMA-200 --------------------------------------
                ohlc_df = filtered_row.get("ohlc")
                if (
                    ohlc_df is not None
                    and not ohlc_df.empty
                    and "date" not in ohlc_df.columns
                ):
                    ohlc_df = ohlc_df.reset_index().rename(columns={"index": "date"})

                sma200_val, _, sma_slope = calc_sma200_metrics(ohlc_df)
                if sma200_val is not None and ohlc_df is not None and not ohlc_df.empty:
                    latest_close = float(ohlc_df["close"].iloc[-1])
                    sma_status = (latest_close - sma200_val) / sma200_val
                else:
                    sma_status = sma_slope = None

                # ---------- PE & PEG -------------------------------------
                pe_series = _series_from_df(filtered_row.get("pe"))
                pe_val = float(pe_series.iloc[-1]) if not pe_series.empty else None

                pe_list = pe_series.dropna().astype(float).tolist()[-3:]
                cagr_list = calculate_closing_CAGR(
                    avanza=None,
                    ticker=company_name,
                    use_hist=True,
                    hist_row=row,
                    years_tuple=(3, 2, 1),
                )
                peg_classic = calculate_PEG(pe_list, cagr_list)

                # ---------- price-CAGR for this span ---------------------
                if ohlc_df is not None and not ohlc_df.empty and years_in_span > 0:
                    price_cagr_val = calculate_CAGR_helper(
                        ohlc_df[["date", "close"]]
                        .rename(columns={"date": "Date"})
                        .set_index("Date")
                        .astype(float)
                        .sort_index(),
                        years_in_span,
                    )
                else:
                    price_cagr_val = None

                # ---------- fill every metric ----------------------------
                for metric in metrics:
                    if "trend" in metric:
                        # map to the raw datapoint column
                        dp_col = METRIC_TO_DATAPOINT.get(metric)
                        if years_in_span == 1 and "_YoY-" in label:
                            # YoY window → use percentage change versus previous year
                            value = _trend_metric_yoy(row.get(dp_col), start_d)
                        else:
                            # TOTAL or 3Y/5Y slope as before
                            value = _trend_metric(filtered_row.get(dp_col))
                    elif metric == "sma200 status":
                        value = sma_status

                    elif metric == "sma200 slope status":
                        value = sma_slope

                    elif metric == "peg status":
                        value = peg_classic

                    elif metric == "cagr-pe compare status":
                        value = [price_cagr_val, pe_val]

                    else:
                        dp_col = METRIC_TO_DATAPOINT.get(metric)
                        value = (
                            _aggregate_for_timespan(metric, filtered_row.get(dp_col))
                            if dp_col
                            else None
                        )

                    result_entry[metric] = value

                results.append(result_entry)

    # save
    results_df = pd.DataFrame(results)
    results_df.set_index(["company", "timespan"], inplace=True)
    results_df.to_csv("metrics_by_timespan.csv")

    return 0
