import pandas as pd
from currency_converter import CurrencyConverter
import requests
from avanza.avanza import Resolution, TimePeriod
from analyzer.helper import *
from analyzer.get_NAV_data import get_nav_data
from datetime import datetime
from analyzer.metrics import ticker_reporting_currency_map
import numpy as np


# financial_metrics.py
from typing import Any, Dict, List, Optional, Tuple


# ---------- small utils ----------
YEARS = 3


def _as_vals(seq: Optional[List[dict]]) -> List[Optional[float]]:
    """Extract .value from a list of {value,..} dicts, keep order, allow None."""
    if not isinstance(seq, list):
        return []
    out = []
    for x in seq:
        try:
            out.append(float(x.get("value")))
        except Exception:
            out.append(None)
    return out


def _clean(vals: List[Optional[float]]) -> List[float]:
    """Drop None; keep order."""
    return [v for v in vals if v is not None]


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    try:
        if a is None or b in (None, 0):
            return None
        return float(a) / float(b)
    except Exception:
        return None


def _cagr(first: Optional[float], last: Optional[float], years: int) -> Optional[float]:
    """CAGR = (last/first)^(1/years) - 1.

    When both values are positive, uses standard CAGR formula.
    When first is negative and last is positive (turnaround), returns an
    annualized absolute growth rate so these stocks aren't silently dropped.
    """
    try:
        if first is None or last is None or years <= 0:
            return None
        if first == 0:
            return None
        # Standard case: both positive
        if first > 0 and last > 0:
            return (last / first) ** (1.0 / years) - 1.0
        # Turnaround: negative → positive (strong improvement)
        if first < 0 and last > 0:
            # Use absolute change annualized: treats as growth from |first| to last
            return ((last - first) / abs(first)) ** (1.0 / years) - 1.0
        # Deterioration: positive → negative
        if first > 0 and last <= 0:
            return -1.0  # total loss
        # Both negative: still deteriorating, or improving toward zero
        if first < 0 and last < 0:
            # Improving if loss is shrinking (last closer to 0)
            if abs(last) < abs(first):
                return (abs(first) / abs(last)) ** (1.0 / years) - 1.0
            return -(abs(last) / abs(first)) ** (1.0 / years) + 1.0
        return None
    except Exception:
        return None


def _rolling_yoy_from_quarterly(vals):
    out = [None] * len(vals)
    for i in range(4, len(vals)):
        ratio = _safe_div(vals[i], vals[i - 4])
        out[i] = (ratio - 1.0) if ratio is not None else None
    return out


def _hit_rate(last_n: List[Optional[float]]) -> Optional[float]:
    """Share of positives among last_n (ignoring None)."""
    valid = [v for v in last_n if v is not None]
    if not valid:
        return None
    return sum(1 for v in valid if v > 0) / float(len(valid))


def _avg(vals: List[Optional[float]]) -> Optional[float]:
    xs = _clean(vals)
    if not xs:
        return None
    return sum(xs) / len(xs)


# ---------- 1) Revenue y CAGR ----------


def calculate_revenue_y_cagr(
    ticker_analysis: Dict[str, Any],
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Uses companyFinancialsByYear.sales (list of dicts with 'value') to compute y CAGR.
    Returns (cagr, meta) where cagr is a float or None.
    """
    sales = _as_vals(ticker_analysis.get("companyFinancialsByYear", {}).get("sales"))
    # Need at least years+1 points to span years full years (t-years to t)
    if len(_clean(sales)) < YEARS + 1:
        return None, {"reason": "not_enough_data", "points": len(_clean(sales))}
    first, last = sales[-(YEARS + 1)], sales[-1]
    cagr = _cagr(first, last, YEARS)
    return cagr, {"first": first, "last": last, "years": YEARS}


# ---------- 2) EPS y CAGR ----------


def calculate_eps_y_cagr(
    ticker_analysis: Dict[str, Any],
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Uses companyKeyRatiosByYear.earningsPerShare to compute y CAGR.
    """
    eps = _as_vals(
        ticker_analysis.get("companyKeyRatiosByYear", {}).get("earningsPerShare")
    )
    if len(_clean(eps)) < YEARS + 1:
        return None, {"reason": "not_enough_data", "points": len(_clean(eps))}
    first, last = eps[-(YEARS + 1)], eps[-1]
    cagr = _cagr(first, last, YEARS)
    return cagr, {"first": first, "last": last, "years": YEARS}


# ---------- 3) Revenue YoY hit-rate (last 12 quarters) ----------


def calculate_revenue_yoy_hit_rate(
    ticker_analysis: Dict[str, Any], lookback_quarters: int = 12
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Uses companyFinancialsByQuarter.sales to compute YoY per quarter (4Q lag)
    and then the fraction of positive YoY in the last `lookback_quarters`.
    """
    q_sales = _as_vals(
        ticker_analysis.get("companyFinancialsByQuarter", {}).get("sales")
    )
    if len(_clean(q_sales)) < 8:  # need at least 8 quarters to form some YoY
        return None, {"reason": "not_enough_quarters", "points": len(_clean(q_sales))}
    yoy = _rolling_yoy_from_quarterly(q_sales)
    last_window = yoy[-lookback_quarters:] if lookback_quarters > 0 else yoy
    hit = _hit_rate(last_window)
    return hit, {
        "window": lookback_quarters,
        "positives": (
            None
            if hit is None
            else hit * len([v for v in last_window if v is not None])
        ),
    }


# ---------- 4) EPS YoY hit-rate (last 12 quarters) ----------


def calculate_eps_yoy_hit_rate(
    ticker_analysis: Dict[str, Any], lookback_quarters: int = 12
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Uses companyKeyRatiosByQuarterQuarter.earningsPerShare for quarterly EPS,
    computes YoY with 4Q lag, then fraction >0 in last `lookback_quarters`.
    """
    q_eps = _as_vals(
        ticker_analysis.get("companyKeyRatiosByQuarterQuarter", {}).get(
            "earningsPerShare"
        )
    )
    if len(_clean(q_eps)) < 8:
        return None, {"reason": "not_enough_quarters", "points": len(_clean(q_eps))}
    yoy = _rolling_yoy_from_quarterly(q_eps)
    last_window = yoy[-lookback_quarters:] if lookback_quarters > 0 else yoy
    hit = _hit_rate(last_window)
    return hit, {
        "window": lookback_quarters,
        "positives": (
            None
            if hit is None
            else hit * len([v for v in last_window if v is not None])
        ),
    }


# ---------- 5) Net margin vs y average ----------


def calculate_net_margin_vs_avg(
    ticker_info: Dict[str, Any], ticker_analysis: Dict[str, Any], years: int = 3
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    latest / y average of profit margin. Latest from ticker_info.keyIndicators.netMargin.
    y avg from companyFinancialsByYear.profitMargin (last `years` points).
    """
    latest = ticker_info.get("keyIndicators", {}).get("netMargin")
    yr_margins = _as_vals(
        ticker_analysis.get("companyFinancialsByYear", {}).get("profitMargin")
    )
    avg_y = _avg(yr_margins[-years:]) if yr_margins else None
    ratio = _safe_div(latest, avg_y)
    return ratio, {"latest": latest, "avg_years": years, "avg": avg_y}


# ---------- 6) ROE vs y average ----------


def calculate_roe_vs_avg(
    ticker_info: Dict[str, Any], ticker_analysis: Dict[str, Any], years: int = 3
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    latest / y average of ROE. Latest from ticker_info.keyIndicators.returnOnEquity.
    y avg from companyKeyRatiosByYear.returnOnEquityRatio.
    """
    latest = ticker_info.get("keyIndicators", {}).get("returnOnEquity")
    yr_roe = _as_vals(
        ticker_analysis.get("companyKeyRatiosByYear", {}).get("returnOnEquityRatio")
    )
    avg_y = _avg(yr_roe[-years:]) if yr_roe else None
    ratio = _safe_div(latest, avg_y)
    return ratio, {"latest": latest, "avg_years": years, "avg": avg_y}


def get_ohlc_dataframe(
    *,
    ticker_id: str,
    avanza=None,
    use_hist: bool = False,
    hist_row: pd.Series | None = None,
    years_back: int = 5,
) -> pd.DataFrame | None:
    """
    Return a tidy OHLC DataFrame with columns:
        [date, open, high, low, close, totalVolumeTraded]

    Parameters
    ----------
    ticker_id   : str
        Avanza order-book id **or** the same key you use in your historical store.
    avanza      : Avanza client (needed only when use_hist is False)
    use_hist    : bool (default False)
        * False → live fetch via Avanza
        * True  → use `hist_row["ohlc"]` that you already loaded with get_hist_data()
    hist_row    : pd.Series
        The row from your historical DataFrame containing the `"ohlc"` field.
        Required when use_hist is True.
    years_back  : int
        How many years of history you want (default 5).
    """
    if use_hist:
        if hist_row is None or "ohlc" not in hist_row or hist_row["ohlc"] is None:
            return None
        ohlc = hist_row["ohlc"].copy()
        # keep last `years_back` of data
        cutoff = ohlc.index.max() - pd.Timedelta(days=365 * years_back)
        ohlc = ohlc[ohlc.index >= cutoff]
        ohlc = ohlc.rename(columns={"close": "close"})  # already correct
        ohlc = ohlc.reset_index().rename(columns={"date": "date"})
        ohlc["totalVolumeTraded"] = np.nan  # not available in hist store
        return ohlc[["date", "open", "high", "low", "close", "totalVolumeTraded"]]

    # ------------------------------------------------------------------
    # live fetch via Avanza REST
    if avanza is None:
        raise ValueError("`avanza` client must be supplied when use_hist=False")

    try:
        raw = avanza.get_chart_data(
            order_book_id=ticker_id,
            period=TimePeriod.FIVE_YEARS if years_back >= 5 else TimePeriod.ONE_YEAR,
            resolution=Resolution.DAY,
        )["ohlc"]
    except requests.exceptions.HTTPError:
        return None

    df = pd.DataFrame(raw)
    df["date"] = df["timestamp"].apply(
        lambda ts: datetime.utcfromtimestamp(ts / 1000).date()
    )
    return df[["date", "open", "high", "low", "close", "totalVolumeTraded"]]


def calc_sma200_metrics(
    df_hist: pd.DataFrame,
) -> tuple[float | None, float | None, float | None]:
    """
    Parameters
    ----------
    df_hist : tidy OHLC dataframe (see get_ohlc_dataframe)

    Returns
    -------
    sma200                : float | None  – latest SMA-200 value
    last_week_avg_close   : float | None  – mean close of last 7 trading days
    sma200_slope          : float | None  – slope of entire SMA-200 series
    """
    if df_hist is None or df_hist.empty:
        return None, None, None

    close_ser = df_hist["close"].astype(float).reset_index(drop=True)

    # --- 200-day simple moving average ---------------------------------
    sma200_ser = close_ser.rolling(window=200, min_periods=200).mean().dropna()
    if sma200_ser.empty:
        return None, None, None

    sma200_latest = float(sma200_ser.iloc[-1])
    last_week_avg = float(close_ser.tail(7).mean())

    # slope helper (your existing function)
    sma200_slope = calculate_slope(sma200_ser.tolist())

    return sma200_latest, last_week_avg, float(sma200_slope)


# ----------------------------------------------------------------------
# Convenience wrapper (keeps the old signature working)
# ----------------------------------------------------------------------
def calculate_sma200(avanza, ticker_id, *, use_hist: bool = False, hist_row=None):
    """
    Back-compatible wrapper so existing callers need not change.

    Returns (sma200, last_week_avg_close, slope, df_hist)
    """
    df_hist = get_ohlc_dataframe(
        ticker_id=ticker_id,
        avanza=avanza,
        use_hist=use_hist,
        hist_row=hist_row,
        years_back=5,
    )
    sma200, last_week_avg, slope = calc_sma200_metrics(df_hist)
    return sma200, last_week_avg, slope, df_hist


# -------------------------------------------------------------------
def calculate_profit_per_share_trend(ticker_analysis, ticker_id=None):
    raw = [
        {"date": e["date"], "value": e["value"]}
        for e in ticker_analysis["companyKeyRatiosByYear"]["earningsPerShare"]
        if e.get("reportType") == "FULL_YEAR" and "date" in e
    ]
    if len(raw) < 2:
        return None, None

    values = [d["value"] for d in raw][-5:]  # last five EPS numbers
    slope = calculate_slope(values, ticker_id)  # expects numeric list
    return float(slope), raw  # (slope, full record list)


# -------------------------------------------------------------------


def calculate_profit_margin_trend(ticker_analysis, ticker_id=None):
    ticker_profit_margin = [
        entry["value"]
        for entry in ticker_analysis["companyFinancialsByYear"]["profitMargin"]
        if "reportType" in entry and entry["reportType"] == "FULL_YEAR"
    ][-5:]
    # ticker_profit_margin = z_score(ticker_profit_margin)
    if len(ticker_profit_margin) > 1:
        slope = calculate_slope(ticker_profit_margin, ticker_id)
        return float(slope)
    else:
        return None


def calculate_revenue_trend(ticker_analysis, ticker_id=None):
    # --- build dict-lists --------------------------------------------------
    yr = [
        {"date": e["date"], "value": e["value"]}
        for e in ticker_analysis["companyFinancialsByYear"]["sales"]
        if e.get("reportType") == "FULL_YEAR" and "date" in e
    ]
    qtr = [
        {"date": e["date"], "value": e["value"]}
        for e in ticker_analysis["companyFinancialsByQuarter"]["sales"]
        if "value" in e and "date" in e
    ]

    if len(qtr) > 1 and len(yr) > 1:
        # last five annual values
        y_vals = [d["value"] for d in yr][-5:]
        # all quarterly values
        q_vals = [d["value"] for d in qtr]

        slope_year = float(calculate_slope(y_vals))
        slope_quarter = float(calculate_slope(q_vals))

        return slope_year, slope_quarter, yr, qtr

    return None, None, None, None


def calculate_PE(ticker_analysis):
    pe = [
        {"date": entry["date"], "value": entry["value"]}
        for entry in ticker_analysis["stockKeyRatiosByYear"]["priceEarningsRatio"]
        if "reportType" in entry
        and entry["reportType"] == "FULL_YEAR"
        and "date" in entry
    ]
    latest_vals = [d["value"] for d in pe][-5:]
    if len(pe) >= 1:
        return latest_vals, pe
    else:
        return None, None


def calculate_CAGR_helper(df, years: int):
    """
    CAGR over `years` using the last trading day ON OR BEFORE the target start date
    and the most recent close.
    """
    df = df.sort_index()
    if df.empty or len(df) < 2:
        return None

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
        except Exception:
            return None

    # Cull bad/NaN closes
    if "close" not in df.columns:
        return None
    df = df[["close"]].dropna()
    if df.empty:
        return None

    end_date = df.index[-1]
    start_date = end_date - pd.DateOffset(years=years)

    # Pick last trading day ON/BEFORE the target date (avoids future-shift bias)
    # asof returns the latest label <= given date, or NaT if none
    start_ix = df.index.asof(start_date)
    if pd.isna(start_ix):  # no data that far back
        return None

    start_price = df.loc[start_ix, "close"]
    end_price = df["close"].iloc[-1]

    # Safety—avoid nonsense
    if pd.isna(start_price) or pd.isna(end_price) or start_price <= 0 or end_price <= 0:
        return None

    cagr = (end_price / start_price) ** (1 / years) - 1
    return float(cagr)


def calculate_closing_CAGR(
    avanza,
    ticker,
    *,
    use_hist: bool = False,
    hist_row: pd.Series | None = None,
    years_tuple: tuple[int, ...] | None = None,
) -> list[float] | None:
    """
    Returns [CAGR_y1, CAGR_y2, ...] in the same order as years_tuple.
    If years_tuple is None, defaults to (YEARS, YEARS-1, ..., 1).
    Entries with insufficient history return None.
    """
    # Build default years tuple dynamically from global YEARS
    try:
        Y = int(YEARS)
    except Exception:
        Y = 3  # hard fallback if YEARS isn't defined
    if years_tuple is None:
        years_tuple = tuple(range(Y, 0, -1))  # e.g., 5,4,3,2,1

    # ── HISTORICAL MODE ────────────────────────────────────────────────
    if use_hist:
        if not isinstance(hist_row, pd.Series) or "ohlc" not in hist_row:
            return None
        df = hist_row["ohlc"]
        if df is None or getattr(df, "empty", True):
            return None
        df = df.rename(columns={"close": "close"}).copy()
        # force numeric & sorted
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"]).sort_index()

        vals = [calculate_CAGR_helper(df, y) for y in years_tuple]
        return vals if any(v is not None for v in vals) else None

    # ── LIVE MODE (via Avanza) ────────────────────────────────────────
    # Pick a long enough history window to cover max(years_tuple)
    max_years = max(years_tuple) if years_tuple else 5
    # Choose an Avanza period that comfortably covers the maximum span
    period = TimePeriod.FIVE_YEARS
    try:
        if max_years <= 5:
            period = TimePeriod.FIVE_YEARS
        elif max_years <= 10 and hasattr(TimePeriod, "TEN_YEARS"):
            period = TimePeriod.TEN_YEARS
        elif hasattr(TimePeriod, "FIFTEEN_YEARS"):
            period = TimePeriod.FIFTEEN_YEARS
    except Exception:
        # keep FIVE_YEARS if enum probing fails
        pass

    try:
        ticker_chart_data = avanza.get_chart_data(
            order_book_id=ticker,
            period=period,
            resolution=Resolution.DAY,
        )
    except requests.exceptions.HTTPError:
        return None

    ohlc = ticker_chart_data.get("ohlc") or []
    if not ohlc:
        return None

    dates = pd.to_datetime([e["timestamp"] for e in ohlc], unit="ms", errors="coerce")
    closes = pd.to_numeric([e.get("close") for e in ohlc], errors="coerce")
    data = pd.DataFrame({"close": closes}, index=dates).dropna().sort_index()

    vals = [calculate_CAGR_helper(data, y) for y in years_tuple]
    return vals if any(v is not None for v in vals) else None


def calculate_price_cagr_status(
    avanza,
    ticker,
    *,
    use_hist: bool = False,
    hist_row: pd.Series | None = None,
):
    """
    Returns a single CAGR over the global YEARS for price.
    """
    # one-value tuple with the global YEARS
    try:
        yspan = YEARS
    except NameError:
        yspan = 3  # fallback if YEARS not defined

    vals = calculate_closing_CAGR(
        avanza,
        ticker,
        use_hist=use_hist,
        hist_row=hist_row,
        years_tuple=(yspan,),
    )
    if not vals:
        return None
    return vals[0]


def sync_currency(from_currency: str, to_currency: str = "SEK"):
    c = CurrencyConverter()
    if not from_currency or not to_currency:
        return False, False, None, None
    exchange_rate = c.convert(1, from_currency, to_currency)
    # Keep API stable, but mark SEK path unused by caller after patch
    sek_rate = c.convert(1, from_currency, "SEK")
    currency_match = from_currency == to_currency
    convert_to_sek = False  # <— force off; we won't use SEK conversion in FCFY
    return currency_match, convert_to_sek, exchange_rate, sek_rate


def calculate_free_cashflow_yield(yahoo_ticker, stock_info, df_hist=None):
    try:
        # --- base inputs ---
        from_currency = ticker_reporting_currency_map.get(yahoo_ticker.ticker)
        ki = stock_info.get("keyIndicators", {}) if isinstance(stock_info, dict) else {}
        mc_obj = ki.get("marketCapital", {}) if isinstance(ki, dict) else {}
        to_currency = mc_obj.get("currency")
        market_cap = mc_obj.get("value")

        # guard market_cap
        if (
            market_cap is None
            or not isinstance(market_cap, (int, float))
            or market_cap <= 0
        ):
            return None, None, None, None

        # infer shares outstanding
        shares_outstanding = None
        if (
            df_hist is not None
            and isinstance(df_hist, pd.DataFrame)
            and not df_hist.empty
        ):
            try:
                last_close_price = float(df_hist["close"].iloc[-1])
                if last_close_price > 0:
                    shares_outstanding = market_cap / last_close_price
            except Exception:
                shares_outstanding = None

        # fallback: try yfinance info
        if shares_outstanding is None:
            try:
                so = getattr(yahoo_ticker, "info", {}).get("sharesOutstanding")
                if isinstance(so, (int, float)) and so > 0:
                    shares_outstanding = float(so)
            except Exception:
                shares_outstanding = None

        # if still missing, we can compute current FCFY via market_cap, but no hist yields
        # --- fetch cash flow ---
        cash_flow_df = getattr(yahoo_ticker, "cashflow", None)
        if (
            cash_flow_df is None
            or not isinstance(cash_flow_df, pd.DataFrame)
            or cash_flow_df.empty
        ):
            return None, None, None, None
        if "Free Cash Flow" not in cash_flow_df.index:
            print(" 'Free Cash Flow' not found for", yahoo_ticker.ticker)
            return None, None, None, None

        free_cash_flow_hist = cash_flow_df.loc["Free Cash Flow"].copy()

        # coerce to numeric
        free_cash_flow_hist = pd.to_numeric(free_cash_flow_hist, errors="coerce")

        # --- FX normalization ---
        # sync_currency may return None for rates; treat None as 1.0 to avoid float * NoneType
        currency_match, _, ex_rate, _ = sync_currency(
            from_currency=from_currency, to_currency=to_currency
        )
        ex_rate = 1.0 if ex_rate is None else float(ex_rate)

        # Convert FCF to the same currency as market_cap; do not convert to SEK here.
        if not currency_match:
            free_cash_flow_hist = free_cash_flow_hist * ex_rate

        # --- historical FCF yield (requires shares_outstanding & df_hist) ---
        fcf_yield_hist = {}
        if (
            shares_outstanding
            and df_hist is not None
            and isinstance(df_hist, pd.DataFrame)
            and not df_hist.empty
        ):
            try:
                df_hist = df_hist.copy()
                df_hist.index = pd.to_datetime(df_hist.index)
            except Exception:
                pass
            for dt, fcf in free_cash_flow_hist.items():
                if pd.isna(fcf):
                    continue
                try:
                    # align to the latest price <= dt
                    close_price = df_hist.loc[: pd.to_datetime(dt)]["close"].iloc[-1]
                    market_cap_hist = float(close_price) * float(shares_outstanding)
                    if market_cap_hist > 0:
                        fcf_yield_hist[pd.to_datetime(dt).strftime("%Y-%m-%d")] = (
                            float(fcf) / market_cap_hist
                        )
                except Exception:
                    continue

        # --- latest values ---
        latest_fcf = None
        try:
            latest_fcf = float(free_cash_flow_hist.dropna().iloc[0])
        except Exception:
            latest_fcf = None

        fcf_yield_now = (
            (latest_fcf / market_cap)
            if (latest_fcf is not None and market_cap)
            else None
        )

        # serialize FCF hist
        free_cf_hist_dict = {
            pd.to_datetime(dt).strftime("%Y-%m-%d"): (
                None if pd.isna(val) else float(val)
            )
            for dt, val in free_cash_flow_hist.items()
        }

        return (
            fcf_yield_now,  # scalar or None
            latest_fcf,  # scalar or None
            fcf_yield_hist,  # dict
            free_cf_hist_dict,  # dict
        )

    except (KeyError, IndexError, TypeError, ValueError) as e:
        print("calculate_free_cashflow_yield failed:", e)
        return None, None, None, None


# ------------------------------------------------------------------
# EV / EBIT  – latest value + tidy history
# ------------------------------------------------------------------
def extract_ev_ebit_ratio(avanza_data):
    """
    Parameters
    ----------
    avanza_data : dict
        Raw JSON block from Avanza's /stock endpoint.

    Returns
    -------
    latest_ev_ebit : float | None
        The single most-recent EV/EBIT value.
    ev_ebit_hist : list[dict] | None
        Full history, each item:
            {"date": "YYYY-MM-DD", "value": <float>}
        If nothing valid is found → (None, None)
    """
    raw = avanza_data.get("stockKeyRatiosByYear", {}).get("evEbitRatio", [])

    records = []
    for e in raw:
        if e.get("reportType") != "FULL_YEAR" or "date" not in e:
            continue
        try:
            date_iso = pd.to_datetime(e["date"]).strftime("%Y-%m-%d")
            records.append({"date": date_iso, "value": float(e["value"])})
        except Exception:  # malformed date or non-numeric value
            continue

    if not records:
        return None, None

    records.sort(key=lambda r: r["date"])  # chronological
    latest_ev_ebit = records[-1]["value"]

    return latest_ev_ebit, records


# ------------------------------------------------------------------
# Net-Debt / EBITDA  – tidy list + latest-five helper
# ------------------------------------------------------------------


def extract_netdebt_ebitda_ratio(avanza_data):
    """
    Parameters
    ----------
    avanza_data : dict
        Raw JSON from Avanza’s /company endpoint.

    Returns
    -------
    latest_val : float | None
        The single most-recent Net-Debt / EBITDA value.
    nd_ebitda_hist : list[dict] | None
        Full history, each item:
            {"date": "YYYY-MM-DD", "value": <float>}
        If no valid entries exist → (None, None)
    """
    raw = avanza_data.get("companyKeyRatiosByYear", {}).get("netDebtEbitdaRatio", [])

    records = []
    for e in raw:
        if e.get("reportType") != "FULL_YEAR" or "date" not in e:
            continue
        try:
            date_iso = pd.to_datetime(e["date"]).strftime("%Y-%m-%d")
            records.append({"date": date_iso, "value": float(e["value"])})
        except Exception:  # bad date or non-numeric value
            continue

    if not records:
        return None, None

    # sort chronologically to identify the latest
    records.sort(key=lambda r: r["date"])
    latest_val = records[-1]["value"]
    return latest_val, records


def calculate_NAV_discount(ticker_name):
    """
    Return:
        nav_discount_mean_30           – float
        calculated_nav_discount_mean_30– float
        nav_trend_slope                – float
        nav_discount_raw               – list[dict{date,value}]
        calc_nav_discount_raw          – list[dict{date,value}]
    """

    df_nav = get_nav_data(ticker_name)
    if df_nav is None or df_nav.empty:
        return None, None, None, None, None

    # --- keep last 300 records & ensure we have real datetimes ----------
    df_nav = df_nav.tail(300).copy()
    df_nav["DATUM"] = pd.to_datetime(df_nav["DATUM"], errors="coerce")
    df_nav = df_nav.dropna(subset=["DATUM"])

    # --- core columns ---------------------------------------------------
    nav = df_nav["SUBSTANSVÄRDE"].astype(float)
    nav_calc = df_nav["BERÄKNAT_SUBSTANSVÄRDE"].astype(float)
    price = df_nav["PRIS"].astype(float)

    # avoid division-by-zero → replace 0 with NaN
    nav_discount_series = (price - nav) / nav.replace(0, np.nan)
    calc_discount_series = (price - nav_calc) / nav_calc.replace(0, np.nan)

    # --- build raw [{"date": …, "value": …}, …] -------------------------
    nav_discount_raw = [
        {"date": dt.strftime("%Y-%m-%d"), "value": float(val)}
        for dt, val in zip(df_nav["DATUM"], nav_discount_series)
        if pd.notna(val)
    ]
    calc_nav_discount_raw = [
        {"date": dt.strftime("%Y-%m-%d"), "value": float(val)}
        for dt, val in zip(df_nav["DATUM"], calc_discount_series)
        if pd.notna(val)
    ]

    # need at least two points for a trend
    if len(nav_discount_raw) < 2:
        return None, None, None, None, None

    # --- 30-day means ---------------------------------------------------
    nav_mean_30 = float(nav_discount_series.tail(30).mean())
    calc_nav_mean_30 = float(calc_discount_series.tail(30).mean())

    # --- slope of the last 5 discount values (similar to EPS trend) -----
    nav_trend_slope = calculate_slope([d["value"] for d in nav_discount_raw[-5:]])

    return (
        nav_mean_30,
        calc_nav_mean_30,
        float(nav_trend_slope),
        nav_discount_raw,
        calc_nav_discount_raw,
    )


def calculate_gross_margin_stability(
    ticker_analysis: Dict[str, Any], years: int = 5
) -> Optional[float]:
    """
    Coefficient of variation (std/mean) of gross margin over `years` years.
    Lower = more stable = better for long-term investing.
    Returns the CV as a float, or None if insufficient data.
    Uses profitMargin from companyFinancialsByYear as a proxy when
    gross margin isn't directly available.
    """
    margins = _as_vals(
        ticker_analysis.get("companyFinancialsByYear", {}).get("profitMargin")
    )
    clean = _clean(margins[-years:]) if margins else []
    if len(clean) < 3:
        return None
    avg = sum(clean) / len(clean)
    if avg == 0:
        return None
    std = (sum((x - avg) ** 2 for x in clean) / len(clean)) ** 0.5
    cv = std / abs(avg)
    return float(cv)


def calculate_dividend_yield(ticker_info: Dict[str, Any]) -> Optional[float]:
    """
    Extract current dividend yield from ticker_info.keyIndicators.directYield.
    Returns yield as a fraction (e.g. 0.035 for 3.5%), or None.
    """
    try:
        dy = ticker_info.get("keyIndicators", {}).get("directYield")
        if dy is None:
            return None
        dy = float(dy)
        # Avanza stores as percentage (e.g. 3.5), normalize to fraction
        if dy > 1:
            dy = dy / 100.0
        return dy
    except Exception:
        return None


def calculate_piotroski_f_score(
    ticker_analysis: Dict[str, Any],
    ticker_info: Dict[str, Any],
    fcfy: Optional[float] = None,
    de_ratio: Optional[float] = None,
    roe: Optional[float] = None,
) -> Optional[int]:
    """
    Simplified Piotroski F-Score (0-9).
    Tests profitability, leverage, and operating efficiency signals.

    Profitability (4 points):
      1. ROE > 0
      2. Operating cash flow > 0 (FCF yield > 0 as proxy)
      3. ROE improving year-over-year
      4. Cash flow > net income (accruals quality)

    Leverage (3 points):
      5. D/E ratio decreased year-over-year
      6. Current ratio improved (not available, skip → always 0)
      7. No new share dilution (not available, skip → always 0)

    Efficiency (2 points):
      8. Gross margin improved year-over-year
      9. Asset turnover improved year-over-year
    """
    score = 0

    # --- Profitability ---
    # 1. Positive ROE
    if roe is not None and roe > 0:
        score += 1

    # 2. Positive operating cash flow (FCF yield as proxy)
    if fcfy is not None and fcfy > 0:
        score += 1

    # 3. ROE improving (last year vs prior year)
    roe_series = _as_vals(
        ticker_analysis.get("companyKeyRatiosByYear", {}).get("returnOnEquityRatio")
    )
    roe_clean = _clean(roe_series)
    if len(roe_clean) >= 2 and roe_clean[-1] > roe_clean[-2]:
        score += 1

    # 4. FCF > net income (accruals quality)
    # Compare FCF yield to earnings yield (1/PE) as a proxy
    pe_raw = ticker_info.get("keyIndicators", {}).get("priceEarningsRatio")
    if fcfy is not None and pe_raw is not None:
        try:
            earnings_yield = 1.0 / float(pe_raw) if float(pe_raw) > 0 else None
            if earnings_yield is not None and fcfy > earnings_yield:
                score += 1
        except Exception:
            pass

    # --- Leverage ---
    # 5. D/E ratio decreased
    de_series = [
        e["value"]
        for e in ticker_analysis.get("companyFinancialsByYear", {}).get(
            "debtToEquityRatio", []
        )
        if e.get("reportType") == "FULL_YEAR"
    ]
    if len(de_series) >= 2 and de_series[-1] < de_series[-2]:
        score += 1

    # 6-7: Current ratio / share dilution — data not available from Avanza, skip

    # --- Efficiency ---
    # 8. Gross margin improved
    margin_series = _as_vals(
        ticker_analysis.get("companyFinancialsByYear", {}).get("profitMargin")
    )
    margin_clean = _clean(margin_series)
    if len(margin_clean) >= 2 and margin_clean[-1] > margin_clean[-2]:
        score += 1

    # 9. Asset turnover improved (sales / total assets)
    sales = _as_vals(
        ticker_analysis.get("companyFinancialsByYear", {}).get("sales")
    )
    assets = [
        e["value"]
        for e in ticker_analysis.get("companyFinancialsByYear", {}).get(
            "totalAssets", []
        )
        if e.get("reportType") == "FULL_YEAR"
    ]
    if len(sales) >= 2 and len(assets) >= 2:
        turn_now = _safe_div(sales[-1], assets[-1])
        turn_prev = _safe_div(sales[-2], assets[-2])
        if turn_now is not None and turn_prev is not None and turn_now > turn_prev:
            score += 1

    return score


def calculate_de(ticker_analysis, ticker_id=None):
    raw = [
        {"date": e["date"], "value": e["value"]}
        for e in ticker_analysis["companyFinancialsByYear"]["debtToEquityRatio"]
        if e.get("reportType") == "FULL_YEAR" and "date" in e
    ]
    if not raw:
        return None, None
    latest = float(raw[-1]["value"])
    return latest, raw


def calculate_roe(ticker_analysis, ticker_id=None):
    # --- pull yearly series -------------------------------------------------
    net_profit = [
        {"date": e["date"], "value": e["value"]}
        for e in ticker_analysis["companyFinancialsByYear"]["netProfit"]
        if e.get("reportType") == "FULL_YEAR" and "date" in e
    ]
    total_assets = [
        e["value"]
        for e in ticker_analysis["companyFinancialsByYear"]["totalAssets"]
        if e.get("reportType") == "FULL_YEAR" and "date" in e
    ]
    total_liab = [
        e["value"]
        for e in ticker_analysis["companyFinancialsByYear"]["totalLiabilities"]
        if e.get("reportType") == "FULL_YEAR" and "date" in e
    ]

    if not (net_profit and total_assets and total_liab):
        return None, None  # missing data somewhere

    # --- calculate ROE year-by-year ----------------------------------------
    roe_series = []
    for i, np_entry in enumerate(net_profit):
        try:
            equity = total_assets[i] - total_liab[i]
            roe_val = np_entry["value"] / equity if equity else None
        except IndexError:  # unequal list lengths → skip
            continue
        roe_series.append({"date": np_entry["date"], "value": roe_val})

    if not roe_series:
        return None, None

    latest = roe_series[-1]["value"]
    return float(latest) if latest is not None else None, roe_series
