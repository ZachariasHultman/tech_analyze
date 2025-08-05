import pandas as pd
from currency_converter import CurrencyConverter
import math
import requests
from avanza.avanza import Resolution, TimePeriod
from analyzer.helper import *
from analyzer.get_NAV_data import get_nav_data
import yfinance as yf
from datetime import date, datetime
from analyzer.metrics import ticker_reporting_currency_map, ticker_currency_map
import time


import requests
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np


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


def calculate_profit_per_share(ticker_analysis):
    ticker_profit_per_share = [
        entry["value"]
        for entry in ticker_analysis["companyKeyRatiosByYear"]["earningsPerShare"]
        if "reportType" in entry and entry["reportType"] == "FULL_YEAR"
    ][-1]
    return ticker_profit_per_share


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
def calculate_profit_margin(ticker_analysis):
    raw = [
        {"date": e["date"], "value": e["value"]}
        for e in ticker_analysis["companyFinancialsByYear"]["profitMargin"]
        if e.get("reportType") == "FULL_YEAR" and "date" in e
    ]
    if not raw:
        return None, None

    latest = float(raw[-1]["value"])  # newest numeric margin
    return latest, raw  # (value, full record list)


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


def calculate_CAGR_helper(df, years):
    df = df.sort_index()

    end_date = df.index[-1]

    start_date = end_date - pd.DateOffset(years=years)

    closest_start_idx = df.index.get_indexer([start_date], method="backfill")[0]
    closest_start = df.iloc[closest_start_idx]["close"]
    end_price = df["close"].iloc[-1]

    cagr = (end_price / closest_start) ** (1 / years) - 1
    return float(cagr)


def calculate_closing_CAGR(
    avanza,
    ticker,
    *,
    use_hist: bool = False,
    hist_row: pd.Series | None = None,
    years_tuple: tuple[int, ...] = (3, 2, 1),
) -> list[float] | None:
    """
    Return [CAGR_3Y, CAGR_2Y, CAGR_1Y] by default (old behaviour).

    Parameters
    ----------
    use_hist : bool (default False)
        • False → live fetch via Avanza (unchanged)
        • True  → derive from hist_row["ohlc"] that you loaded with get_hist_data()
    hist_row : pd.Series
        The row from your historical DataFrame; required when use_hist=True.
    years_tuple : tuple[int,...]
        Which year-spans to calculate (defaults to (3,2,1) → old behaviour).
    """
    # ── HISTORICAL MODE ────────────────────────────────────────────────
    if use_hist:
        if (
            hist_row is None
            or "ohlc" not in hist_row
            or hist_row["ohlc"].empty
            or hist_row["ohlc"] is None
        ):
            return None

        df = hist_row["ohlc"].copy()
        df = df.rename(columns={"close": "close"}).astype(float)
        df = df.sort_index()
        cagr_vals = [calculate_CAGR_helper(df, y) for y in years_tuple]
        return cagr_vals if any(v is not None for v in cagr_vals) else None

    # ── LIVE (old) MODE ────────────────────────────────────────────────
    try:
        ticker_chart_data = avanza.get_chart_data(
            order_book_id=ticker,
            period=TimePeriod.FIVE_YEARS,
            resolution=Resolution.DAY,
        )
    except requests.exceptions.HTTPError:
        return None

    timestamps = [e["timestamp"] for e in ticker_chart_data["ohlc"]]
    closing_prices = [e["close"] for e in ticker_chart_data["ohlc"]]
    dates = pd.to_datetime(timestamps, unit="ms")
    data = pd.DataFrame({"close": closing_prices}, index=dates).sort_index()

    return [calculate_CAGR_helper(data, y) for y in years_tuple]


def calculate_PEG(pe: list[float], cagr: list[float] | None):
    """
    PEG = PE / CAGR
    • `pe`    list of the last 3 PE values  (oldest → newest)
    • `cagr`  list of matching CAGR values (ditto)
    """
    if cagr is None or not pe:
        return None

    pe = pe[-3:]
    peg = [float(p / c) if c else None for p, c in zip(pe, cagr)]
    return peg[-1]  # return most recent PEG


from currency_converter import CurrencyConverter
import math


def sync_currency(from_currency: str, to_currency: str = "SEK"):
    """
    Converts between reporting and trading currencies.
    Returns:
        currency_match (bool)
        convert_to_sek (bool)
        exchange_rate (float)
        sek_rate (float)
    """
    c = CurrencyConverter()

    if not from_currency or not to_currency:
        return False, False, None, None

    exchange_rate = c.convert(1, from_currency, to_currency)
    sek_rate = c.convert(1, from_currency, "SEK")

    currency_match = from_currency == to_currency
    convert_to_sek = to_currency != "SEK"

    return currency_match, convert_to_sek, exchange_rate, sek_rate


def calculate_free_cashflow_yield(yahoo_ticker, stock_info, df_hist=None):
    try:
        from_currency = ticker_reporting_currency_map.get(yahoo_ticker.ticker)
        to_currency = stock_info["keyIndicators"]["marketCapital"]["currency"]
        market_cap = stock_info["keyIndicators"]["marketCapital"]["value"]

        # Infer shares outstanding from Avanza market‐cap + last close in df_hist
        try:
            last_close_price = df_hist["close"].iloc[-1]
            shares_outstanding = market_cap / last_close_price
        except (KeyError, TypeError, ZeroDivisionError):
            return None, None, None, None

        #  Fetch FCF history
        cash_flow_df = yahoo_ticker.cashflow
        if "Free Cash Flow" not in cash_flow_df.index:
            print(" 'Free Cash Flow' not found for", yahoo_ticker.ticker)
            return None, None, None, None

        free_cash_flow_hist = cash_flow_df.loc["Free Cash Flow"]

        # Currency conversion
        currency_match, convert_to_sek, ex_rate, sek_rate = sync_currency(
            from_currency=from_currency, to_currency=to_currency
        )
        if not currency_match:
            free_cash_flow_hist = free_cash_flow_hist * ex_rate
        if convert_to_sek:
            free_cash_flow_hist = free_cash_flow_hist * sek_rate

        # Build historical FCF‐yield dict (if price history provided)
        fcf_yield_hist = {}
        if df_hist is not None:
            df_hist.index = pd.to_datetime(df_hist.index)
            for dt, fcf in free_cash_flow_hist.items():
                if pd.isna(fcf):
                    continue
                try:
                    close_price = df_hist.loc[:dt]["close"].iloc[-1]
                    market_cap_hist = close_price * shares_outstanding
                    fcf_yield_hist[dt.strftime("%Y-%m-%d")] = float(
                        fcf / market_cap_hist
                    )
                except (KeyError, IndexError):
                    continue

        # Latest values
        latest_fcf = float(free_cash_flow_hist.iloc[0])
        fcf_yield_now = latest_fcf / market_cap

        # Serialize free_cash_flow_hist to dict with ISO dates + floats
        free_cf_hist_dict = {
            dt.strftime("%Y-%m-%d"): (None if pd.isna(val) else float(val))
            for dt, val in free_cash_flow_hist.items()
        }

        return (
            fcf_yield_now,  # scalar
            latest_fcf,  # scalar
            fcf_yield_hist,  # dict  { "YYYY-MM-DD": float }
            free_cf_hist_dict,  # dict  { "YYYY-MM-DD": float | None }
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
    import numpy as np
    import pandas as pd

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
