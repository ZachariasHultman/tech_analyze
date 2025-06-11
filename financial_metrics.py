import pandas as pd
from currency_converter import CurrencyConverter
import math
import requests
from avanza.avanza import Resolution, TimePeriod
from helper import *
from get_NAV_data import get_nav_data
import yfinance as yf
from datetime import date, datetime
from metrics import ticker_reporting_currency_map, ticker_currency_map
import time


def calculate_sma200(avanza, ticker_id):
    try:
        ticker_chart_data = avanza.get_chart_data(
            order_book_id=ticker_id,
            period=TimePeriod.FIVE_YEARS,
            resolution=Resolution.DAY,
        )["ohlc"]
    except requests.exceptions.HTTPError:
        return None, None, None

    last_week_average_close = (
        sum(entry["close"] for entry in ticker_chart_data[-7:]) / 7
    )

    df_hist = pd.DataFrame(ticker_chart_data)
    df_hist["date"] = df_hist["timestamp"].apply(
        lambda ts: datetime.utcfromtimestamp(ts / 1000).date()
    )
    df_hist = df_hist[["date", "open", "high", "low", "close", "totalVolumeTraded"]]

    closing_prices = [entry["close"] for entry in ticker_chart_data]
    data = pd.DataFrame({"close": closing_prices})
    # Calculate the Rolling SMA200
    data["sma200"] = data["close"].rolling(window=200).mean()[-200:]
    # Drop rows with NaN (first 199 rows won't have SMA200)
    data = data.dropna()

    sma200 = data["sma200"].to_list()[-1]
    slope = calculate_slope(data["sma200"].to_list())

    return sma200, last_week_average_close, float(slope), df_hist


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
        if e.get("reportType") == "FULL_YEAR"
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
        if e.get("reportType") == "FULL_YEAR"
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
        if e.get("reportType") == "FULL_YEAR"
    ]
    qtr = [
        {"date": e["date"], "value": e["value"]}
        for e in ticker_analysis["companyFinancialsByQuarter"]["sales"]
        if "value" and "date" in e
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
        if "reportType" in entry and entry["reportType"] == "FULL_YEAR"
    ]
    latest_vals = [d["value"] for d in pe][-5:]
    if len(pe) >= 1:
        return latest_vals, pe
    else:
        return None, None


def calculate_CAGR_helper(df, years):
    """Calculates CAGR for the given number of years."""

    # Ensure the DataFrame index is sorted
    df = df.sort_index()

    # Define end date as the last available date in the dataset
    end_date = df.index[-1]

    # Define start date based on the number of years
    start_date = end_date - pd.DateOffset(years=years)

    # Find the closest available price **on or after** start_date
    closest_start_idx = df.index.get_indexer([start_date], method="backfill")[0]
    closest_start = df.iloc[closest_start_idx]["close"]

    # Get the closing price at the end date
    end_price = df["close"].iloc[-1]

    # Compute CAGR
    cagr = (end_price / closest_start) ** (1 / years) - 1

    return float(cagr)


def calculate_closing_CAGR(avanza, ticker):
    try:
        ticker_chart_data = avanza.get_chart_data(
            order_book_id=ticker,
            period=TimePeriod.FIVE_YEARS,
            resolution=Resolution.DAY,
        )
    except requests.exceptions.HTTPError as e:
        return None
    # Extract timestamps and closing prices
    timestamps = [entry["timestamp"] for entry in ticker_chart_data["ohlc"]]
    closing_prices = [entry["close"] for entry in ticker_chart_data["ohlc"]]

    # Convert timestamps to pandas datetime format (milliseconds since 1970)
    dates = pd.to_datetime(timestamps, unit="ms")

    # Create DataFrame with both Date and Closing Price
    data = pd.DataFrame({"Date": dates, "close": closing_prices})

    # Set the date as index
    data.set_index("Date", inplace=True)
    data = data.sort_index()
    years = [3, 2, 1]
    cagr = [calculate_CAGR_helper(data, y) for y in years]

    return cagr


def calculate_PEG(pe, cagr):
    # PEG is orderd as CAGR, which is [oldest, ... , newest]
    if cagr == None or not pe:
        return None

    pe = pe[-3:]
    peg = []
    for p, c in zip(pe, cagr):
        if c != 0:  # Avoid division by zero
            peg.append(float(p / c))
        else:
            peg.append(None)  # Handle cases where CAGR is 0 or negative
    return peg[-1]


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


def extract_ev_ebit_ratio(avanza_data):
    """
    Extracts the EV/EBIT ratio from Avanza’s `stockKeyRatiosByYear`.

    Returns:
        latest_ev_ebit (float or None): The most recent EV/EBIT value.
        ev_ebit_hist (dict[str, float]): Mapping from "YYYY-MM-DD" to EV/EBIT.
    """
    ev_ebit_list = avanza_data.get("stockKeyRatiosByYear", {}).get("evEbitRatio", [])
    ev_ebit_hist_raw = {}
    for entry in ev_ebit_list:
        try:
            date_ts = pd.Timestamp(entry["date"])
            ev_ebit_hist_raw[date_ts] = entry["value"]
        except (KeyError, ValueError):
            continue

    # Determine latest
    if ev_ebit_hist_raw:
        latest_date = max(ev_ebit_hist_raw.keys())
        latest_ev_ebit = ev_ebit_hist_raw[latest_date]
    else:
        latest_ev_ebit = None

    # Convert keys to ISO‐string
    ev_ebit_hist = {
        dt.strftime("%Y-%m-%d"): float(val) for dt, val in ev_ebit_hist_raw.items()
    }

    return latest_ev_ebit, ev_ebit_hist


def extract_netdebt_ebitda_ratio(avanza_data):
    """
    Extracts the Net Debt/EBITDA ratio from Avanza’s `companyKeyRatiosByYear`.

    Returns:
        latest_nd_ebitda (float or None): The most recent Net Debt/EBITDA value.
        nd_ebitda_hist (dict[str, float]): Mapping from "YYYY-MM-DD" to Net Debt/EBITDA.
    """
    nd_ebitda_list = avanza_data.get("companyKeyRatiosByYear", {}).get(
        "netDebtEbitdaRatio", []
    )
    nd_ebitda_hist_raw = {}
    for entry in nd_ebitda_list:
        try:
            date_ts = pd.Timestamp(entry["date"])
            nd_ebitda_hist_raw[date_ts] = entry["value"]
        except (KeyError, ValueError):
            continue

    # Determine latest
    if nd_ebitda_hist_raw:
        latest_date = max(nd_ebitda_hist_raw.keys())
        latest_nd_ebitda = nd_ebitda_hist_raw[latest_date]
    else:
        latest_nd_ebitda = None

    # Convert keys to ISO‐string
    nd_ebitda_hist = {
        dt.strftime("%Y-%m-%d"): float(val) for dt, val in nd_ebitda_hist_raw.items()
    }
    return latest_nd_ebitda, nd_ebitda_hist


def calculate_NAV_discount(ticker_name):
    df_nav = get_nav_data(ticker_name)
    nav = df_nav.iloc[-300:]["SUBSTANSVÄRDE"].to_numpy()
    calculated_nav = df_nav.iloc[-300:]["BERÄKNAT_SUBSTANSVÄRDE"].to_numpy()
    price = df_nav.iloc[-300:]["PRIS"].to_numpy()

    nav_discount = (price - nav) / nav
    calculated_nav_discount = (price - calculated_nav) / calculated_nav
    if nav_discount.size > 0:
        nav_trend = calculate_slope(nav_discount.tolist())
    else:
        return None, None, None, None, None
    return (
        float(nav_discount[-30:].mean()),
        float(calculated_nav_discount[-30:].mean()),
        float(nav_trend),
        nav_discount,
        calculated_nav_discount,
    )


def calculate_de(ticker_analysis, ticker_id=None):
    raw = [
        {"date": e["date"], "value": e["value"]}
        for e in ticker_analysis["companyFinancialsByYear"]["debtToEquityRatio"]
        if e.get("reportType") == "FULL_YEAR"
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
        if e.get("reportType") == "FULL_YEAR"
    ]
    total_assets = [
        e["value"]
        for e in ticker_analysis["companyFinancialsByYear"]["totalAssets"]
        if e.get("reportType") == "FULL_YEAR"
    ]
    total_liab = [
        e["value"]
        for e in ticker_analysis["companyFinancialsByYear"]["totalLiabilities"]
        if e.get("reportType") == "FULL_YEAR"
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
