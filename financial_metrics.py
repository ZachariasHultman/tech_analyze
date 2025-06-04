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


def calculate_profit_per_share_trend(ticker_analysis, ticker_id=None):
    ticker_profit_per_share = [
        entry["value"]
        for entry in ticker_analysis["companyKeyRatiosByYear"]["earningsPerShare"]
        if "reportType" in entry and entry["reportType"] == "FULL_YEAR"
    ]

    if len(ticker_profit_per_share) > 1:
        slope = calculate_slope(ticker_profit_per_share[-5:], ticker_id)
        return float(slope), ticker_profit_per_share
    else:
        return None, None


def calculate_profit_margin(ticker_analysis):
    ticker_profit_margin = [
        entry["value"]
        for entry in ticker_analysis["companyFinancialsByYear"]["profitMargin"]
        if "reportType" in entry and entry["reportType"] == "FULL_YEAR"
    ]
    if ticker_profit_margin:
        return float(ticker_profit_margin[-1]), ticker_profit_margin
    else:
        return None, None


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
    ticker_revenue_by_year = [
        entry["value"]
        for entry in ticker_analysis["companyFinancialsByYear"]["sales"]
        if "reportType" in entry and entry["reportType"] == "FULL_YEAR"
    ]
    ticker_revenue_by_quarter = [
        entry["value"]
        for entry in ticker_analysis["companyFinancialsByQuarter"]["sales"]
        if "value" in entry
    ]
    if len(ticker_revenue_by_quarter) > 1 and len(ticker_revenue_by_year) > 1:
        slope_year = calculate_slope(ticker_revenue_by_year[-5:])
        slope_quarter = calculate_slope(ticker_revenue_by_quarter)
        return (
            float(slope_year),
            float(slope_quarter),
            ticker_revenue_by_year,
            ticker_revenue_by_quarter,
        )
    else:
        return None, None, None, None


def calculate_PE(ticker_analysis):
    pe = [
        entry["value"]
        for entry in ticker_analysis["stockKeyRatiosByYear"]["priceEarningsRatio"]
        if "reportType" in entry and entry["reportType"] == "FULL_YEAR"
    ]

    if len(pe) >= 1:
        return pe[-5:], pe
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


def calculate_free_cashflow_yield(yahoo_ticker, stock_info):
    try:
        from_currency = ticker_reporting_currency_map.get(
            yahoo_ticker.ticker
        )  # Reporting currency
        # Get market cap and its currency from Avanza stock_info
        market_cap = stock_info["keyIndicators"]["marketCapital"]["value"]
        to_currency = stock_info["keyIndicators"]["marketCapital"]["currency"]

        # Get free cash flow from Yahoo
        cash_flow = yahoo_ticker.cashflow
        free_cash_flow = cash_flow.loc["Free Cash Flow"].iloc[0]

        # Handle currency conversion
        currency_match, convert_to_sek, exchange_rate, sek_rate = sync_currency(
            from_currency=from_currency, to_currency=to_currency
        )

    except (KeyError, IndexError, TypeError) as e:
        return None, None

    if (
        market_cap is not None
        and free_cash_flow is not None
        and not math.isnan(market_cap)
        and not math.isnan(free_cash_flow)
    ):
        # Adjust FCF if needed
        if not currency_match:
            free_cash_flow *= exchange_rate
        if convert_to_sek:
            free_cash_flow *= sek_rate

        free_cash_flow_yield = free_cash_flow / market_cap
        return free_cash_flow_yield, free_cash_flow

    return None, None


def calculate_ebitda(yahoo_ticker):
    income_statement = yahoo_ticker.financials
    try:
        ebitda = income_statement.loc["EBITDA"].iloc[0]

        return ebitda if not np.isnan(ebitda) else None

    except KeyError as e:
        return None


def calculate_ebit(yahoo_ticker):
    income_statement = yahoo_ticker.financials
    try:
        net_income = income_statement.loc["Net Income"].iloc[0]
        interest_expense = income_statement.loc["Interest Expense"].iloc[0]
        tax_provision = income_statement.loc["Tax Provision"].iloc[0]
        # Calculate EBITDA
        ebit = net_income + interest_expense + tax_provision
        return ebit if not np.isnan(ebit) else None

    except KeyError as e:
        return None


def calculate_net_debt(yahoo_ticker):
    balance_sheet = yahoo_ticker.balance_sheet
    net_debt = 0
    total_debt = np.nan
    try:
        # Extract Total Debt
        if "Net Debt" in balance_sheet.index:
            net_debt = balance_sheet.loc["Net Debt"].iloc[0]
            if pd.notna(net_debt):
                return net_debt

        if "Total Debt" in balance_sheet.index:
            total_debt = balance_sheet.loc["Total Debt"].iloc[0]
        elif "Total Liabilities Net Minority Interest" in balance_sheet.index:
            total_debt = balance_sheet.loc[
                "Total Liabilities Net Minority Interest"
            ].iloc[0]
        if np.isnan(total_debt):
            # Manually calculate Total Debt if necessary
            long_term_debt = balance_sheet.loc["Long Term Debt"].iloc[0]
            current_debt = balance_sheet.loc["Current Debt"].iloc[0]
            total_debt = long_term_debt + current_debt

        # Extract Cash and Cash Equivalents
        cash_and_equivalents = balance_sheet.loc["Cash And Cash Equivalents"].iloc[0]

        # Calculate Net Debt
        net_debt = total_debt - cash_and_equivalents

        if np.isnan(net_debt):
            net_debt = None
        return net_debt

    except (KeyError, yf.exceptions.YFRateLimitError) as e:
        print(e)
        return None


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
        return None, None, None

    return (
        float(nav_discount[-30:].mean()),
        float(calculated_nav_discount[-30:].mean()),
        float(nav_trend),
        nav_discount,
        calculated_nav_discount,
        nav_trend,
    )


def calculate_de(ticker_analysis, ticker_id=None):
    debt_to_equity = [
        entry["value"]
        for entry in ticker_analysis["companyFinancialsByYear"]["debtToEquityRatio"]
        if "reportType" in entry and entry["reportType"] == "FULL_YEAR"
    ]
    if debt_to_equity:
        return float(debt_to_equity[-1]), debt_to_equity
    else:
        return None


def calculate_roe(ticker_analysis, ticker_id=None):
    net_profit = np.array(
        [
            entry["value"]
            for entry in ticker_analysis["companyFinancialsByYear"]["netProfit"]
            if "reportType" in entry and entry["reportType"] == "FULL_YEAR"
        ]
    )
    total_assets = np.array(
        [
            entry["value"]
            for entry in ticker_analysis["companyFinancialsByYear"]["totalAssets"]
            if "reportType" in entry and entry["reportType"] == "FULL_YEAR"
        ]
    )
    total_liabilities = np.array(
        [
            entry["value"]
            for entry in ticker_analysis["companyFinancialsByYear"]["totalLiabilities"]
            if "reportType" in entry and entry["reportType"] == "FULL_YEAR"
        ]
    )

    shareholder_equity = total_assets - total_liabilities
    roe = net_profit / shareholder_equity
    if roe.size > 1:
        return float(roe[-1]), roe
    else:
        return None
