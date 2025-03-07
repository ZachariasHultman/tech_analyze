import yfinance as yf
from tqdm import tqdm
import re
from metrics import HIGHEST_WEIGHT_METRICS
from helper import *
from financial_metrics import *


def get_data(ticker_ids, manager, avanza):
    for ticker_id in tqdm(ticker_ids, desc="Processing tickers"):
        # for ticker_id in ticker_ids:
        ticker_info = avanza.get_stock_info(ticker_id)
        # print(ticker_info["listing"]["tickerSymbol"])

        ticker_analysis = avanza.get_analysis(ticker_id)
        yahoo_ticker_name = ticker_info["listing"]["tickerSymbol"]
        if ticker_info["listing"]["countryCode"] == "SE":
            yahoo_ticker_name = yahoo_ticker_name.replace(" ", "-") + ".ST"
        elif ticker_info["listing"]["countryCode"] == "DK":
            yahoo_ticker_name = yahoo_ticker_name.replace(" ", "-") + ".CO"
        elif ticker_info["listing"]["countryCode"] == "NO":
            yahoo_ticker_name = yahoo_ticker_name.replace(" ", "-") + ".OL"
        elif ticker_info["listing"]["countryCode"] == "DE":
            yahoo_ticker_name = re.match(r"^[A-Z]+", yahoo_ticker_name)
            yahoo_ticker_name = yahoo_ticker_name.group() + ".DE"

        # print(yahoo_ticker_name)
        yahoo_ticker = yf.Ticker(yahoo_ticker_name)
        # print("ticker analysis keys",ticker_analysis.keys())

        if not ticker_info["sectors"] or ticker_id == "1640718":
            continue
        investment = any(
            sector["sectorName"] == "Investmentbolag"
            for sector in ticker_info["sectors"]
        )

        ticker_name = ticker_info["name"] + " " + ticker_info["orderbookId"]

        if not investment:
            sector = [sector for sector in ticker_info["sectors"]]
            manager._initialize_template(ticker_name, sector)
            # Calculate sma200
            sma200, weekly_average_close, sma200_slope = calculate_sma200(
                avanza, ticker_id
            )

            manager._update(ticker_name, sector, "sma200 slope status", sma200_slope)

            # Calculate profit per share trend
            profit_per_share_trend = calculate_profit_per_share_trend(ticker_analysis)
            manager._update(
                ticker_name,
                sector,
                "profit per share trend status",
                profit_per_share_trend,
            )
            # Calculate profit margin trend.
            profit_margin = calculate_average_profit_margin(ticker_analysis)
            profit_margin_trend = calculate_profit_margin_trend(ticker_analysis)
            manager._update(ticker_name, sector, "profit margin status", profit_margin)
            manager._update(
                ticker_name,
                sector,
                "profit margin trend status",
                profit_margin_trend,
            )
            # Calculate Revenue trend for last years.
            revenue_trend_year, revenue_trend_quarter = calculate_revenue_trend(
                ticker_analysis
            )
            manager._update(
                ticker_name, sector, "revenue trend year status", revenue_trend_year
            )
            manager._update(
                ticker_name,
                sector,
                "revenue trend quarter status",
                revenue_trend_quarter,
            )
            # Calculate P/E
            pe = calculate_PE(ticker_analysis)

            # Calculate The PEG ratio (Price/Earnings-to-Growth)
            cagr = calculate_closing_CAGR(avanza, ticker_id)
            peg = calculate_PEG(pe, cagr)
            manager._update(ticker_name, sector, "peg status", peg)
            manager._update(
                ticker_name,
                sector,
                "cagr-pe compare status",
                [cagr[-1], pe[-1]] if cagr and pe else [None, None],
            )
            # Calculate free cashflow yield.
            free_cashflow_yield, free_cashflow = calculate_free_cashflow_yield(
                yahoo_ticker
            )
            manager._update(ticker_name, sector, "fcfy status", free_cashflow_yield)
            manager._update(ticker_name, sector, "fcf status", free_cashflow)

            # Calculate ebitda and net debt.
            ebitda = calculate_ebitda(yahoo_ticker)
            net_debt = calculate_net_debt(yahoo_ticker)
            manager._update(
                ticker_name, sector, "net debt - ebitda status", [net_debt, ebitda]
            )
            # Calculate debt to equity ratio
            de_ratio = calculate_de(ticker_analysis)
            manager._update(ticker_name, sector, "de status", de_ratio)
            # Calculate debt to equity ratio
            roe = calculate_roe(ticker_analysis)
            manager._update(ticker_name, sector, "roe status", roe)

        else:
            sector = [{"sectorId": "51", "sectorName": "Investmentbolag"}]
            manager._initialize_template(ticker_name, sector)
            # Calculate sma200
            sma200, weekly_average_close, sma200_slope = calculate_sma200(
                avanza, ticker_id
            )

            manager._update(ticker_name, sector, "sma200 slope status", sma200_slope)

            # Calculate profit per share trend
            profit_per_share_trend = calculate_profit_per_share_trend(ticker_analysis)
            manager._update(
                ticker_name,
                sector,
                "profit per share trend status",
                profit_per_share_trend,
            )
            # Calculate P/E trend
            pe = calculate_PE(ticker_analysis)

            cagr = calculate_closing_CAGR(avanza, ticker_id)
            # The combination of CAGR
            manager._update(
                ticker_name,
                sector,
                "cagr-pe compare status",
                [cagr[-1], pe[-1]] if cagr else [None, None],
            )
            # Calculate ebitda and net debt.
            ebit = calculate_ebit(yahoo_ticker)
            net_debt = calculate_net_debt(yahoo_ticker)
            manager._update(
                ticker_name, sector, "net debt - ebit status", [net_debt, ebit]
            )
            # Check NAV discount and trend.
            nav_discount, calculated_nav_discount, nav_discount_trend = (
                calculate_NAV_discount(ticker_info["listing"]["tickerSymbol"])
            )
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
            free_cashflow_yield, free_cashflow = calculate_free_cashflow_yield(
                yahoo_ticker
            )
            manager._update(ticker_name, sector, "fcf status", free_cashflow)
            roe = calculate_roe(ticker_analysis)
            manager._update(ticker_name, sector, "roe status", roe)


def calculate_score(manager):
    excluded_columns = {"sector", "points"}  # Set of columns to exclude

    if bool(manager.summary):
        manager.summary = pd.DataFrame(manager.summary).T
        # Filter only existing `_score` columns to avoid KeyErrors
        bonus_score_columns = [
            col for col in HIGHEST_WEIGHT_METRICS if col in manager.summary.columns
        ]
        # Create a list of the corresponding `_score` column names
        existing_bonus_score_columns = [col + "_score" for col in bonus_score_columns]
        for col in manager.template:
            if col not in excluded_columns:
                manager.summary[col + "_score"] = manager.summary.apply(
                    lambda row: manager._assign_points(row, col), axis=1
                )

        # Calculate total points based on the newly created *_score columns
        manager.summary["points"] = manager.summary[
            [col + "_score" for col in manager.template if col not in excluded_columns]
        ].sum(axis=1)

        # Give bonus point of 1 if all high weight columns are green
        if existing_bonus_score_columns:
            print("checks bonus score")
            manager.summary["points"] += (
                (manager.summary[existing_bonus_score_columns] > 0)
                .all(axis=1)
                .astype(int)
            )
    else:
        manager.summary = pd.DataFrame()

    if bool(manager.summary_investment):
        manager.summary_investment = pd.DataFrame(manager.summary_investment).T

        bonus_score_columns_investment = [
            col
            for col in HIGHEST_WEIGHT_METRICS
            if col in manager.summary_investment.columns
        ]
        # Create a list of the corresponding `_score` column names
        existing_bonus_score_columns_investment = [
            col + "_score" for col in bonus_score_columns_investment
        ]
        # Apply scoring logic and create new columns for each metric
        for col in manager.template_investment:
            if col not in excluded_columns:
                manager.summary_investment[col + "_score"] = (
                    manager.summary_investment.apply(
                        lambda row: manager._assign_points(row, col), axis=1
                    )
                )

        # Calculate total points based on the newly created *_score columns
        manager.summary_investment["points"] = manager.summary_investment[
            [
                col + "_score"
                for col in manager.template_investment
                if col not in excluded_columns
            ]
        ].sum(axis=1)

        # Give bonus point of 1 if all high weight columns are green
        if existing_bonus_score_columns_investment:
            print("checks bonus score")
            manager.summary_investment["points"] += (
                (
                    manager.summary_investment[existing_bonus_score_columns_investment]
                    > 0
                )
                .all(axis=1)
                .astype(int)
            )
    else:
        manager.summary_investment = pd.DataFrame()
