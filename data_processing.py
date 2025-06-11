import yfinance as yf
from tqdm import tqdm
import re
from metrics import HIGHEST_WEIGHT_METRICS
from helper import *
from financial_metrics import *


def get_data(
    ticker_id,
    ticker_info,
    manager,
    avanza,
    yahoo_ticker,
    get_hist=False,
):
    # print(ticker_info["listing"]["tickerSymbol"])

    ticker_analysis = avanza.get_analysis(ticker_id)

    investment = any(
        sector["sectorName"] == "Investmentbolag" for sector in ticker_info["sectors"]
    )

    if get_hist:
        hist = {}

    ticker_name = ticker_info["name"] + " " + ticker_info["orderbookId"]

    if not investment:
        sector = [sector for sector in ticker_info["sectors"]]
        manager._initialize_template(ticker_name, sector)
        # Calculate sma200
        sma200, weekly_average_close, sma200_slope, closing_hist_data = (
            calculate_sma200(avanza, ticker_id)
        )

        manager._update(ticker_name, sector, "sma200 slope status", sma200_slope)

        # Calculate profit per share trend
        profit_per_share_trend, profit_per_share_hist = (
            calculate_profit_per_share_trend(ticker_analysis)
        )
        manager._update(
            ticker_name,
            sector,
            "profit per share trend status",
            profit_per_share_trend,
        )
        # Calculate profit margin trend.
        profit_margin, profit_margin_hist = calculate_profit_margin(ticker_analysis)
        profit_margin_trend = calculate_profit_margin_trend(ticker_analysis)
        manager._update(ticker_name, sector, "profit margin status", profit_margin)
        manager._update(
            ticker_name,
            sector,
            "profit margin trend status",
            profit_margin_trend,
        )
        # Calculate Revenue trend for last years.
        (
            revenue_trend_year,
            revenue_trend_quarter,
            revenue_year_hist,
            revenue_quarter_hist,
        ) = calculate_revenue_trend(ticker_analysis)
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
        pe, pe_hist = calculate_PE(ticker_analysis)

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
        (
            free_cashflow_yield,
            free_cashflow,
            free_cashflow_yield_hist,
            free_cashflow_hist,
        ) = calculate_free_cashflow_yield(yahoo_ticker, ticker_info, closing_hist_data)
        manager._update(ticker_name, sector, "fcfy status", free_cashflow_yield)
        manager._update(ticker_name, sector, "fcf status", free_cashflow)

        # Calculate ebitda and net debt.
        netDebtEbitdaRatio, netDebtEbitdaRatio_hist = extract_netdebt_ebitda_ratio(
            ticker_analysis
        )
        manager._update(
            ticker_name,
            sector,
            "net debt - ebitda status",
            netDebtEbitdaRatio,
        )
        # Calculate debt to equity ratio
        de_ratio, de_ratio_hist = calculate_de(ticker_analysis)
        manager._update(ticker_name, sector, "de status", de_ratio)
        # Calculate debt to equity ratio
        roe, roe_hist = calculate_roe(ticker_analysis)
        manager._update(ticker_name, sector, "roe status", roe)
        if get_hist:
            hist["sector"] = sector
            hist["ohlc"] = closing_hist_data
            hist["profit_per_share"] = profit_per_share_hist
            hist["pe"] = pe_hist
            hist["roe"] = roe_hist
            hist["profit_margin"] = profit_margin_hist
            hist["revenue_year"] = revenue_year_hist
            hist["revenue_quarter"] = revenue_quarter_hist
            hist["de_ratio"] = de_ratio_hist
            hist["free_cashflow_yield"] = free_cashflow_yield_hist
            hist["free_cashflow"] = free_cashflow_hist
            hist["netDebtEbitdaRatio"] = netDebtEbitdaRatio_hist

    else:
        sector = [{"sectorId": "51", "sectorName": "Investmentbolag"}]
        manager._initialize_template(ticker_name, sector)
        # Calculate sma200
        sma200, weekly_average_close, sma200_slope, closing_hist_data = (
            calculate_sma200(avanza, ticker_id)
        )

        manager._update(ticker_name, sector, "sma200 slope status", sma200_slope)

        # Calculate profit per share trend
        profit_per_share_trend, profit_per_share_hist = (
            calculate_profit_per_share_trend(ticker_analysis)
        )
        manager._update(
            ticker_name,
            sector,
            "profit per share trend status",
            profit_per_share_trend,
        )
        # Calculate P/E trend
        pe, pe_hist = calculate_PE(ticker_analysis)

        cagr = calculate_closing_CAGR(avanza, ticker_id)
        # The combination of CAGR
        manager._update(
            ticker_name,
            sector,
            "cagr-pe compare status",
            [cagr[-1], pe[-1]] if cagr else [None, None],
        )
        # Calculate ebitda and net debt.
        evEbit, evEbit_hist = extract_ev_ebit_ratio(ticker_analysis)
        manager._update(ticker_name, sector, "net debt - ebit status", evEbit)
        # Check NAV discount and trend.
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
        (
            free_cashflow_yield,
            free_cashflow,
            free_cashflow_yield_hist,
            free_cashflow_hist,
        ) = calculate_free_cashflow_yield(yahoo_ticker, ticker_info)
        manager._update(ticker_name, sector, "fcf status", free_cashflow)
        roe, roe_hist = calculate_roe(ticker_analysis)
        manager._update(ticker_name, sector, "roe status", roe)

        if get_hist:
            hist["sector"] = sector
            hist["ohlc"] = closing_hist_data
            hist["profit_per_share"] = profit_per_share_hist
            hist["pe"] = pe_hist
            hist["roe"] = roe_hist
            hist["nav_discount"] = nav_discount_hist
            hist["calculated_nav_discount"] = calculated_nav_discount_hist
            hist["evEbit"] = evEbit_hist
            hist["free_cashflow_yield"] = free_cashflow_yield_hist
            hist["free_cashflow"] = free_cashflow_hist

    if get_hist:
        return ticker_name, hist
    else:
        return ticker_name, None


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
