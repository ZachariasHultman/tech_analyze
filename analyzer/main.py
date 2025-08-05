import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from avanza.avanza import Avanza
from avanza.models import *
import os
import pandas as pd
from helper import *
from summary_manager import SummaryManager
from data_processing import *
from importlib.metadata import version
import pyotp
import hashlib
from historical_calc import calculate_metrics_given_hist
from datetime import date
import argparse

# TODO Analyze historical data and create tuning params from that


def setup_env():
    username = os.getenv("USERNAME")
    if username is None:
        raise Exception("Expected .env file to have a key named USERNAME")

    password = os.getenv("PASSWORD")
    if password is None:
        raise Exception("Expected .env file to have a key named PASSWORD")

    totpSecret = os.getenv("MY_TOTP_SECRET")
    if totpSecret is None:
        raise Exception("Expected .env file to have a key named TOTP_SECRET")
    # totp = pyotp.TOTP(totpSecret, digest=hashlib.sha1)
    # print(totpSecret)
    # print(totp.now())

    avanza_user = Avanza(
        {"username": username, "password": password, "totpSecret": totpSecret}
    )
    return avanza_user


def main():
    pd.set_option("display.max_rows", None)  # Show all rows
    pd.set_option("display.max_colwidth", None)  # Show full cell content
    print("Avanza API version: ", version("avanza-api"))
    print("yfinance version: ", version("yfinance"))
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--get_hist",
        type=bool,
        help="true/false. Get and store historical data for tickers",
        default=False,
    )
    ap.add_argument(
        "--use_hist",
        type=bool,
        help="true/false. Use historical data to run the script",
        default=False,
    )
    args = ap.parse_args()
    os.makedirs("data", exist_ok=True)

    manager = SummaryManager()
    if not args.use_hist:
        avanza = setup_env()
        ticker_ids = next(
            (
                item
                for item in avanza.get_watchlists()
                if item.get("name")
                == "Test"  # "Mina favoritaktier"  # "Berkshire"   # "Mina favoritaktier"  # "Äger"
            ),
            None,
        )["orderbookIds"]

        for ticker_id in tqdm(ticker_ids, desc="Processing tickers"):
            ticker_info = avanza.get_stock_info(ticker_id)

            if not ticker_info["sectors"] or ticker_id == "1640718":
                continue
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
            yahoo = yf.Ticker(yahoo_ticker_name)

            ticker_name, hist = get_data(
                ticker_id,
                ticker_info,
                manager,
                avanza,
                yahoo,
                get_hist=args.get_hist,
            )
            if args.get_hist:
                save_snapshot(
                    hist,
                    f"data/{ticker_name}_{date.today()}.csv",
                    asof=date.today(),
                )

        calculate_score(manager)

        manager._display(save_df=True)
    else:
        calculate_metrics_given_hist()

    return 0


if __name__ == "__main__":

    result = main()
