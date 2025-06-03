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

from datetime import date
import argparse


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
        "--start", type=lambda s: pd.to_datetime(s).date(), help="YYYY-MM-DD"
    )
    ap.add_argument("--end", type=lambda s: pd.to_datetime(s).date(), help="YYYY-MM-DD")
    args = ap.parse_args()
    os.makedirs("data", exist_ok=True)

    manager = SummaryManager()
    avanza = setup_env()
    ticker_ids = next(
        (
            item
            for item in avanza.get_watchlists()
            if item.get("name")
            == "Äger"  # "Mina favoritaktier"  # "Berkshire"   # "Mina favoritaktier"  # "Äger"
        ),
        None,
    )["orderbookIds"]

    for ticker_id in tqdm(ticker_ids, desc="Processing tickers"):
        ticker_info = avanza.get_stock_info(ticker_id)

        if not ticker_info["sectors"] or ticker_id == "1640718":
            continue
        ticker_name, hist = get_data(
            ticker_id,
            ticker_info,
            manager,
            avanza,
            start_date=args.start,
            end_date=args.end,
        )
        if hist is not None:
            save_snapshot(
                hist,
                f"data/{ticker_name}_{args.start}_{args.end}.csv",
                asof=args.end or date.today(),
            )

    calculate_score(manager)

    manager._display()

    return 0


if __name__ == "__main__":

    result = main()
