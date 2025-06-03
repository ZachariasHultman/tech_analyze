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

    get_data(ticker_ids, manager, avanza)

    calculate_score(manager)

    manager._display()

    return 0


if __name__ == "__main__":

    result = main()
