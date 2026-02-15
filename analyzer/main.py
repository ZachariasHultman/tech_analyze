import sys
import os
import yfinance as yf
from tqdm import tqdm
import re


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

from historical_calc import calculate_metrics_given_hist
from correlation import baseline_correlation, optimize_weights_and_thresholds, optimize_combo, optimize_stepwise
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


def _load_optimized_params(variant=None):
    """Load optimized weights and thresholds from the appropriate results file.

    variant: None (default/legacy), "individual", "combo", or "stepwise"
    Returns (weights_dict, thresholds_dict) — either may be None.
    """
    import json
    if variant == "individual":
        filename = "optimization_results_individual.json"
    elif variant == "combo":
        filename = "optimization_results_combo.json"
    elif variant == "stepwise":
        filename = "optimization_results_stepwise.json"
    else:
        # Legacy fallback: try individual first, then old name
        filename = "optimization_results_individual.json"
        path = os.path.join(project_root, filename)
        if not os.path.exists(path):
            filename = "optimization_results.json"

    weights_path = os.path.join(project_root, filename)
    if not os.path.exists(weights_path):
        return None, None
    try:
        with open(weights_path) as f:
            data = json.load(f)
        weights = data.get("optimized_weights")
        thresholds = data.get("optimized_thresholds")
        if weights:
            print(f"Loaded optimized params from {weights_path}")
            return weights, thresholds
    except Exception as e:
        print(f"Warning: could not load optimized params: {e}")
    return None, None


def main():
    pd.set_option("display.max_rows", None)  # Show all rows
    pd.set_option("display.max_colwidth", None)  # Show full cell content
    print("Avanza API version: ", version("avanza-api"))
    print("yfinance version: ", version("yfinance"))
    ap = argparse.ArgumentParser(
        description="Stock scoring & analysis tool",
        epilog="""
Usage examples:
  python main.py                    Run live analysis (uses optimized weights if available)
  python main.py --save             Run live analysis AND save data snapshots to data/
  python main.py --correlate        Show per-metric correlation with stock returns
  python main.py --optimize-individual Optimize weights (independent correlation)
  python main.py --optimize-combo     Optimize weights (grid sweep + cross-validation)
  python main.py --optimize-stepwise  Optimize weights (scipy numerical + cross-validation)
  python main.py --no-opt             Run live analysis with default (hardcoded) weights
  python main.py --use-individual     Use individual optimization results for live analysis
  python main.py --use-combo          Use combo optimization results for live analysis
  python main.py --use-stepwise       Use stepwise optimization results for live analysis
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--save",
        action="store_true",
        help="Save data snapshots to data/ (during live run)",
    )
    ap.add_argument(
        "--get_hist",
        action="store_true",
        help="(deprecated, same as --save)",
    )
    ap.add_argument(
        "--correlate",
        action="store_true",
        help="Show per-metric Spearman correlation with forward returns",
    )
    ap.add_argument(
        "--optimize", "--optimize-individual",
        action="store_true",
        dest="optimize_individual",
        help="Optimize metric weights based on individual correlation (saves optimization_results_individual.json)",
    )
    ap.add_argument(
        "--optimize-combo",
        action="store_true",
        help="Optimize with grid sweep + cross-validation (saves optimization_results_combo.json)",
    )
    ap.add_argument(
        "--optimize-stepwise",
        action="store_true",
        help="Optimize with scipy numerical + cross-validation (saves optimization_results_stepwise.json)",
    )
    ap.add_argument(
        "--no-opt",
        action="store_true",
        help="Ignore optimized weights, use hardcoded defaults",
    )
    ap.add_argument(
        "--use-individual",
        action="store_true",
        help="Use individual optimization results for live analysis",
    )
    ap.add_argument(
        "--use-combo",
        action="store_true",
        help="Use combo optimization results for live analysis",
    )
    ap.add_argument(
        "--use-stepwise",
        action="store_true",
        help="Use stepwise optimization results for live analysis",
    )
    args = ap.parse_args()
    os.makedirs("data", exist_ok=True)

    save_data = args.save or args.get_hist

    # --correlate and --optimize*: use saved historical data
    if args.correlate or args.optimize_individual or args.optimize_combo or args.optimize_stepwise:
        calculate_metrics_given_hist()
        baseline_correlation("metrics_by_timespan.csv")
        if args.optimize_individual:
            optimize_weights_and_thresholds("metrics_by_timespan.csv")
        if args.optimize_combo:
            optimize_combo("metrics_by_timespan.csv")
        if args.optimize_stepwise:
            optimize_stepwise("metrics_by_timespan.csv")
        return 0

    # --- Live analysis ---
    manager = SummaryManager()

    # Load optimized weights and thresholds unless --no-opt
    if not args.no_opt:
        if args.use_individual:
            opt_weights, opt_thresholds = _load_optimized_params("individual")
        elif args.use_combo:
            opt_weights, opt_thresholds = _load_optimized_params("combo")
        elif args.use_stepwise:
            opt_weights, opt_thresholds = _load_optimized_params("stepwise")
        else:
            opt_weights, opt_thresholds = _load_optimized_params()
        if opt_weights:
            manager._weight_overrides = opt_weights
        if opt_thresholds:
            manager._threshold_overrides = opt_thresholds

    avanza = setup_env()
    ticker_ids = next(
        (
            item
            for item in avanza.get_watchlists()
            if item.get("name")
            == "Test"  # "Utdelare"  # "Test"  # "Mina favoritaktier"  # "Berkshire"   # "Mina favoritaktier"  # "Äger"
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
            get_hist=save_data,
        )
        if save_data:
            save_snapshot(
                hist,
                f"data/{ticker_name}_{date.today()}.csv",
                asof=date.today(),
            )

    calculate_score(manager)

    manager._display(save_df=True)

    return 0


if __name__ == "__main__":

    result = main()
