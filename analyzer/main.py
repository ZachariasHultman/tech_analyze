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


def _extract_orderbook_id(index_name):
    """Extract the orderbookId from an index like 'Glencore plc 2165695'."""
    parts = str(index_name).rsplit(" ", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[1]
    return None


def _update_watchlist(avanza, manager, top_n=10):
    """Add top-scoring stocks to the 'Bör köpa' watchlist.

    - Gets or identifies the watchlist
    - Extracts top N stocks by points from both summary tables
    - Skips stocks already on the list
    - Adds new ones
    """
    watchlists = avanza.get_watchlists()
    target = next(
        (wl for wl in watchlists if wl.get("name") == "Bör köpa"), None
    )

    if target is None:
        print("\n[WARN] Watchlist 'Bör köpa' not found on Avanza.")
        print("  Please create it manually in Avanza first, then re-run.")
        return

    watchlist_id = target["id"]
    existing_ids = set(str(oid) for oid in target.get("orderbookIds", []))

    # Collect all scored stocks from both summaries
    frames = []
    for summary in [manager.summary, manager.summary_investment]:
        if summary is not None and isinstance(summary, pd.DataFrame) and not summary.empty:
            if "points" in summary.columns:
                frames.append(summary)

    if not frames:
        print("\n[WARN] No scored stocks to add to watchlist.")
        return

    combined = pd.concat(frames)
    combined["_pts"] = pd.to_numeric(combined["points"], errors="coerce")
    combined = combined.sort_values("_pts", ascending=False).head(top_n)

    added = []
    already = []
    failed = []

    for idx in combined.index:
        orderbook_id = _extract_orderbook_id(idx)
        if not orderbook_id:
            failed.append((idx, "could not extract orderbookId"))
            continue

        if orderbook_id in existing_ids:
            already.append(idx)
            continue

        try:
            avanza.add_to_watchlist(orderbook_id, watchlist_id)
            added.append(idx)
        except Exception as e:
            failed.append((idx, str(e)))

    # Report
    print(f"\n{'=' * 70}")
    print(f"  WATCHLIST UPDATE: 'Bör köpa' (top {top_n})")
    print(f"{'=' * 70}")
    if added:
        print(f"\n  Added {len(added)} stock(s):")
        for name in added:
            pts = combined.loc[name, "_pts"]
            print(f"    + {name}  ({pts:+.2f} pts)")
    if already:
        print(f"\n  Already on list ({len(already)}):")
        for name in already:
            pts = combined.loc[name, "_pts"]
            print(f"    = {name}  ({pts:+.2f} pts)")
    if failed:
        print(f"\n  Failed ({len(failed)}):")
        for name, err in failed:
            print(f"    ! {name}: {err}")
    print(f"{'=' * 70}\n")


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
  python main.py --watchlist          Add top 10 stocks to 'Bör köpa' watchlist
  python main.py --watchlist --watchlist-top 5   Add top 5 instead
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
    ap.add_argument(
        "--watchlist",
        action="store_true",
        help="Add top-scoring stocks to 'Bör köpa' watchlist on Avanza",
    )
    ap.add_argument(
        "--watchlist-top",
        type=int,
        default=10,
        help="Number of top stocks to add to watchlist (default: 10)",
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

    skipped = []
    for ticker_id in tqdm(ticker_ids, desc="Processing tickers"):
        try:
            ticker_info = avanza.get_stock_info(ticker_id)
        except Exception as e:
            skipped.append((ticker_id, str(e)))
            continue

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

    if skipped:
        print(f"\n[WARN] Skipped {len(skipped)} ticker(s) due to API errors:")
        for tid, err in skipped:
            print(f"  ID {tid}: {err}")
            print(f"    → https://www.avanza.se/aktier/om-aktien.html/{tid}/")
        print()

    calculate_score(manager)

    manager._display(save_df=True)

    if args.watchlist:
        _update_watchlist(avanza, manager, top_n=args.watchlist_top)

    return 0


if __name__ == "__main__":

    result = main()
