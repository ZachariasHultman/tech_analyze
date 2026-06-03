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
from analyzer.helper import *
from analyzer.summary_manager import SummaryManager
from analyzer.data_processing import *
from importlib.metadata import version

from analyzer.historical_calc import calculate_metrics_given_hist
from analyzer.correlation import baseline_correlation, optimize_weights_and_thresholds, optimize_combo, optimize_stepwise
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


def _load_reliability_map():
    """Load company reliability scores from company_reliability.csv."""
    rel_path = os.path.join(project_root, "company_reliability.csv")
    reliability = {}
    reliability_val=0.4
    if os.path.exists(rel_path):
        try:
            rel_df = pd.read_csv(rel_path)
            for _, r in rel_df.iterrows():
                reliability[r["company"]] = {
                    "spearman": r["spearman"],
                    "reliable": r.get("reliable", r["spearman"] > reliability_val),
                }
        except Exception:
            pass
    return reliability


def _update_watchlist(avanza, manager, top_n=10):
    """Add top-scoring stocks with good reliability to 'Bör köpa' watchlist.

    - Filters by both score (points) and reliability (spearman > 0.4)
    - Adds qualified stocks that aren't already on the list
    - Removes stocks from the list that no longer qualify
    """
    watchlists = avanza.get_watchlists()

    def _wl_attr(wl, key):
        """Get attribute from watchlist (supports both dict and pydantic model)."""
        if isinstance(wl, dict):
            return wl.get(key)
        return getattr(wl, key, None)

    target = next(
        (wl for wl in watchlists if _wl_attr(wl, "name") == "Bör köpa"), None
    )

    if target is None:
        print("\n[WARN] Watchlist 'Bör köpa' not found on Avanza.")
        print("  Please create it manually in Avanza first, then re-run.")
        return

    watchlist_id = _wl_attr(target, "watchListId") or _wl_attr(target, "id")
    existing_ids = set(str(oid) for oid in (_wl_attr(target, "orderbookIds") or []))

    # Load reliability data
    reliability = _load_reliability_map()

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

    # Add reliability info and filter
    combined["_spearman"] = combined.index.map(
        lambda c: reliability.get(c, {}).get("spearman", float("nan"))
    )
    combined["_reliable"] = combined.index.map(
        lambda c: reliability.get(c, {}).get("reliable", False)
    )

    # Only keep stocks that are both high-scoring and reliable
    qualified = combined[combined["_reliable"]].copy()
    qualified = qualified.sort_values("_pts", ascending=False).head(top_n)

    # Build set of orderbook IDs that should be on the list
    qualified_ids = set()
    for idx in qualified.index:
        oid = _extract_orderbook_id(idx)
        if oid:
            qualified_ids.add(oid)

    added = []
    already = []
    failed = []
    removed = []
    skipped_unreliable = []

    # Report stocks that were filtered out due to bad reliability
    unreliable = combined[~combined["_reliable"]].sort_values("_pts", ascending=False)
    for idx in unreliable.head(5).index:
        pts = unreliable.loc[idx, "_pts"]
        sp = unreliable.loc[idx, "_spearman"]
        if pd.notna(pts) and pts > 0:
            skipped_unreliable.append((idx, pts, sp))

    # Add qualified stocks not yet on list
    for idx in qualified.index:
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

    # Remove stocks from watchlist that no longer qualify
    # Build a reverse map: orderbookId -> company name (from combined)
    id_to_name = {}
    for idx in combined.index:
        oid = _extract_orderbook_id(idx)
        if oid:
            id_to_name[oid] = idx

    for oid in existing_ids:
        if oid not in qualified_ids:
            name = id_to_name.get(oid, f"Unknown ({oid})")
            try:
                avanza.remove_from_watchlist(oid, watchlist_id)
                removed.append(name)
            except Exception as e:
                failed.append((name, f"remove failed: {e}"))

    # Report
    print(f"\n{'=' * 70}")
    print(f"  WATCHLIST UPDATE: 'Bör köpa' (top {top_n}, reliable only)")
    print(f"{'=' * 70}")
    if added:
        print(f"\n  Added {len(added)} stock(s):")
        for name in added:
            pts = qualified.loc[name, "_pts"]
            sp = qualified.loc[name, "_spearman"]
            print(f"    + {name}  ({pts:+.2f} pts, r={sp:.2f})")
    if already:
        print(f"\n  Already on list ({len(already)}):")
        for name in already:
            pts = qualified.loc[name, "_pts"]
            sp = qualified.loc[name, "_spearman"]
            print(f"    = {name}  ({pts:+.2f} pts, r={sp:.2f})")
    if removed:
        print(f"\n  Removed {len(removed)} stock(s) (no longer qualify):")
        for name in removed:
            print(f"    - {name}")
    if skipped_unreliable:
        print(f"\n  Skipped (high score but unreliable):")
        for name, pts, sp in skipped_unreliable:
            sp_str = f"{sp:.2f}" if pd.notna(sp) else "N/A"
            print(f"    ~ {name}  ({pts:+.2f} pts, r={sp_str})")
    if failed:
        print(f"\n  Failed ({len(failed)}):")
        for name, err in failed:
            print(f"    ! {name}: {err}")
    print(f"{'=' * 70}\n")


# OMXS30 and OMXS Mid Cap tickers as of 2024-2025.
# These are searched by name via search_for_stock so they survive minor name changes.
_PRESETS: dict[str, list[str]] = {
    "omxs30": [
        "ABB", "Alfa Laval", "Assa Abloy B", "AstraZeneca",
        "Atlas Copco A", "Atlas Copco B", "Autoliv",
        "Boliden", "Electrolux B", "Ericsson B", "Essity B",
        "Evolution", "Getinge B", "Hexagon B", "H&M B",
        "Investor B", "Kinnevik B", "Nordea Bank",
        "Sandvik", "SEB A", "SCA B", "SKF B", "SSAB A",
        "Swedbank A", "Tele2 B", "Telia", "Volvo B",
        "Husqvarna B", "Latour B", "Nibe B",
    ],
    "omxs-mid": [
        "Addtech B", "Avanza Bank", "Bilia A", "BioGaia B",
        "Bufab", "Catena", "Clas Ohlson B", "Dustin",
        "Fabege", "Hexpol B", "Hufvudstaden A",
        "Indutrade", "Intrum", "JM", "Kungsleden",
        "Lifco B", "Lindab", "Nobia", "OEM International B",
        "Peab B", "Ratos B", "Skistar B", "Sweco B",
        "Troax", "Veidekke", "Vitec Software B",
    ],
}


def _search_preset(avanza, preset_name: str) -> tuple[set[str], str]:
    """Look up preset stock names via search_for_stock. Returns (id_set, label)."""
    names = _PRESETS.get(preset_name.lower())
    if names is None:
        available = ", ".join(_PRESETS)
        print(f"[WARN] Unknown preset '{preset_name}'. Available: {available}")
        return set(), ""

    from avanza.avanza import InstrumentType
    found: set[str] = set()
    not_found: list[str] = []

    print(f"  Searching {len(names)} stocks for preset '{preset_name}'...")
    for name in names:
        try:
            hits = avanza.search_for_stock(name, limit=3)
            if not hits:
                not_found.append(name)
                continue
            # Take the first Swedish (SE) hit, fall back to first hit overall
            hit = next(
                (h for h in hits
                 if (h.get("flagCode") or getattr(h, "flagCode", None) or "").upper() == "SE"),
                hits[0],
            )
            hit_id = hit.get("id") if isinstance(hit, dict) else getattr(hit, "id", None)
            if hit_id:
                found.add(str(hit_id))
            else:
                not_found.append(name)
        except Exception as e:
            not_found.append(f"{name} ({e})")

    if not_found:
        print(f"  [WARN] Could not find: {', '.join(not_found)}")

    label = f"preset:{preset_name} ({len(found)} stocks)"
    return found, label


def main():
    pd.set_option("display.max_rows", None)  # Show all rows
    pd.set_option("display.max_colwidth", None)  # Show full cell content
    print("Avanza API version: ", version("avanza-api"))
    print("yfinance version: ", version("yfinance"))
    ap = argparse.ArgumentParser(
        description="Stock scoring & analysis tool",
        epilog="""
--- Stock universe (pick one or combine) ---
  python main.py                               Default: uses watchlist named "Test"
  python main.py --watchlists Test Utdelare    Use one or more personal watchlists
  python main.py --preset omxs30               Use built-in OMXS30 preset (~30 stocks)
  python main.py --preset omxs30 omxs-mid      Combine presets (~55 stocks)

--- Data & analysis ---
  python main.py --save                        Save today's snapshot to data/
  python main.py --correlate                   Show correlation of scores with past returns
  python main.py --correlate --optimize        Re-optimize metric weights (do occasionally)

--- Watchlist management ---
  python main.py --watchlist                   Push top 10 to 'Bör köpa' on Avanza
  python main.py --watchlist --watchlist-top 5

--- Weight variants ---
  python main.py --no-opt                      Ignore saved weights, use hardcoded defaults
  python main.py --use-combo                   Use combo-optimized weights
  python main.py --use-stepwise                Use stepwise-optimized weights
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
        "--watchlists",
        nargs="+",
        default=["Test"],
        metavar="NAME",
        help='Names of Avanza personal watchlists to analyze (default: "Test"). '
             'Use --discover to see Avanza inspiration lists instead.',
    )
    ap.add_argument(
        "--preset",
        nargs="+",
        default=[],
        metavar="NAME",
        help=f"Built-in stock presets to include. Available: {', '.join(_PRESETS)}. "
             "Stocks are looked up via search, so no hardcoded IDs needed. "
             "Can be combined with --watchlists.",
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

    # Collect tickers from personal watchlists
    ticker_id_set: set[str] = set()
    sources: list[str] = []

    all_watchlists = avanza.get_watchlists()
    missing_wls = []
    for wl_name in args.watchlists:
        wl = next((w for w in all_watchlists if w.get("name") == wl_name), None)
        if wl is None:
            missing_wls.append(wl_name)
            continue
        ids = [str(oid) for oid in (wl.get("orderbookIds") or [])]
        ticker_id_set.update(ids)
        sources.append(f"{wl_name} ({len(ids)} stocks)")
    if missing_wls:
        print(f"[WARN] Watchlist(s) not found on Avanza: {', '.join(missing_wls)}")

    # Add tickers from built-in presets (search-based, no hardcoded IDs)
    for preset_name in args.preset:
        preset_ids, label = _search_preset(avanza, preset_name)
        ticker_id_set.update(preset_ids)
        if label:
            sources.append(label)

    if not ticker_id_set:
        print("[ERROR] No stocks found. Check --watchlists / --inspiration names, or run --discover.")
        return 1

    ticker_ids = list(ticker_id_set)
    print(f"Analyzing {len(ticker_ids)} unique tickers from: {', '.join(sources)}")

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
