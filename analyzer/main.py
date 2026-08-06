import sys
import os
import yfinance as yf
from tqdm import tqdm
import re


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _load_dotenv():
    """Load .env from project root into environment, without overriding existing vars.
    Handles both KEY=VALUE and export KEY=VALUE formats.
    """
    env_path = os.path.join(project_root, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.removeprefix("export").strip()
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


_load_dotenv()

from avanza.avanza import Avanza
from avanza.models import *
import os
import pandas as pd
from analyzer.helper import *
from analyzer.summary_manager import SummaryManager
from analyzer.data_processing import *
from importlib.metadata import version

from analyzer.historical_calc import calculate_metrics_given_hist
from analyzer.correlation import baseline_correlation, optimize_weights_and_thresholds, optimize_combo
from analyzer.config import (
    RELIABILITY_DEFAULT_CUTOFF,
    RELIABILITY_ESTABLISHED,
    RELIABILITY_INVERSE,
    SLEEVE_GATE_MIN,
    EXCLUDED_TICKER_IDS,
)
from datetime import date
import argparse


def setup_env():
    # AVANZA_* preferred; falls back to the legacy USERNAME/PASSWORD/MY_TOTP_SECRET
    # names for older .env files.
    username = os.getenv("AVANZA_USERNAME") or os.getenv("USERNAME")
    if username is None:
        raise Exception("Expected .env file to have a key named AVANZA_USERNAME (or legacy USERNAME)")

    password = os.getenv("AVANZA_PASSWORD") or os.getenv("PASSWORD")
    if password is None:
        raise Exception("Expected .env file to have a key named AVANZA_PASSWORD (or legacy PASSWORD)")

    totpSecret = os.getenv("AVANZA_TOTP_SECRET") or os.getenv("MY_TOTP_SECRET")
    if totpSecret is None:
        raise Exception("Expected .env file to have a key named AVANZA_TOTP_SECRET (or legacy MY_TOTP_SECRET)")
    # totp = pyotp.TOTP(totpSecret, digest=hashlib.sha1)
    # print(totpSecret)
    # print(totp.now())

    avanza_user = Avanza(
        {"username": username, "password": password, "totpSecret": totpSecret}
    )
    return avanza_user


def _load_optimized_params(variant=None):
    """Load optimized weights and thresholds from the appropriate results file.

    variant: None (default/legacy), "individual", or "combo"
    Returns (weights_dict, thresholds_dict) — either may be None.
    """
    import json
    if variant == "individual":
        filename = "optimization_results_individual.json"
    elif variant == "combo":
        filename = "optimization_results_combo.json"
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


def to_yahoo_symbol(ticker_info) -> str | None:
    """Map an Avanza ticker_info dict to a Yahoo Finance ticker symbol.

    Returns None for unrecognized country codes, or when a DE ticker's
    symbol doesn't start with a recognizable letter prefix — callers should
    skip yfinance-dependent metrics (FCFY) for that stock rather than crash.

    US is a no-suffix passthrough: Avanza's raw tickerSymbol for
    NYSE/NASDAQ-listed stocks already matches the bare Yahoo symbol
    (verified against live data, e.g. "GM", "KR", "DIS") — this mirrors the
    pre-fix fallthrough behavior, which happened to be correct for US.
    Other unmapped countries (e.g. FI) are NOT a safe passthrough — Yahoo
    requires an exchange suffix there (e.g. "UPM" resolves nothing, only
    "UPM.HE" does) — so they still return None rather than guessing.
    """
    listing = ticker_info.get("listing", {}) if isinstance(ticker_info, dict) else {}
    symbol = listing.get("tickerSymbol")
    country = listing.get("countryCode")
    if not symbol or not country:
        return None

    if country == "SE":
        return symbol.replace(" ", "-") + ".ST"
    if country == "DK":
        return symbol.replace(" ", "-") + ".CO"
    if country == "NO":
        return symbol.replace(" ", "-") + ".OL"
    if country == "DE":
        m = re.match(r"^[A-Z]+", symbol)
        return m.group() + ".DE" if m else None
    if country == "US":
        return symbol
    return None


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
    if os.path.exists(rel_path):
        try:
            rel_df = pd.read_csv(rel_path)
            for _, r in rel_df.iterrows():
                reliability[r["company"]] = {
                    "spearman": r["spearman"],
                    "spearman_shrunk": r.get("spearman_shrunk", r["spearman"]),
                    "n_windows": r.get("n_windows", float("nan")),
                    "reliable": r.get("reliable", r["spearman"] > RELIABILITY_DEFAULT_CUTOFF),
                }
        except Exception:
            pass
    return reliability


def _fmt_reliability(sp, n):
    """Format a shrunk reliability figure with its sample size, e.g. '+0.60 (n=5)'."""
    if pd.isna(sp):
        return "N/A"
    n_str = f"{int(n)}" if pd.notna(n) else "?"
    return f"{sp:+.2f} (n={n_str})"


def _fmt_scored_row(prefix, r):
    """Format a (name, pts, shrunk_spearman, rank_key, quality_pct, value_pct,
    combined_score, n_windows) row -- shared by the terminal watchlist report
    and the email summary so both stay consistent. Falls back to a plain
    name when the stock wasn't part of this run's scored universe (e.g. a
    watchlist holding outside the current --preset/--watchlists scope)."""
    name, pts, sp, comb, qual, val, cscore, n = r
    if pd.isna(qual) and pd.isna(val) and pd.isna(cscore):
        return f"{prefix} {name}  (not scored this run)"
    return (f"{prefix} {name}  (q={qual:.2f}, v={val:.2f}, "
            f"combined={cscore:.2f}, r={_fmt_reliability(sp, n)})")


# Shared explainer for q/v/combined/r, printed in both the terminal watchlist
# report and the email summary.
LEGEND_LINES = [
    "q = quality percentile (0-1, peer-ranked business health)",
    "v = value percentile (0-1, peer-ranked cheapness)",
    "combined = q x v -- the actual ranking key",
    "r = shrunk reliability -- does this company's own past score history",
    "    actually track its own returns? (n = independent windows behind",
    "    it; positive = trustworthy signal for this stock, negative =",
    "    fundamentals have historically moved opposite to its returns,",
    "    near zero = no track record either way)",
]


def _wl_attr(wl, key):
    """Get attribute from a watchlist (supports both dict and pydantic model)."""
    if isinstance(wl, dict):
        return wl.get(key)
    return getattr(wl, key, None)


def _update_watchlist(avanza, manager, top_n=10, target_name="Bör köpa"):
    """Add top-scoring stocks with good reliability to the target watchlist.

    - Filters by both score (points) and reliability (spearman > 0.4)
    - Adds qualified stocks that aren't already on the list
    - Removes stocks from the list that no longer qualify
    """
    watchlists = avanza.get_watchlists()

    target = next(
        (wl for wl in watchlists if _wl_attr(wl, "name") == target_name), None
    )

    if target is None:
        print(f"\n[WARN] Watchlist '{target_name}' not found on Avanza.")
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

    # Add reliability info. Display the shrunk figure (matches what actually
    # drives _combined below), not the raw spearman -- a raw correlation
    # from only 5 non-overlapping windows is too noisy to show unshrunk.
    combined["_shrunk"] = combined.index.map(
        lambda c: reliability.get(c, {}).get("spearman_shrunk", float("nan"))
    )
    combined["_n_windows"] = combined.index.map(
        lambda c: reliability.get(c, {}).get("n_windows", float("nan"))
    )

    # Combined rank: combined_score scaled up by shrunk reliability. Reliability
    # tilts the ranking but never zeroes a stock out (factor >= 1 when shrunk
    # is missing or negative-but-clipped is not applied here — fillna(0) only).
    combined["_combined_score"] = pd.to_numeric(
        combined.get("combined_score"), errors="coerce"
    )
    combined["_combined"] = combined["_combined_score"] * (
        1 + combined["_shrunk"].fillna(0)
    )

    # Two-sleeve gate: qualify only stocks ranking well in BOTH quality and value.
    combined["_quality"] = pd.to_numeric(combined.get("quality_pct"), errors="coerce")
    combined["_value"] = pd.to_numeric(combined.get("value_pct"), errors="coerce")
    qualified = combined[
        (combined["_quality"] >= SLEEVE_GATE_MIN)
        & (combined["_value"] >= SLEEVE_GATE_MIN)
    ].copy()
    qualified = qualified.sort_values("_combined", ascending=False).head(top_n)

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

    def _num(name, col, df=None):
        df = qualified if df is None else df
        try:
            v = df.loc[name, col]
            return float(v) if pd.notna(v) else float("nan")
        except Exception:
            return float("nan")

    def _row(name, df=None):
        # (name, pts, shrunk_spearman, rank_key, quality_pct, value_pct, combined_score, n_windows)
        df = qualified if df is None else df
        return (
            name,
            _num(name, "_pts", df),
            _num(name, "_shrunk", df),
            _num(name, "_combined", df),
            _num(name, "_quality", df),
            _num(name, "_value", df),
            _num(name, "_combined_score", df),
            _num(name, "_n_windows", df),
        )

    added_rows   = [_row(n) for n in added]
    already_rows = [_row(n) for n in already]
    # removed stocks are excluded from `qualified` by definition (that's why
    # they were removed) -- look their metrics up in `combined` instead,
    # which still has every scored company's row.
    removed_rows = [_row(n, df=combined) for n in removed]

    def _fmt_row(marker, r):
        return f"    {_fmt_scored_row(marker, r)}"

    # Report
    print(f"\n{'=' * 70}")
    print(f"  WATCHLIST UPDATE: '{target_name}' (top {top_n} by combined × reliability)")
    print(f"{'=' * 70}")
    for line in LEGEND_LINES:
        print(f"  {line}")
    if added_rows:
        print(f"\n  Added {len(added_rows)} stock(s):")
        for r in added_rows:
            print(_fmt_row("+", r))
    if already_rows:
        print(f"\n  Already on list ({len(already_rows)}):")
        for r in already_rows:
            print(_fmt_row("=", r))
    if removed_rows:
        print(f"\n  Removed {len(removed_rows)} stock(s) (no longer in top {top_n}):")
        for r in removed_rows:
            print(_fmt_row("-", r))
    if failed:
        print(f"\n  Failed ({len(failed)}):")
        for name, err in failed:
            print(f"    ! {name}: {err}")
    print(f"{'=' * 70}\n")

    return {
        "target_name": target_name,
        "top_n": top_n,
        "added": added_rows,
        "already": already_rows,
        "removed": removed_rows,
        "failed": failed,
    }


def _compute_sell_signals(avanza, manager, portfolio_watchlist: str) -> list[dict]:
    """Flag stocks in portfolio_watchlist that have deteriorating fundamentals.

    A stock is flagged when its combined score (pts × reliability) is below 1.5,
    or when its fundamental score is negative and reliability is established.
    Returns a list of dicts sorted worst-first.
    """
    reliability = _load_reliability_map()

    all_wls = avanza.get_watchlists()
    wl = next((w for w in all_wls if _wl_attr(w, "name") == portfolio_watchlist), None)
    if wl is None:
        print(f"[WARN] Watchlist '{portfolio_watchlist}' not found — skipping sell check.")
        return []

    portfolio_ids = set(str(oid) for oid in (_wl_attr(wl, "orderbookIds") or []))

    frames = []
    for summary in [manager.summary, manager.summary_investment]:
        if summary is not None and isinstance(summary, pd.DataFrame) and not summary.empty:
            frames.append(summary)
    if not frames:
        return []

    combined = pd.concat([f.dropna(axis=1, how="all") for f in frames])
    combined["_pts"]      = pd.to_numeric(combined["points"], errors="coerce")
    # Use the shrunk reliability figure (small-sample-adjusted) for the
    # established-relationship comparisons below.
    combined["_spearman"] = combined.index.map(lambda c: reliability.get(c, {}).get("spearman_shrunk", float("nan")))
    combined["_n_windows"] = combined.index.map(lambda c: reliability.get(c, {}).get("n_windows", float("nan")))
    combined["_combined"] = combined["_pts"] * combined["_spearman"].clip(lower=0)

    scored_ids = {_extract_orderbook_id(i) for i in combined.index}
    for missing_id in sorted(portfolio_ids - scored_ids):
        try:
            name = avanza.get_stock_info(missing_id).get("name", missing_id)
        except Exception:
            name = missing_id
        print(f"[WARN] {name} ({missing_id}) not analyzed — no signal possible.")

    signals = []
    for idx in combined.index:
        oid = _extract_orderbook_id(idx)
        if oid not in portfolio_ids:
            continue
        pts  = float(combined.loc[idx, "_pts"])   if pd.notna(combined.loc[idx, "_pts"])      else None
        sp   = float(combined.loc[idx, "_spearman"]) if pd.notna(combined.loc[idx, "_spearman"]) else None
        n_w  = float(combined.loc[idx, "_n_windows"]) if pd.notna(combined.loc[idx, "_n_windows"]) else None
        comb = float(combined.loc[idx, "_combined"]) if pd.notna(combined.loc[idx, "_combined"]) else None

        reasons = []
        # Only flag genuine deterioration: negative score where the relationship is known
        if pts is not None and pts < 0 and sp is not None and sp > RELIABILITY_ESTABLISHED:
            reasons.append("fundamentals have deteriorated")
        # Flag stocks where the score actively predicts the wrong direction
        if sp is not None and sp < RELIABILITY_INVERSE:
            reasons.append("score moves opposite to returns for this stock")

        if reasons:
            signals.append({"name": idx, "pts": pts, "spearman": sp, "n_windows": n_w, "combined": comb, "reasons": ", ".join(reasons)})

    return sorted(signals, key=lambda x: (x["combined"] or 0))


def _send_email(push_results: dict | None, sell_signals: list[dict]) -> None:
    """Send a plain-text email summary via Gmail SMTP (STARTTLS, port 587).

    Requires in .env (same vars as stryket):
      SMTP_USER=you@gmail.com
      SMTP_PASSWORD=xxxx        (Gmail app password)
      EMAIL_TO=you@gmail.com
      EMAIL_FROM=you@gmail.com  (optional, defaults to SMTP_USER)
    """
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from datetime import date

    smtp_user     = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    email_to      = os.getenv("EMAIL_TO")
    email_from    = os.getenv("EMAIL_FROM") or smtp_user

    if not smtp_user or not smtp_password or not email_to:
        print("[WARN] Email not sent — SMTP_USER, SMTP_PASSWORD, EMAIL_TO must be set in environment")
        return

    lines = [f"Stock Screener — {date.today()}", ""]

    if push_results or sell_signals:
        lines.extend(LEGEND_LINES)
        lines.append("")

    if push_results:
        n = push_results["top_n"]
        lines.append("=" * 50)
        lines.append(f"  CONSIDER BUYING (top {n})")
        lines.append("=" * 50)

        top10 = push_results["added"] + push_results["already"]
        top10.sort(key=lambda x: x[3], reverse=True)  # sort by rank key (combined × reliability)
        added_set = {r[0] for r in push_results["added"]}
        for r in top10:
            tag = "NEW" if r[0] in added_set else "   "
            lines.append(f"  {_fmt_scored_row(tag, r)}")

        if push_results["removed"]:
            lines.append(f"\nRemoved from list ({len(push_results['removed'])}):")
            for r in push_results["removed"]:
                lines.append(f"  {_fmt_scored_row('-', r)}")

    if sell_signals:
        lines.append("\n" + "=" * 50)
        lines.append("  CONSIDER SELLING")
        lines.append("=" * 50)
        for sig in sell_signals:
            pts_str = f"{sig['pts']:+.2f}" if sig["pts"] is not None else "N/A"
            sp_str  = _fmt_reliability(sig["spearman"], sig.get("n_windows"))
            lines.append(f"  ! {sig['name']}  ({pts_str} pts, r={sp_str}) — {sig['reasons']}")
    else:
        lines.append("\n✓ No sell signals in portfolio.")

    body = "\n".join(lines)
    msg = MIMEMultipart()
    msg["From"]    = email_from
    msg["To"]      = email_to
    msg["Subject"] = f"Stock Update {date.today()}"
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(email_from, [email_to], msg.as_string())
        print(f"Email sent to {email_to}")
    except Exception as e:
        print(f"[WARN] Failed to send email: {e}")


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
        "Indutrade", "Intrum", "JM",
        "Lifco B", "Lindab", "Nobia", "OEM International B",
        "Peab B", "Ratos B", "Skistar B", "Sweco B",
        "Troax", "Veidekke", "Vitec Software B",
    ],
}

# Applied only when BOTH --preset and --watchlists are omitted entirely --
# matches the universe cron_pi.sh already runs weekly, so a bare invocation
# gives the same "everything" scope instead of quietly falling back to just
# the "Test" watchlist. Giving either flag explicitly overrides this and
# uses exactly what was passed, with no default mixed in.
_DEFAULT_PRESETS = ["omxs30", "omxs-mid"]
_DEFAULT_WATCHLISTS = ["Test", "Utdelare", "Äger", "Berkshire"]


def _resolve_universe(preset_arg, watchlists_arg):
    """Resolve --preset/--watchlists args to (preset_names, watchlist_names).

    Falls back to the standard default universe only when BOTH were omitted
    entirely (None). Giving either flag explicitly uses exactly what was
    passed, with no default mixed in.
    """
    if preset_arg is None and watchlists_arg is None:
        return _DEFAULT_PRESETS, _DEFAULT_WATCHLISTS
    return preset_arg or [], watchlists_arg or []


def _search_preset(avanza, preset_name: str) -> tuple[set[str], str]:
    """Look up preset stock names via search_for_stock. Returns (id_set, label)."""
    names = _PRESETS.get(preset_name.lower())
    if names is None:
        available = ", ".join(_PRESETS)
        print(f"[WARN] Unknown preset '{preset_name}'. Available: {available}")
        return set(), ""

    found: set[str] = set()
    not_found: list[str] = []

    def _get(obj, key):
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

    print(f"  Searching {len(names)} stocks for preset '{preset_name}'...")
    for name in names:
        try:
            hits = avanza.search_for_stock(name, limit=5)
            if not hits:
                not_found.append(name)
                continue
            # Prefer first Swedish hit (flagCode=SE), fall back to first result
            hit = next(
                (h for h in hits if (_get(h, "flagCode") or "").upper() == "SE"),
                hits[0],
            )
            hit_id = _get(hit, "orderBookId")
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

--- Push results to Avanza ---
  python main.py --push                        Push top 10 to 'Bör köpa' (default)
  python main.py --push --push-to "Min lista"  Push to a different watchlist
  python main.py --push --push-top 5           Push top 5 instead of 10

--- Weight variants ---
  python main.py --no-opt                      Ignore saved weights, use hardcoded defaults
  python main.py --use-combo                   Use combo-optimized weights
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
        "--watchlists", "--watchlist",
        nargs="+",
        default=None,
        metavar="NAME",
        dest="watchlists",
        help='Names of Avanza personal watchlists to read stocks from. '
             f'Defaults to {_DEFAULT_WATCHLISTS} when neither this nor '
             '--preset is given.',
    )
    ap.add_argument(
        "--preset",
        nargs="+",
        default=None,
        metavar="NAME",
        help=f"Built-in stock presets to include. Available: {', '.join(_PRESETS)}. "
             "Stocks are looked up via search, so no hardcoded IDs needed. "
             f"Can be combined with --watchlists. Defaults to {_DEFAULT_PRESETS} "
             "when neither this nor --watchlists is given.",
    )
    ap.add_argument(
        "--email",
        action="store_true",
        help="Send an email summary after the run (requires EMAIL_* vars in .env)",
    )
    ap.add_argument(
        "--sell-from",
        default=None,
        metavar="NAME",
        help='Watchlist to check for sell signals (e.g. "Äger"). Included in email.',
    )
    ap.add_argument(
        "--push",
        action="store_true",
        help="Push top-scoring reliable stocks to an Avanza watchlist",
    )
    ap.add_argument(
        "--push-to",
        default="Bör köpa",
        metavar="NAME",
        help='Name of the Avanza watchlist to push results to (default: "Bör köpa")',
    )
    ap.add_argument(
        "--push-top",
        type=int,
        default=10,
        metavar="N",
        help="How many top stocks to push (default: 10)",
    )
    args = ap.parse_args()
    os.makedirs("data", exist_ok=True)

    save_data = args.save or args.get_hist
    want_correlate = (
        args.correlate or args.optimize_individual or args.optimize_combo
    )

    def _run_correlate_optimize():
        calculate_metrics_given_hist()
        baseline_correlation("metrics_by_timespan.csv")
        if args.optimize_individual:
            optimize_weights_and_thresholds("metrics_by_timespan.csv")
        if args.optimize_combo:
            optimize_combo("metrics_by_timespan.csv")

    # --correlate/--optimize* without --save: just use already-saved
    # historical data, no live Avanza fetch needed.
    if want_correlate and not save_data:
        _run_correlate_optimize()
        return 0

    # --- Live analysis ---
    manager = SummaryManager()

    # Load optimized weights and thresholds unless --no-opt
    if not args.no_opt:
        if args.use_individual:
            opt_weights, opt_thresholds = _load_optimized_params("individual")
        elif args.use_combo:
            opt_weights, opt_thresholds = _load_optimized_params("combo")
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

    preset_names, watchlist_names = _resolve_universe(args.preset, args.watchlists)

    all_watchlists = avanza.get_watchlists()
    missing_wls = []
    for wl_name in watchlist_names:
        wl = next((w for w in all_watchlists if _wl_attr(w, "name") == wl_name), None)
        if wl is None:
            missing_wls.append(wl_name)
            continue
        ids = [str(oid) for oid in (_wl_attr(wl, "orderbookIds") or [])]
        ticker_id_set.update(ids)
        sources.append(f"{wl_name} ({len(ids)} stocks)")
    if missing_wls:
        print(f"[WARN] Watchlist(s) not found on Avanza: {', '.join(missing_wls)}")

    # Add tickers from built-in presets (search-based, no hardcoded IDs)
    for preset_name in preset_names:
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

        if not ticker_info["sectors"] or ticker_id in EXCLUDED_TICKER_IDS:
            continue
        yahoo_symbol = to_yahoo_symbol(ticker_info)
        if yahoo_symbol is None:
            print(
                f"[WARN] No Yahoo ticker mapping for "
                f"{ticker_info.get('name', ticker_id)} "
                f"(countryCode={ticker_info['listing'].get('countryCode')}) — skipping FCFY."
            )
            yahoo = None
        else:
            yahoo = yf.Ticker(yahoo_symbol)

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

    # --save combined with --correlate/--optimize*: fresh snapshots were just
    # written above, so re-run the backtest/optimizer against them now,
    # matching the documented `--save ... --correlate --optimize` workflow.
    # (Without --save, this was already handled by the early return above.)
    if want_correlate:
        _run_correlate_optimize()
        return 0

    calculate_score(manager)

    manager._display(save_df=True)

    push_results = None
    if args.push:
        push_results = _update_watchlist(avanza, manager, top_n=args.push_top, target_name=args.push_to)

    sell_signals = []
    if args.sell_from:
        sell_signals = _compute_sell_signals(avanza, manager, args.sell_from)
        if sell_signals:
            print(f"\n{'=' * 70}")
            print(f"  SELL SIGNALS IN '{args.sell_from}'")
            print(f"{'=' * 70}")
            if not push_results:
                # _update_watchlist already printed this legend when --push
                # ran in the same invocation -- avoid repeating it.
                for line in LEGEND_LINES:
                    print(f"  {line}")
            for sig in sell_signals:
                pts_str = f"{sig['pts']:+.2f}" if sig["pts"] is not None else "N/A"
                sp_str  = _fmt_reliability(sig["spearman"], sig.get("n_windows"))
                print(f"  ! {sig['name']}  ({pts_str} pts, r={sp_str}) — {sig['reasons']}")
            print(f"{'=' * 70}\n")
        else:
            print(f"\n✓ No sell signals in '{args.sell_from}'.\n")

    if args.email:
        _send_email(push_results, sell_signals)

    return 0


if __name__ == "__main__":

    result = main()
