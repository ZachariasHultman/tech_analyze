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

from analyzer.config import (
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
    """Load optimized weights and thresholds from the panel results file.

    variant: None (default) or "panel" — both read
    optimization_results_panel.json, the only file the optimizer can produce.
    Returns (weights_dict, thresholds_dict) — either may be None.

    Written by the challenger gate on both accept and reject, so when it
    exists it always holds a defensible recommendation. When it doesn't (a
    fresh clone, or a machine that has never run --backfill-panel +
    --optimize), scoring falls back to metrics.py's hardcoded defaults.
    """
    import json
    filename = "optimization_results_panel.json"

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


def _fmt_scored_row(prefix, r):
    """Format a (name, pts, quality_pct, value_pct, combined_score) row --
    shared by the terminal watchlist report and the email summary so both
    stay consistent. Falls back to a plain name when the stock wasn't part
    of this run's scored universe (e.g. a watchlist holding outside the
    current --preset/--watchlists scope)."""
    name, pts, qual, val, cscore = r
    if pd.isna(qual) and pd.isna(val) and pd.isna(cscore):
        return f"{prefix} {name}  (not scored this run)"
    pts_str = f"{pts:+.2f}" if pd.notna(pts) else "N/A"
    return (f"{prefix} {name}  (pts={pts_str}, q={qual:.2f}, v={val:.2f}, "
            f"combined={cscore:.2f})")


# Weights older than this are flagged in the email; the panel only gains a
# new fiscal year once a year, so this is about noticing neglect, not decay.
_STALE_AFTER_DAYS = 120


def _load_optimizer_status(variant=None):
    """Read the `validation` block written by the challenger gate, if present.

    Returns None when the weights file is missing or predates this block (an
    older SCP, or a machine that has never run the optimizer). Callers render
    a short "unknown" notice instead -- the Pi must never fail its weekly run
    because the provenance metadata is stale.
    """
    import json
    filename = ("optimization_results_panel.json" if variant in (None, "panel")
                else f"optimization_results_{variant}.json")
    path = os.path.join(project_root, filename)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return None
    validation = data.get("validation")
    if not isinstance(validation, dict):
        return None
    return {
        "accepted": data.get("accepted"),
        "dsr": data.get("dsr"),
        "confidence": data.get("confidence"),
        "source": filename,
        **validation,
    }


def _format_optimizer_status(status):
    """Render the system-status block shown above the buy/sell lists.

    The point is that a tidy top-10 list implies more confidence than four
    fiscal years can support, so the confidence verdict is stated on every
    run rather than left for the reader to remember.
    """
    if not status:
        return [
            "SYSTEM STATUS: unknown — no validation metadata found.",
            "  Run the optimizer on the Mac and copy",
            "  optimization_results_panel.json across.",
        ]

    lines = []
    verdict = ("ACCEPTED" if status.get("accepted") else
               "REJECTED (running equal weight)")
    fitted = status.get("fitted_at") or "unknown date"
    stale = ""
    try:
        age = (pd.Timestamp.now() - pd.Timestamp(fitted)).days
        if age >= _STALE_AFTER_DAYS:
            stale = f"  [STALE — {age} days old, consider re-running]"
    except Exception:
        pass
    lines.append(f"SYSTEM STATUS  (weights fitted {fitted[:10]}, "
                 f"panel challenger: {verdict}){stale}")

    n_periods = status.get("n_periods") or 0
    basis = status.get("return_basis", "forward return")
    years = [r for r in (status.get("per_year") or []) if r.get("ic") is not None]
    if years:
        span = f"{years[0]['fiscal_year']}-{years[-1]['fiscal_year']}"
        lines.append(f"  Out-of-sample, {len(years)} fiscal years ({span}), {basis}:")
        lines.append("    IC by year:   " + "   ".join(
            f"{r['fiscal_year']} {r['ic']:+.3f}" for r in years))

    mean_ic, mean_spread = status.get("mean_ic"), status.get("mean_spread")
    if mean_ic is not None:
        bits = [f"mean IC {mean_ic:+.3f}"]
        if mean_spread is not None:
            bits.append(f"top-bottom spread {mean_spread:+.1%}/yr")
        lines.append("    " + "  |  ".join(bits))

    n_beat, n_folds = status.get("n_folds_beating_equal"), status.get("n_folds")
    if n_beat is not None and n_folds:
        # On a reject the rows above describe equal weight (what is running),
        # while this count describes the challenger that was turned down --
        # say which, or the two read as contradicting each other.
        who = "beat" if status.get("accepted") else "challenger beat"
        tail = "" if status.get("accepted") else " but missed the significance bar"
        lines.append(f"    {who} equal weight in {n_beat} of {n_folds} "
                     f"held-out years{tail}")

    p_perm, n_perm = status.get("permutation_p_value"), status.get("n_permutations")
    if p_perm is not None and n_perm:
        lines.append(f"    permutation test ({n_perm} refits on shuffled "
                     f"targets): p={p_perm:.3f}")

    # The verdict is derived, not asserted. It used to be hardcoded "LOW",
    # which became self-contradictory the moment the IC test started clearing
    # 5% ("CONFIDENCE: LOW ... p=0.02").
    t_stat, p_value = status.get("t_stat"), status.get("p_value")
    detail = ""
    if t_stat is not None and p_value is not None:
        detail = f" (t={t_stat:+.2f}, p={p_value:.2f})"
    if p_value is not None and p_value < 0.05:
        lines.append(f"  CONFIDENCE: MODERATE — ranking beats chance across "
                     f"{n_periods} periods{detail}.")
    else:
        lines.append(f"  CONFIDENCE: LOW — {n_periods} periods only{detail}. "
                     "Directional, not proof.")

    # A significant IC and a flat quintile spread are different claims, and
    # the spread is the one a top-N watchlist actually depends on. Say so
    # rather than letting a good IC imply the picks are validated.
    if mean_spread is not None and abs(mean_spread) < 0.02:
        lines.append(f"    CAVEAT: top-bottom spread is only {mean_spread:+.1%}/yr "
                     "— the ranking works on average, the extremes do not "
                     "separate, and the watchlist is an extreme.")
    lines.append("    Survivorship bias is not corrected for and inflates all "
                 "of the above.")
    n_companies = status.get("n_companies")
    if n_companies:
        lines.append(f"  Universe: {n_companies} stocks "
                     "(today's list applied backwards).")
    return lines


# Shared explainer for q/v/combined, printed in both the terminal watchlist
# report and the email summary.
LEGEND_LINES = [
    "q = quality percentile (0-1, peer-ranked business health)",
    "v = value percentile (0-1, peer-ranked cheapness)",
    "pts = summed metric score -- the ranking key (the validated one:",
    "      IC +0.041, p=0.017 across 7 fiscal years, vs p=0.18 for q x v)",
    "combined = q x v -- shown for context, and the sleeve gate still applies",
]


def _wl_attr(wl, key):
    """Get attribute from a watchlist (supports both dict and pydantic model)."""
    if isinstance(wl, dict):
        return wl.get(key)
    return getattr(wl, key, None)


def _update_watchlist(avanza, manager, top_n=25, target_name="Bör köpa"):
    """Add top-scoring stocks to the target watchlist.

    - Filters by the two-sleeve gate (quality_pct and value_pct both above
      SLEEVE_GATE_MIN), ranks the rest by combined_score
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
    combined["_combined_score"] = pd.to_numeric(
        combined.get("combined_score"), errors="coerce"
    )

    # Two-sleeve gate: qualify only stocks ranking well in BOTH quality and value.
    combined["_quality"] = pd.to_numeric(combined.get("quality_pct"), errors="coerce")
    combined["_value"] = pd.to_numeric(combined.get("value_pct"), errors="coerce")
    qualified = combined[
        (combined["_quality"] >= SLEEVE_GATE_MIN)
        & (combined["_value"] >= SLEEVE_GATE_MIN)
    ].copy()
    # Ranked on pts, not combined_score. Over 7 fiscal years pts carried the
    # only significant signal (IC +0.041, t=+3.29, p=0.017, positive in 6 of 7
    # years) while combined_score -- despite a higher point estimate -- was not
    # significant (p=0.18, 5 of 7) because multiplying two noisy percentiles
    # amplifies the noise. The sleeve gate above still enforces "decent on
    # both dimensions"; this only changes the ordering within that set.
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
        # (name, pts, quality_pct, value_pct, combined_score)
        df = qualified if df is None else df
        return (
            name,
            _num(name, "_pts", df),
            _num(name, "_quality", df),
            _num(name, "_value", df),
            _num(name, "_combined_score", df),
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
    print(f"  WATCHLIST UPDATE: '{target_name}' (top {top_n} by pts)")
    print(f"{'=' * 70}")
    for line in _format_optimizer_status(_load_optimizer_status()):
        print(f"  {line}")
    print()
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
    """Flag stocks in portfolio_watchlist whose fundamentals have deteriorated.

    A stock is flagged when its points score (pts) is negative. Returns a
    list of dicts sorted worst-first (most negative pts first).
    """
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
    combined["_pts"] = pd.to_numeric(combined["points"], errors="coerce")

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
        pts = float(combined.loc[idx, "_pts"]) if pd.notna(combined.loc[idx, "_pts"]) else None

        if pts is not None and pts < 0:
            signals.append({"name": idx, "pts": pts, "reasons": "fundamentals have deteriorated"})

    return sorted(signals, key=lambda x: x["pts"])


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

    # Provenance first: a tidy top-10 list reads as more confident than four
    # fiscal years justify, so the evidence for the weights leads the email.
    lines.extend(_format_optimizer_status(_load_optimizer_status()))
    lines.append("")

    if push_results or sell_signals:
        lines.extend(LEGEND_LINES)
        lines.append("")

    if push_results:
        n = push_results["top_n"]
        lines.append("=" * 50)
        lines.append(f"  CONSIDER BUYING (top {n} by pts)")
        lines.append("=" * 50)

        top10 = push_results["added"] + push_results["already"]
        top10.sort(key=lambda x: (x[1] is not None, x[1]), reverse=True)  # by pts
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
            lines.append(f"  ! {sig['name']}  ({pts_str} pts) — {sig['reasons']}")
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


def build_arg_parser():
    """The CLI surface, split out of main() so tests can render the help
    screen without running the tool (a literal `%` in any help string only
    ever blows up at --help time)."""
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
  python main.py --backfill-panel              Build the fiscal-year panel from data/*.csv
  python main.py --optimize                    Run the panel challenger gate (do occasionally)

--- Push results to Avanza ---
  python main.py --push                        Push top 10 to 'Bör köpa' (default)
  python main.py --push --push-to "Min lista"  Push to a different watchlist
  python main.py --push --push-top 5           Push top 5 instead of 10

--- Weight variants ---
  (default: uses optimization_results_panel.json when it exists -- run
  --backfill-panel then --optimize once to create it -- else metrics.py's
  hardcoded defaults)
  python main.py --no-opt                      Ignore saved weights, use hardcoded defaults
  python main.py --use-panel                   Force the panel challenger's result
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
        "--optimize",
        action="store_true",
        help="Run the panel challenger gate against data/panel_scores.csv and "
             "save optimization_results_panel.json (run --backfill-panel first)",
    )
    ap.add_argument(
        "--no-opt",
        action="store_true",
        help="Ignore optimized weights, use hardcoded defaults",
    )
    ap.add_argument(
        "--use-panel",
        action="store_true",
        help="Force panel challenger results for live analysis. Not usually "
             "needed -- panel results are already used by default when "
             "optimization_results_panel.json exists (run --backfill-panel "
             "then --optimize once to create it).",
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
        default=25,
        metavar="N",
        help="How many top stocks to push (default: 25). Breadth, not "
             "conviction, is what harvests a weak ranking signal: information "
             "ratio scales as IC x sqrt(breadth), so at IC +0.041 a 10-stock "
             "pick throws away most of the measured edge. 25 is roughly the "
             "top quintile of the universe, matching the bucket the backtest "
             "actually validates. Measured across 7 fiscal years, the "
             "year-to-year standard deviation of the pick's excess return "
             "falls from 9.98%% at N=10 to 4.48%% at N=25.",
    )
    ap.add_argument(
        "--backfill-panel",
        action="store_true",
        help="Build the fiscal-year cross-sectional panel from data/*.csv "
             "(writes data/panel_fundamentals.csv + data/panel_scores.csv). "
             "No live Avanza call.",
    )
    ap.add_argument(
        "--validate",
        action="store_true",
        help="Run the cross-sectional validation battery against "
             "data/panel_scores.csv (run --backfill-panel first). No live call.",
    )
    ap.add_argument(
        "--backfill-prices",
        action="store_true",
        help="One-time Mac-only step: fetch dividend-adjusted daily closes from "
             "Yahoo for every company in data/, verify each against the Avanza "
             "prices already in the snapshots, and cache them for the panel. "
             "Avanza's OHLC is a rolling ~5y unadjusted window, which is what "
             "caps the panel at 4 usable fiscal years; this lifts it toward the "
             "~7 the power calculation asks for. Resumable -- re-run after a "
             "rate limit. Needs Avanza credentials only to resolve symbols the "
             "first time (cached in data/yahoo_symbols.json).",
    )
    ap.add_argument(
        "--prices-from",
        default="2015-01-01",
        metavar="YYYY-MM-DD",
        help="Earliest date for --backfill-prices (default 2015-01-01).",
    )
    ap.add_argument(
        "--prices-batch-size",
        type=int, default=8, metavar="N",
        help="Symbols per Yahoo request (default 8). Lower it if you keep "
             "hitting the rate limit.",
    )
    ap.add_argument(
        "--prices-cooldown",
        type=float, default=600.0, metavar="SECONDS",
        help="Wait after a batch exhausts its retries before moving to the "
             "next one (default 600). Within a batch the backoff is already "
             "30s doubling to a 900s cap; this is the longer between-batch "
             "pause. Raise it for a slow overnight run.",
    )
    ap.add_argument(
        "--fetch-fx",
        action="store_true",
        help="Mac-only manual step: fetch daily ECB reference rates (via "
             "Frankfurter, no API key) into data/fx_sek.csv so the panel's "
             "forward return can be computed in SEK. 44%% of the universe is "
             "not SEK-listed and USD/SEK moved +24%% over the sample, so "
             "without this the within-year demeaning attributes a shared FX "
             "move to whatever metric correlates with being US-listed. "
             "Idempotent -- re-running when the cache already covers the "
             "range makes no requests. The Pi never needs this; the panel "
             "falls back to listing currency when the file is absent.",
    )
    ap.add_argument(
        "--fx-from",
        default="2015-01-01",
        metavar="YYYY-MM-DD",
        help="Earliest date for --fetch-fx (default 2015-01-01, matching "
             "--prices-from).",
    )
    ap.add_argument(
        "--permutations",
        type=int,
        default=200,
        metavar="N",
        help="Permutation refits used to measure the challenger gate's null "
             "distribution (default 200). The target is shuffled within each "
             "fiscal year and the optimizer refitted, so the DSR's benchmark "
             "is measured rather than approximated. 0 skips it and falls back "
             "to the Euler-Mascheroni approximation, whose sigma is grid "
             "dispersion rather than sampling noise -- faster, but the "
             "resulting DSR is not meaningful.",
    )
    ap.add_argument(
        "--challenger-confidence",
        type=float,
        default=0.925,
        metavar="X",
        help="Deflated Sharpe Ratio bar the --optimize panel challenger gate "
             "must clear to accept optimized weights over equal weight "
             "(default 0.925). Only matters once data/panel_scores.csv "
             "exists (run --backfill-panel first).",
    )
    return ap


def main():
    pd.set_option("display.max_rows", None)  # Show all rows
    pd.set_option("display.max_colwidth", None)  # Show full cell content
    print("Avanza API version: ", version("avanza-api"))
    print("yfinance version: ", version("yfinance"))
    ap = build_arg_parser()
    args = ap.parse_args()
    os.makedirs("data", exist_ok=True)

    save_data = args.save or args.get_hist

    def _run_optimize():
        """The whole of --optimize: the panel challenger gate.

        Errors are NOT swallowed. The gate used to be an optional add-on to
        the (now removed) rolling-window optimizers, so a failure there still
        left the command with something to show; it is the only work --optimize
        does now, and a silent "skipped" would look like a successful run that
        wrote nothing.
        """
        if not (os.path.exists("data/panel_scores.csv")
                and os.path.exists("data/panel_fundamentals.csv")):
            print("[optimize] data/panel_scores.csv + data/panel_fundamentals.csv "
                  "not found — run --backfill-panel first.")
            return 1
        from analyzer.panel import load_gate_panel
        from analyzer.correlation import (
            gate_optimized_weights,
            save_panel_optimization_results,
            optimize_panel_combo,
            _all_scored_metrics,
        )
        panel_df = load_gate_panel()
        gate_result = gate_optimized_weights(
            panel_df, _all_scored_metrics(),
            optimizer_fn=optimize_panel_combo,
            confidence=args.challenger_confidence,
            n_permutations=args.permutations,
        )
        # Persisted so live scoring can pick it up automatically (see
        # _load_optimized_params) -- written on both accept and reject, since
        # a reject's chosen_weights is already the gate's own equal-weight
        # fallback, not a broken state.
        save_panel_optimization_results(gate_result)
        return 0

    # --fetch-fx: its own terminal branch, same shape as --backfill-prices --
    # a manual Mac-only fetch whose output is then consumed by
    # --backfill-panel. No Avanza session needed.
    if args.fetch_fx:
        from analyzer.fx import fetch_sek_rates
        end = date.today().strftime("%Y-%m-%d")
        try:
            fetch_sek_rates(args.fx_from, end)
        except Exception as exc:
            print(f"[fx] fetch failed ({exc}) — the panel keeps working "
                  "without the cache, in listing currency.")
            return 1
        print("\n[fx] done. Re-run --backfill-panel to rebuild the panel with "
              "SEK forward returns.")
        return 0

    # --backfill-prices: offline apart from a one-time symbol resolution, and
    # deliberately its own terminal branch -- it is a slow manual step whose
    # output is then consumed by --backfill-panel.
    if args.backfill_prices:
        from analyzer.yahoo_prices import (
            SYMBOLS_PATH, backfill_prices, company_keys, load_symbol_map,
            resolve_symbols_via_avanza, save_symbol_map, verify_all,
        )
        keys = company_keys("data")
        if not keys:
            print("[yahoo] no snapshots in data/ — run --save first.")
            return 1
        symbol_map = load_symbol_map()
        missing = [k for k in keys if not symbol_map.get(k)]
        if missing:
            print(f"[yahoo] resolving {len(missing)} symbol(s) via Avanza...")
            try:
                symbol_map, unresolved = resolve_symbols_via_avanza(
                    setup_env(), keys, existing=symbol_map
                )
                save_symbol_map(symbol_map)
                print(f"[yahoo] wrote {SYMBOLS_PATH} ({len(symbol_map)} symbols)")
                for key, why in unresolved:
                    print(f"  [unresolved] {key}: {why} — add it by hand to "
                          f"{SYMBOLS_PATH} if you know the Yahoo ticker")
            except Exception as exc:
                print(f"[yahoo] symbol resolution failed ({exc}).")
                if not symbol_map:
                    print(f"[yahoo] nothing cached yet — populate {SYMBOLS_PATH} "
                          "by hand or retry with working Avanza credentials.")
                    return 1
                print("[yahoo] continuing with the symbols already cached.")
        else:
            print(f"[yahoo] {len(symbol_map)} symbol(s) already resolved")

        result = backfill_prices(
            symbol_map.values(), start=args.prices_from,
            batch_size=args.prices_batch_size, cooldown=args.prices_cooldown,
        )
        verified = verify_all(symbol_map)
        missing = len(result["rate_limited"]) + len(result["failed"])
        if missing:
            print(f"\n[yahoo] {missing} symbol(s) not downloaded — re-run this "
                  "command to pick them up before rebuilding the panel.")
        print("\n[yahoo] done. Re-run --backfill-panel to rebuild the panel "
              "on the deeper, dividend-adjusted history.")
        if len(verified["verified"]) < 0.9 * len(symbol_map):
            print("[yahoo] NOTE: the panel refuses fiscal years covering under "
                  "60% of the universe, so a partial backfill will not add "
                  "years until it is complete.")
        return 0

    # --backfill-panel / --validate: both operate on already-saved data/*.csv
    # snapshots, no live Avanza session needed. Combinable in one invocation
    # (backfill first, then validate).
    if args.backfill_panel or args.validate:
        if args.backfill_panel:
            from analyzer.panel import build_fundamentals_panel, build_scores_panel
            fundamentals = build_fundamentals_panel("data")
            fundamentals.to_csv("data/panel_fundamentals.csv", index=False)
            print(f"Wrote data/panel_fundamentals.csv ({len(fundamentals)} rows)")
            scores = build_scores_panel(fundamentals, "data")
            scores.to_csv("data/panel_scores.csv", index=False)
            print(f"Wrote data/panel_scores.csv ({len(scores)} rows)")
        if args.validate:
            from analyzer.validation import run_validation_battery
            if not os.path.exists("data/panel_scores.csv"):
                print("[validate] data/panel_scores.csv not found — "
                      "run --backfill-panel first.")
                return 1
            run_validation_battery("data/panel_scores.csv")
        return 0

    # --optimize without --save: runs against the already-built panel, no
    # live Avanza fetch needed.
    if args.optimize and not save_data:
        return _run_optimize()

    # --- Live analysis ---
    manager = SummaryManager()

    # Load optimized weights and thresholds unless --no-opt
    if not args.no_opt:
        if args.use_panel:
            opt_weights, opt_thresholds = _load_optimized_params("panel")
        else:
            # Reads optimization_results_panel.json when present -- see
            # _load_optimized_params's docstring.
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

    # One browser-impersonating session, reused for every ticker. Yahoo blocks
    # plain yfinance on TLS fingerprint: `yf.Ticker("ABB.ST").cashflow` comes
    # back EMPTY, which is why `fcfy_pe ratio status` was 0% populated in the
    # last live run while every other metric was 100%. Through this session the
    # same call returns a full statement. Falls back to None (previous
    # behaviour) if curl_cffi is unavailable.
    from analyzer.yahoo_prices import _impersonating_session
    yf_session = _impersonating_session()
    if yf_session is None:
        print("[WARN] curl_cffi unavailable — Yahoo may block FCFY lookups. "
              "Run `uv sync` to install it.")

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
            yahoo = (yf.Ticker(yahoo_symbol, session=yf_session)
                     if yf_session is not None else yf.Ticker(yahoo_symbol))

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

    # --save combined with --optimize: fresh snapshots were just written
    # above, so run the gate now. Note the panel itself is NOT rebuilt here --
    # --backfill-panel is still its own separate invocation.
    # (Without --save, this was already handled by the early return above.)
    if args.optimize:
        return _run_optimize()

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
                print(f"  ! {sig['name']}  ({pts_str} pts) — {sig['reasons']}")
            print(f"{'=' * 70}\n")
        else:
            print(f"\n✓ No sell signals in '{args.sell_from}'.\n")

    if args.email:
        _send_email(push_results, sell_signals)

    return 0


if __name__ == "__main__":

    result = main()
