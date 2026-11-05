import sys
import os
import yfinance as yf
from tqdm import tqdm
import re
import html


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
from analyzer.summary_manager import (
    SummaryManager, RANK_COLUMNS, one_per_company, rank_key,
)
from analyzer.data_processing import *
from importlib.metadata import version

from analyzer.config import (
    SLEEVE_GATE_MIN,
    MOMENTUM_BLEND_WEIGHT,
    EXCLUDED_TICKER_IDS,
    EXCLUDED_SECTORS,
    CORE_MIX,
    CORE_LABEL,
    SATELLITE_SHARE,
    WITHHOLDING_LOSS,
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


def to_yahoo_symbol(ticker_info) -> str | None:
    """Map an Avanza ticker_info dict to a Yahoo Finance ticker symbol.

    Returns None for unrecognized country codes, or when a DE ticker's
    symbol doesn't start with a recognizable letter prefix — callers should
    skip yfinance-dependent metrics (FCFY) for that stock rather than crash.

    US is a no-suffix passthrough: Avanza's raw tickerSymbol for
    NYSE/NASDAQ-listed stocks already matches the bare Yahoo symbol
    (verified against live data, e.g. "GM", "KR", "DIS") — this mirrors the
    pre-fix fallthrough behavior, which happened to be correct for US.
    FI takes ".HE" with spaces as hyphens, like the other Nordic venues
    (verified on Yahoo: NOKIA.HE, KNEBV.HE, WRT1V.HE, NDA-FI.HE). Other
    unmapped countries are NOT a safe passthrough -- Yahoo needs an exchange
    suffix -- so they still return None rather than guessing; a wrong guess
    is caught later by yahoo_prices.verify_symbol, but FCFY would read it live.
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
    if country == "FI":
        return symbol.replace(" ", "-") + ".HE"
    if country == "DE":
        m = re.match(r"^[A-Z]+", symbol)
        return m.group() + ".DE" if m else None
    if country == "US":
        return symbol
    return None


def _is_excluded(ticker_id, ticker_info):
    """Should this ticker be left out of the run entirely?

    Three reasons, all deliberate: no sector metadata at all (nothing can be
    scored), an explicitly excluded id, or an excluded sector. The last is
    investment companies -- see config.EXCLUDED_SECTORS for why they are out
    rather than scored on a partial template.
    """
    sectors = ticker_info.get("sectors") or []
    if not sectors:
        return True
    if ticker_id in EXCLUDED_TICKER_IDS:
        return True
    return any(s.get("sectorName") in EXCLUDED_SECTORS for s in sectors)


def _extract_orderbook_id(index_name):
    """Extract the orderbookId from an index like 'Glencore plc 2165695'."""
    parts = str(index_name).rsplit(" ", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[1]
    return None


def _fmt_scored_row(prefix, r):
    """Format a (name, pts, quality_pct, value_pct, combined_score, blend,
    momentum) row -- shared by the terminal watchlist report and the email
    summary so both stay consistent. Falls back to a plain name when the stock
    wasn't part of this run's scored universe (e.g. a watchlist holding outside
    the current --preset/--watchlists scope).

    Accepts the older 5-field row so callers that don't carry the blend (and
    the tests pinning that shape) keep working."""
    name, pts, qual, val, cscore = r[:5]
    blend = r[5] if len(r) > 5 else float("nan")
    mom = r[6] if len(r) > 6 else float("nan")
    if pd.isna(qual) and pd.isna(val) and pd.isna(cscore):
        return f"{prefix} {name}  (not scored this run)"
    pts_str = f"{pts:+.2f}" if pd.notna(pts) else "N/A"
    head = f"{prefix} {name}  ("
    # "score" is blend_score -- renamed on display only, the column keeps its
    # name for the panel and validation. combined_score is no longer shown: it
    # decides nothing (the gate reads q and v directly).
    if pd.notna(blend):
        head += f"score={blend:.2f}, mom={mom:.2f}, " if pd.notna(mom) else f"score={blend:.2f}, "
    return f"{head}pts={pts_str}, q={qual:.2f}, v={val:.2f})"


def _evaluation_status_lines():
    """The SYSTEM STATUS block from evaluation_results.json (--evaluate).

    Never raises: the Pi must send its weekly email whether or not it has been
    sent a current evaluation file.
    """
    from analyzer.evaluation import RESULTS_PATH, format_status, load_status
    try:
        return format_status(load_status(os.path.join(project_root, RESULTS_PATH)))
    except Exception as e:
        return [f"SYSTEM STATUS: could not render evaluation ({type(e).__name__}: {e})"]


# Shared explainer for the row fields, printed in both the terminal watchlist
# report and the email summary.
LEGEND_LINES = [
    "score = THE ranking key (blend_score internally): 50% momentum percentile",
    "        + 50% fundamentals (pts) percentile, 0-1 vs all scored stocks.",
    "        (IC +0.118 over 7 fiscal years, p=0.052, positive 7 of 7, vs",
    "        +0.081 p=0.078 for pts alone; top-25 basket +6.36%/yr excess",
    "        but only p=0.146 -- the ordering is the solid part, not the tail)",
    "mom = momentum: price vs its own SMA200, as a peer percentile (0-1)",
    "pts = fundamentals: summed metric score -- the other half of score, and",
    "      what the sell-signal deterioration check reads",
    "q = quality percentile, v = value percentile (0-1, peer-ranked); both",
    "    must be >= 0.40 to qualify at all",
]


def _wl_attr(wl, key):
    """Get attribute from a watchlist (supports both dict and pydantic model)."""
    if isinstance(wl, dict):
        return wl.get(key)
    return getattr(wl, key, None)


def _resolve_target_watchlist(avanza, target_name):
    """Find the named watchlist and return (watchlist_id, existing_orderbook_ids).

    Returns None when it does not exist -- the caller must not create it, since
    a typo'd name would otherwise silently produce a second list that the
    weekly run then starts pushing to.
    """
    watchlists = avanza.get_watchlists()
    target = next(
        (wl for wl in watchlists if _wl_attr(wl, "name") == target_name), None
    )
    if target is None:
        print(f"\n[WARN] Watchlist '{target_name}' not found on Avanza.")
        print("  Please create it manually in Avanza first, then re-run.")
        return None

    watchlist_id = _wl_attr(target, "watchListId") or _wl_attr(target, "id")
    existing_ids = set(str(oid) for oid in (_wl_attr(target, "orderbookIds") or []))
    return watchlist_id, existing_ids


def _gate_and_rank(manager, top_n):
    """Apply the two-sleeve gate, rank, and cut to the top N.

    Returns (combined, qualified) or None when there is nothing rankable.
    Split out of _update_watchlist because this -- not the Avanza calls around
    it -- is the part that decides *what gets bought*, and it is testable with
    a bare DataFrame and no fake broker at all.
    """
    # Admitted on any of summary_manager.RANK_COLUMNS, not on `points` alone:
    # gating the intake on one rung of the fallback chain silently defeated the
    # other two. A combined_score-only cross-section used to be dropped here,
    # so `_display` printed a full table while this pushed an empty watchlist.
    frames = []
    for summary in [manager.summary]:
        if summary is not None and isinstance(summary, pd.DataFrame) and not summary.empty:
            if any(col in summary.columns for col in RANK_COLUMNS):
                frames.append(summary)

    if not frames:
        print("\n[WARN] No scored stocks to add to watchlist.")
        return None

    combined = pd.concat(frames)
    combined["_pts"] = pd.to_numeric(combined.get("points"), errors="coerce")
    combined["_combined_score"] = pd.to_numeric(
        combined.get("combined_score"), errors="coerce"
    )

    # Two-sleeve gate: qualify only stocks ranking well in BOTH quality and value.
    combined["_quality"] = pd.to_numeric(combined.get("quality_pct"), errors="coerce")
    combined["_value"] = pd.to_numeric(combined.get("value_pct"), errors="coerce")
    combined["_blend"] = pd.to_numeric(combined.get("blend_score"), errors="coerce")
    combined["_momentum"] = pd.to_numeric(combined.get("momentum_pct"), errors="coerce")
    qualified = combined[
        (combined["_quality"] >= SLEEVE_GATE_MIN)
        & (combined["_value"] >= SLEEVE_GATE_MIN)
    ].copy()

    # Ranked on blend_score: half the raw SMA200 distance, half the fundamental
    # point total, each as a cross-sectional percentile (see
    # config.MOMENTUM_BLEND_WEIGHT for the measurements behind the 50/50).
    #
    # Ranking on pts alone -- the previous key, itself chosen because pts beat
    # combined_score -- gives the top-25 basket +2.59%/yr excess (p=0.495,
    # positive 4 of 7 fiscal years) on the current panel. The blend gives
    # +6.36%/yr (p=0.146, 4 of 7). Momentum alone scores a higher mean
    # (+6.82%/yr, 5 of 7) but is the metric survivorship bias most inflates.
    #
    # None of those basket p-values clear anything: the case for the blend is
    # the ordering, where it beats both (IC +0.118 positive 7 of 7, against
    # momentum's +0.112 and pts' +0.081). See config.MOMENTUM_BLEND_WEIGHT.
    #
    # The sleeve gate above still enforces "decent on both dimensions"; this
    # only changes the ordering within that set. `pts` is untouched and still
    # drives the sell-signal deterioration check.
    #
    # One shared implementation with SummaryManager._display -- see
    # summary_manager.rank_key for why the chain lives in exactly one place.
    key = rank_key(qualified)
    if key is None:
        print("\n[WARN] No ranking column (blend_score/points/combined_score) "
              "on the scored stocks; refusing to push an arbitrary order.")
        return None
    qualified = one_per_company(
        qualified.loc[key.sort_values(ascending=False).index]).head(top_n)
    return combined, qualified


def _sync_watchlist(avanza, watchlist_id, combined, qualified, existing_ids):
    """Add what qualifies, remove what no longer does.

    Returns (added, already, failed, removed) as lists of index labels; every
    Avanza call is individually guarded so one rejected symbol cannot abort the
    weekly run partway through the list.
    """
    qualified_ids = set()
    for idx in qualified.index:
        oid = _extract_orderbook_id(idx)
        if oid:
            qualified_ids.add(oid)

    added, already, failed, removed = [], [], [], []

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

    return added, already, failed, removed


def _scored_rows(names, qualified, df=None):
    """(name, pts, quality, value, combined_score, blend, momentum) per name."""
    frame = qualified if df is None else df

    def _num(name, col):
        try:
            v = frame.loc[name, col]
            return float(v) if pd.notna(v) else float("nan")
        except Exception:
            return float("nan")

    return [
        (n, _num(n, "_pts"), _num(n, "_quality"), _num(n, "_value"),
         _num(n, "_combined_score"), _num(n, "_blend"), _num(n, "_momentum"))
        for n in names
    ]


def _print_watchlist_report(target_name, top_n, added_rows, already_rows,
                            removed_rows, failed):
    print(f"\n{'=' * 70}")
    print(f"  WATCHLIST UPDATE: '{target_name}' (top {top_n} by rank)")
    print(f"{'=' * 70}")
    for line in _evaluation_status_lines():
        print(f"  {line}")
    print()
    for line in LEGEND_LINES:
        print(f"  {line}")
    if added_rows:
        print(f"\n  Added {len(added_rows)} stock(s):")
        for r in added_rows:
            print(f"    {_fmt_scored_row('+', r)}")
    if already_rows:
        print(f"\n  Already on list ({len(already_rows)}):")
        for r in already_rows:
            print(f"    {_fmt_scored_row('=', r)}")
    if removed_rows:
        print(f"\n  Removed {len(removed_rows)} stock(s) (no longer in top {top_n}):")
        for r in removed_rows:
            print(f"    {_fmt_scored_row('-', r)}")
    if failed:
        print(f"\n  Failed ({len(failed)}):")
        for name, err in failed:
            print(f"    ! {name}: {err}")
    print(f"{'=' * 70}\n")


def _update_watchlist(avanza, manager, top_n=25, target_name="Bör köpa"):
    """Add top-scoring stocks to the target watchlist.

    - Filters by the two-sleeve gate (quality_pct and value_pct both above
      SLEEVE_GATE_MIN), ranks the rest by summary_manager.rank_key
    - Adds qualified stocks that aren't already on the list
    - Removes stocks from the list that no longer qualify
    """
    resolved = _resolve_target_watchlist(avanza, target_name)
    if resolved is None:
        return
    watchlist_id, existing_ids = resolved

    ranked = _gate_and_rank(manager, top_n)
    if ranked is None:
        return
    combined, qualified = ranked

    added, already, failed, removed = _sync_watchlist(
        avanza, watchlist_id, combined, qualified, existing_ids
    )

    added_rows = _scored_rows(added, qualified)
    already_rows = _scored_rows(already, qualified)
    # removed stocks are excluded from `qualified` by definition (that's why
    # they were removed) -- look their metrics up in `combined` instead,
    # which still has every scored company's row.
    removed_rows = _scored_rows(removed, qualified, df=combined)

    _print_watchlist_report(target_name, top_n, added_rows, already_rows,
                            removed_rows, failed)

    return {
        "target_name": target_name,
        "top_n": top_n,
        "added": added_rows,
        "already": already_rows,
        "removed": removed_rows,
        "failed": failed,
        # For analyzer/frontier.py: the pushed names, and the blend over the
        # whole scored universe (alpha's z-score is standardised there).
        "qualified_names": list(qualified.index),
        "blend_all": combined["_blend"].dropna(),
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
    for summary in [manager.summary]:
        if summary is not None and isinstance(summary, pd.DataFrame) and not summary.empty:
            frames.append(summary)
    if not frames:
        return []

    combined = pd.concat([f.dropna(axis=1, how="all") for f in frames])
    # `points` is the only key here by design -- the deterioration check is a
    # sum-of-scores question, not a ranking one, so there is no fallback chain
    # to walk (see summary_manager.rank_key for the side that does have one).
    # It still has to be absent-safe: `dropna(axis=1, how="all")` above drops
    # an all-NaN `points` column outright, which turned a missing score into a
    # KeyError instead of a skipped check.
    if "points" not in combined.columns:
        print("[WARN] No `points` column on the scored stocks — "
              "skipping sell check.")
        return []
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


def _get_data_or_skip(ticker_id, ticker_info, manager, avanza, yahoo,
                      save_data, skipped, retries=1):
    """``get_data`` with one retry on a network error, else skip the ticker.

    A single dropped connection (seen live: ChunkedEncodingError from
    Avanza's chart endpoint) used to abort the whole run -- the weekly Pi job
    included. A ticker that still fails is removed from the summary, since
    get_data initialises its row before fetching and a half-filled row would
    otherwise be scored on whatever metrics happened to land.
    """
    import requests
    name = f'{ticker_info.get("name")} {ticker_info.get("orderbookId")}'
    for attempt in range(retries + 1):
        try:
            return get_data(ticker_id, ticker_info, manager, avanza, yahoo,
                            get_hist=save_data)
        except requests.exceptions.RequestException as e:
            if attempt < retries:
                continue
            if isinstance(manager.summary, dict):
                manager.summary.pop(name, None)
            manager.price_hist.pop(name, None)
            skipped.append((ticker_id, f"{type(e).__name__}: {e}"))
            return None


def _listing_currency(ticker_info, yahoo_symbol):
    """The currency Avanza quotes this listing in, or None.

    Avanza's own field first; the Yahoo suffix map as fallback (same listing,
    so same currency -- only the GBP minor unit differs, which
    frontier.avanza_close_to_sek deliberately does not apply).
    """
    from analyzer.fx import currency_for_symbol
    listing = ticker_info.get("listing", {}) if isinstance(ticker_info, dict) else {}
    ccy = listing.get("currency") if isinstance(listing, dict) else None
    return str(ccy).upper() if ccy else currency_for_symbol(yahoo_symbol)


# Portfolio -> Avanza watchlist it is mirrored to. Created by hand in Avanza,
# like the push target: _resolve_target_watchlist refuses to create lists.
PORTFOLIO_WATCHLISTS = {"minvar": "min-var", "maxsharpe": "max-sharpe"}


def _sync_portfolio_watchlists(avanza, p, known_names):
    """Mirror each portfolio's held names onto its Avanza watchlist.

    Returns one note per list for the email. A list that is missing, or a
    portfolio that was not computed, is left untouched -- emptying a list
    because a computation failed would read as "sell everything".
    """
    from analyzer.frontier import held_names
    notes = []
    for key, wl_name in PORTFOLIO_WATCHLISTS.items():
        port = p.get(key) if p else None
        if port is None:
            continue
        try:
            resolved = _resolve_target_watchlist(avanza, wl_name)
            if resolved is None:
                notes.append(f"Avanza '{wl_name}': list not found, not updated.")
                continue
            wl_id, existing = resolved
            held = held_names(port["weights"])
            added, _, failed, removed = _sync_watchlist(
                avanza, wl_id, pd.DataFrame(index=list(known_names)),
                pd.DataFrame(index=held), existing)
            note = (f"Avanza '{wl_name}': {len(held)} stocks "
                    f"(+{len(added)} added, -{len(removed)} removed)")
            if failed:
                note += f", {len(failed)} failed: " + "; ".join(
                    f"{n}: {err}" for n, err in failed)
        except Exception as e:
            note = f"Avanza '{wl_name}': sync failed ({type(e).__name__}: {e})"
        notes.append(note)
    return notes


def _fetch_core(avanza):
    """The core funds' combined SEK price series from Avanza, or None with a
    note. Fetched live every run (two chart calls), so the Pi needs no cache;
    a failure degrades to optimising the satellite on its own."""
    if avanza is None:
        return None, []
    from analyzer.benchmarks import core_close, fetch_avanza_series
    try:
        series = {oid: fetch_avanza_series(avanza, oid) for oid in CORE_MIX}
        core = core_close(series, CORE_MIX)
        if core is None:
            return None, ["Core funds returned no overlapping history: "
                          "satellite optimised on its own."]
        return core, []
    except Exception as e:
        return None, [f"Core funds unavailable ({type(e).__name__}): "
                      "satellite optimised on its own."]


def _portfolio_section(manager, push_results, avanza=None):
    """Min-variance and max-Sharpe weights over the pushed names, as
    ``(text_lines, html)``, mirrored onto the min-var / max-sharpe watchlists
    when an Avanza client is given.

    Never raises: a failure here must not cost the weekly email its buy/sell
    sections, so it degrades to a one-line note.
    """
    from analyzer import frontier, fx
    try:
        names = push_results.get("qualified_names") or []
        hist = {n: manager.price_hist.get(n) for n in names}
        ccys = {n: manager.price_ccy.get(n) for n in names}
        rates = None
        if any(c and c != "SEK" for c in ccys.values()):
            today = pd.Timestamp(date.today())
            try:
                rates = fx.fetch_sek_rates(
                    (today - pd.DateOffset(years=6)).date().isoformat(),
                    today.date().isoformat(),
                    path=os.path.join(project_root, fx.FX_CACHE_PATH),
                    currencies=sorted({c for c in ccys.values() if c and c != "SEK"}),
                )
            except Exception as e:
                print(f"[WARN] FX fetch failed ({e}); using cached rates if any.")
                rates = fx.load_sek_rates(os.path.join(project_root, fx.FX_CACHE_PATH))
        blend_all = push_results.get("blend_all")
        core, notes = _fetch_core(avanza)
        drag = frontier.withholding_drag(
            {n: manager.div_yield.get(n) for n in names},
            {n: manager.domicile.get(n) for n in names}, WITHHOLDING_LOSS)
        p = frontier.build_portfolios(
            hist, ccys, blend_all, names, rates, core_close=core,
            satellite_share=SATELLITE_SHARE if core is not None else 1.0,
            tax_drag=drag,
            sectors={n: manager.sector_group.get(n) for n in names})
        if p is not None and p.get("core"):
            p["core"]["label"] = CORE_LABEL
        if avanza is not None:
            known = set(names) | set(getattr(blend_all, "index", []))
            notes += _sync_portfolio_watchlists(avanza, p, known)
        if p is not None and p.get("core_note"):
            notes.append(p["core_note"])
        return (frontier.format_portfolios(p, notes),
                frontier.format_portfolios_html(p, notes))
    except Exception as e:
        msg = f"Not computed: {type(e).__name__}: {e}"
        return (["=" * 50, "  PORTFOLIO WEIGHTS", "=" * 50, f"  {msg}"],
                f"<h3>PORTFOLIO WEIGHTS</h3><p>{html.escape(msg)}</p>")


_TH = "padding:6px 10px;text-align:right;border-bottom:2px solid #444;"
_TD = "padding:4px 10px;text-align:right;font-variant-numeric:tabular-nums;"
_TDL = "padding:4px 10px;text-align:left;"


def _buy_table_html(n, rows, added_set, removed):
    """The CONSIDER BUYING list as an HTML table, same rows and order as the
    text part. Inline styles only -- Gmail strips <style> blocks."""
    from analyzer.frontier import display_name
    e = html.escape

    def _f(x, fmt):
        return format(x, fmt) if x is not None and pd.notna(x) else "–"

    cols = ["", "Stock", "Score", "Momentum", "Pts", "Quality", "Value"]
    sub = ["", "", "ranking key", "50% of score", "50% of score", "", ""]

    def _tr(tag, r, bg, color=""):
        name, pts, q, v = r[:4]
        blend = r[5] if len(r) > 5 else None
        mom = r[6] if len(r) > 6 else None
        cells = [e(tag), e(display_name(name)), f"<b>{_f(blend, '.2f')}</b>",
                 _f(mom, ".2f"), _f(pts, "+.2f"), _f(q, ".2f"), _f(v, ".2f")]
        tds = "".join(f'<td style="{_TDL if i < 2 else _TD}{color}">{c}</td>'
                      for i, c in enumerate(cells))
        return f'<tr style="background:{bg}">{tds}</tr>'

    head = "".join(
        f'<th style="{_TH}{"text-align:left;" if i < 2 else ""}">{e(c)}'
        + (f'<br><span style="font-weight:normal;color:#666">{e(s)}</span>' if s else "")
        + "</th>"
        for i, (c, s) in enumerate(zip(cols, sub)))
    out = [f'<h3 style="margin:24px 0 6px;font-family:sans-serif">'
           f"CONSIDER BUYING (top {n} by score)</h3>",
           '<table style="border-collapse:collapse;font-family:sans-serif;font-size:13px">',
           f"<tr>{head}</tr>"]
    for i, r in enumerate(rows):
        tag = "NEW" if r[0] in added_set else ""
        out.append(_tr(tag, r, "#fafafa" if i % 2 else "#ffffff"))
    if removed:
        out.append(f'<tr><td colspan="7" style="{_TDL}padding-top:14px;'
                   f'border-bottom:1px solid #ccc"><b>Removed from list '
                   f"({len(removed)})</b></td></tr>")
        for r in removed:
            out.append(_tr("−", r, "#ffffff", "color:#888;"))
    out.append("</table>")
    return "\n".join(out)


def _send_email(push_results: dict | None, sell_signals: list[dict],
                portfolio_lines: list[str] | None = None,
                portfolio_html: str | None = None) -> None:
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
    lines.extend(_evaluation_status_lines())
    lines.append("")

    if push_results or sell_signals:
        lines.extend(LEGEND_LINES)
        lines.append("")

    buy_html = ""
    buy_start = len(lines)  # the HTML part swaps lines[buy_start:] for a table
    if push_results:
        n = push_results["top_n"]
        lines.append("=" * 50)
        lines.append(f"  CONSIDER BUYING (top {n} by score)")
        lines.append("=" * 50)

        top10 = push_results["added"] + push_results["already"]

        # Ordered by the same key the list was selected on. Sorting the email
        # by pts while _update_watchlist selected on the blend would print the
        # right names in the wrong order, which reads as a different list.
        def _order(row):
            blend = row[5] if len(row) > 5 else None
            key = blend if blend is not None and pd.notna(blend) else row[1]
            return (key is not None and pd.notna(key), key)

        top10.sort(key=_order, reverse=True)
        added_set = {r[0] for r in push_results["added"]}
        for r in top10:
            tag = "NEW" if r[0] in added_set else "   "
            lines.append(f"  {_fmt_scored_row(tag, r)}")

        if push_results["removed"]:
            lines.append(f"\nRemoved from list ({len(push_results['removed'])}):")
            for r in push_results["removed"]:
                lines.append(f"  {_fmt_scored_row('-', r)}")
        buy_html = _buy_table_html(n, top10, added_set, push_results["removed"])

    # Split around the portfolio block: the text part splices its lines in,
    # the HTML part a real table, with everything else in a monospace <pre>
    # (Gmail renders text/plain proportionally, which breaks the columns).
    head = lines
    lines = []

    if sell_signals:
        lines.append("\n" + "=" * 50)
        lines.append("  CONSIDER SELLING")
        lines.append("=" * 50)
        for sig in sell_signals:
            pts_str = f"{sig['pts']:+.2f}" if sig["pts"] is not None else "N/A"
            lines.append(f"  ! {sig['name']}  ({pts_str} pts) — {sig['reasons']}")
    else:
        lines.append("\n✓ No sell signals in portfolio.")

    tail = lines
    portfolio_lines = portfolio_lines or []
    body = "\n".join(head + ([""] + portfolio_lines if portfolio_lines else []) + tail)

    def _pre(block):
        return ('<pre style="font-family:Menlo,Consolas,monospace;font-size:12px">'
                + html.escape("\n".join(block)) + "</pre>")

    html_body = ("<html><body>"
                 + (_pre(head[:buy_start]) + buy_html if buy_html else _pre(head))
                 + (portfolio_html or (_pre(portfolio_lines) if portfolio_lines else ""))
                 + _pre(tail) + "</body></html>")

    msg = MIMEMultipart("alternative")
    msg["From"]    = email_from
    msg["To"]      = email_to
    msg["Subject"] = f"Stock Update {date.today()}"
    # Order matters: clients show the last part they can render.
    msg.attach(MIMEText(body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

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
  python main.py --evaluate                    Is the score useful? Variants, index, weighting

--- Push results to Avanza ---
  python main.py --push                        Push top 10 to 'Bör köpa' (default)
  python main.py --push --push-to "Min lista"  Push to a different watchlist
  python main.py --push --push-top 5           Push top 5 instead of 10

Metrics are always weighted equally (correlation.equal_metric_weights).
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
        "--evaluate", "--optimize",
        action="store_true",
        dest="evaluate",
        help="Evaluate the score on data/panel_scores.csv: score variants vs "
             "the universe, vs OMXS30 GI / MSCI World, and equal vs min-var vs "
             "max-Sharpe weighting. Refreshes the index caches first (Avanza "
             "login for OMXS30 GI). Writes evaluation_results.json, which the "
             "email's status block reads. --optimize is the old name.",
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
             "ratio scales as IC x sqrt(breadth), so at IC +0.118 a 10-stock "
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
        "--prices-refresh-after",
        type=int, default=30, metavar="DAYS",
        help="Refetch a cached symbol when its series ends more than DAYS "
             "before today (default 30). Without this, --backfill-prices only "
             "ever downloaded symbols it had no file for, so the cache could "
             "not move forward and the newest fiscal year kept reading a "
             "truncated price series. A refresh refetches the FULL range and "
             "replaces the file rather than appending: Yahoo restates adjusted "
             "closes retroactively on every dividend, so a spliced tail would "
             "put a step the size of the dividend into the panel's forward "
             "returns. Use a large number to keep the old skip-if-present "
             "behavior, or 0 to refetch everything.",
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

    def _run_evaluate():
        """The whole of --evaluate. Errors are NOT swallowed: it is the only
        work this mode does, and a silent skip would look like a run that
        wrote nothing."""
        if not os.path.exists("data/panel_scores.csv"):
            print("[evaluate] data/panel_scores.csv not found — "
                  "run --backfill-panel first.")
            return 1
        from analyzer.benchmarks import WORLD_SYMBOL, fetch_omxs30gi
        from analyzer.evaluation import run_evaluation
        from analyzer.yahoo_prices import backfill_prices
        # Index caches first. Either failing leaves the cached copy (or n/a)
        # in place rather than stopping the evaluation.
        from analyzer.benchmarks import core_path, fetch_avanza_series
        try:
            av = setup_env()
            fetch_omxs30gi(av)
            for oid in CORE_MIX:
                fetch_avanza_series(av, oid, core_path(oid), f"core {oid}")
        except Exception as exc:
            print(f"[bench] Avanza index refresh failed ({exc}); using cache if any.")
        try:
            backfill_prices([WORLD_SYMBOL], start=args.prices_from,
                            refresh_after_days=args.prices_refresh_after)
        except Exception as exc:
            print(f"[bench] {WORLD_SYMBOL} refresh failed ({exc}); using cache if any.")
        run_evaluation()
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
            refresh_after_days=args.prices_refresh_after,
        )
        verified = verify_all(symbol_map)
        if result.get("refreshed"):
            print(f"[yahoo] refreshed {len(result['refreshed'])} stale "
                  "symbol(s) to today")
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

    # --evaluate without --save: runs against the already-built panel.
    if args.evaluate and not save_data:
        return _run_evaluate()

    # --- Live analysis ---
    manager = SummaryManager()

    # Equal metric weights and metrics.py's thresholds -- the same scoring the
    # panel (and so --evaluate) measures. The weight optimizer that used to
    # override this was removed; its gate never accepted a result.
    from analyzer.correlation import equal_metric_weights
    manager._weight_overrides = equal_metric_weights()

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

        if _is_excluded(ticker_id, ticker_info):
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

        fetched = _get_data_or_skip(ticker_id, ticker_info, manager, avanza,
                                    yahoo, save_data, skipped)
        if fetched is None:
            continue
        ticker_name, hist = fetched
        manager.price_ccy[ticker_name] = _listing_currency(ticker_info, yahoo_symbol)
        ki = ticker_info.get("keyIndicators") or {}
        manager.div_yield[ticker_name] = ki.get("directYield")
        manager.domicile[ticker_name] = str(ticker_info.get("isin") or "")[:2] or None
        manager.sector_group[ticker_name] = frontier.sector_group(ticker_info.get("sectors"))
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

    # --save combined with --evaluate: fresh snapshots were just written
    # above. Note the panel itself is NOT rebuilt here -- --backfill-panel is
    # still its own separate invocation.
    if args.evaluate:
        return _run_evaluate()

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

    # Always alongside --push + --email: weights for the list just pushed.
    # Every --push: weights for the list just pushed, mirrored onto the
    # min-var / max-sharpe watchlists, and shown in the email when sent.
    portfolio_lines = portfolio_html = None
    if push_results:
        portfolio_lines, portfolio_html = _portfolio_section(
            manager, push_results, avanza)
        print("\n".join(portfolio_lines) + "\n")

    if args.email:
        _send_email(push_results, sell_signals, portfolio_lines, portfolio_html)

    return 0


if __name__ == "__main__":

    result = main()
