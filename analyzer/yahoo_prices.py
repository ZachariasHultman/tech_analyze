"""Dividend-adjusted price history from Yahoo, as a one-time offline backfill.

Why this exists
---------------
Avanza serves a *rolling* ~5-year OHLC window and its closes are not
dividend-adjusted. Two consequences, both measured:

* Only 4 fiscal years have a usable 1-year forward return, though fundamentals
  reach back to 2019 for 87 companies and 2020 for 115. The price window, not
  the financials, is what caps the panel -- and it slides forward rather than
  accumulating, so re-saving never deepens it.
* Dividends are missing from the target. Handelsbanken A drops ~12.5% on its
  ex-div day in the raw snapshot; Yahoo's adjusted close moves ~1.2%.

Yahoo's ``auto_adjust`` close fixes both at once and reaches back decades. The
power calculation says 7 fiscal years is where this system's observed ICIR
would become distinguishable from noise; this is how 4 becomes 7.

Rate limits are the whole design problem
----------------------------------------
Yahoo's search/quote endpoints rate-limit almost immediately and, once tripped,
the limit spills onto the bulk download endpoint too (observed directly). So:

* **No search, ever.** Symbols come from Avanza's own listing metadata via
  ``to_yahoo_symbol`` and are cached in a hand-editable JSON map.
* **Small batches, backoff, and resumability.** Each symbol's series is written
  to its own file the moment it arrives, so hitting a cap costs the current
  batch rather than the run. Re-running resumes.
* **Every symbol is verified, not trusted.** A candidate's *unadjusted* close is
  correlated against the Avanza close already in the snapshots. That catches a
  wrong share class (VOLV-A vs VOLV-B), a wrong exchange suffix, and tickers
  that have been reused -- failures that would otherwise silently pollute the
  panel with another company's returns.

This module is Mac-only and manual. The Pi never calls it.
"""

import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd

SYMBOLS_PATH = "data/yahoo_symbols.json"
PRICES_DIR = "data/yahoo_prices"

# Verification bars.
#
# The primary test is the *fraction of days whose daily returns agree*, not
# their Pearson correlation. Correlation looked like the obvious choice and is
# far too fragile: Hexagon, Tele2 and Electrolux each agree with Avanza on
# 99.5-99.8% of 1257 days, yet two corporate-action days apiece (where one
# source adjusts and the other does not) drag Pearson down to 0.82-0.90. That
# would reject three correct symbols and silently cost their price history.
#
# A genuinely wrong company fails this test overwhelmingly -- two unrelated
# series at ~1% daily vol agree within 0.5% on roughly a quarter of days, far
# under the bar -- so robustness here does not buy laxity.
_RETURN_AGREE_TOL = 0.005
_MIN_RETURN_AGREEMENT = 0.95
_MAX_MEDIAN_REL_DIFF = 0.02
_MIN_OVERLAP_DAYS = 100


# ---------------------------------------------------------------- symbol map
def company_orderbook_id(company_key):
    """'Volvo B 5269' -> '5269'. Returns None when the key has no trailing id."""
    parts = str(company_key).rsplit(" ", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[1]
    return None


def company_keys(data_dir="data"):
    """Every company key present as a snapshot in ``data_dir``."""
    keys = set()
    for path in glob.glob(os.path.join(data_dir, "*.csv")):
        name = os.path.basename(path)
        if name.startswith("panel_"):
            continue
        m = re.match(r"(.+)_(\d{4}-\d{2}-\d{2})\.csv$", name)
        if m:
            keys.add(m.group(1))
    return sorted(keys)


def load_symbol_map(path=SYMBOLS_PATH):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_symbol_map(mapping, path=SYMBOLS_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(sorted(mapping.items())), f, indent=2, ensure_ascii=False)
    return path


def resolve_symbols_via_avanza(avanza, keys, existing=None):
    """Map company keys to Yahoo symbols using Avanza's listing metadata.

    Deliberately reuses ``main.to_yahoo_symbol`` rather than guessing from the
    company name: it already encodes the exchange-suffix rules (and which
    countries are *not* safe to pass through), and a wrong guess here would be
    silently absorbed as another company's price history.

    Entries already present in ``existing`` are kept, so a hand-corrected map
    survives re-resolution.
    """
    from analyzer.main import to_yahoo_symbol

    resolved = dict(existing or {})
    unresolved = []
    for key in keys:
        if resolved.get(key):
            continue
        oid = company_orderbook_id(key)
        if oid is None:
            unresolved.append((key, "no orderbook id in key"))
            continue
        try:
            info = avanza.get_stock_info(oid)
            symbol = to_yahoo_symbol(info)
        except Exception as exc:
            unresolved.append((key, f"avanza lookup failed: {exc}"))
            continue
        if symbol:
            resolved[key] = symbol
        else:
            unresolved.append((key, "unsupported country / missing ticker"))
    return resolved, unresolved


# ------------------------------------------------------------------ download
class RateLimited(RuntimeError):
    pass


def _is_rate_limit(exc):
    return "rate" in str(exc).lower() and "limit" in str(exc).lower()


def _impersonating_session():
    """A ``curl_cffi`` session that presents as a real browser, or None.

    Yahoo throttles on TLS/HTTP fingerprint, not just request volume: plain
    ``yf.download`` gets ``YFRateLimitError`` within a handful of calls and
    the block then persists for many minutes, regardless of batching or
    backoff (measured -- 45s/90s/180s backoff never cleared it). The same
    request through a Chrome-impersonating session returns 11+ years for
    several tickers immediately.

    Returned as best-effort: if ``curl_cffi`` is missing the caller falls back
    to plain yfinance and the backoff path still applies.
    """
    try:
        from curl_cffi import requests as curl_requests
    except Exception:
        return None
    try:
        return curl_requests.Session(impersonate="chrome")
    except Exception:
        return None


def _download_batch(symbols, start, end, max_retries=6, base_sleep=30.0,
                    sleep_fn=time.sleep, downloader=None):
    """One ``yf.download`` call with exponential backoff on rate limiting.

    Returns a DataFrame with a column MultiIndex, or raises RateLimited after
    exhausting retries -- the caller keeps whatever earlier batches already
    landed on disk rather than losing the run.
    """
    kwargs = {}
    if downloader is None:
        import yfinance as yf
        downloader = yf.download
        session = _impersonating_session()
        if session is not None:
            kwargs["session"] = session

    delay = base_sleep
    last_exc = None
    for attempt in range(max_retries):
        try:
            data = downloader(
                symbols, start=start, end=end, auto_adjust=False,
                progress=False, threads=False, **kwargs,
            )
            if data is not None and not data.empty and "Close" in data:
                closes = data["Close"]
                if closes.notna().any().any():
                    return data
            last_exc = RuntimeError("empty response")
        except Exception as exc:  # noqa: BLE001 - yfinance raises many types
            last_exc = exc
            if not _is_rate_limit(exc):
                raise
        if attempt < max_retries - 1:
            print(f"  [yahoo] rate limited / empty; sleeping {delay:.0f}s "
                  f"(attempt {attempt + 1}/{max_retries})")
            sleep_fn(delay)
            delay = min(delay * 2, 900.0)
    raise RateLimited(f"gave up after {max_retries} attempts: {last_exc}")


def _price_path(symbol, out_dir=PRICES_DIR):
    safe = symbol.replace("/", "_")
    return os.path.join(out_dir, f"{safe}.csv")


def backfill_prices(symbols, start="2015-01-01", end=None, out_dir=PRICES_DIR,
                    batch_size=8, pause=3.0, resume=True, max_retries=6,
                    base_sleep=30.0, cooldown=600.0, max_consecutive_walls=3,
                    sleep_fn=time.sleep, downloader=None):
    """Download adjusted + raw closes for ``symbols``, one file per symbol.

    Resumable by design: each symbol is written the moment its batch returns
    and ``resume=True`` skips symbols already on disk, so nothing downloaded is
    ever re-fetched or lost.

    Rate-limit handling has two layers, because a wall is not always fatal:

    * Within a batch, ``_download_batch`` backs off exponentially (30s doubling
      to a 900s cap, ~15 min total).
    * Across batches, a batch that exhausts its retries is *skipped*, not
      fatal: the run sleeps ``cooldown`` and moves to the next batch. Only
      ``max_consecutive_walls`` batches failing back to back ends the run, on
      the reasoning that Yahoo is then blocking wholesale rather than
      throttling. Skipped symbols are simply picked up by the next invocation.

    Note the empirical caveat: against plain ``yfinance`` the block never
    cleared within 45s/90s/180s of backoff. Waiting alone is not a reliable
    remedy -- ``_impersonating_session`` is what actually avoids the limit, and
    this layer is the fallback for when it is unavailable.
    """
    os.makedirs(out_dir, exist_ok=True)
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    symbols = sorted(set(s for s in symbols if s))

    todo = [s for s in symbols
            if not (resume and os.path.exists(_price_path(s, out_dir)))]
    skipped = len(symbols) - len(todo)
    if skipped:
        print(f"  [yahoo] {skipped} symbol(s) already on disk, skipping")

    n_batches = (len(todo) + batch_size - 1) // batch_size
    written, failed, walled = [], [], []
    consecutive_walls = 0
    aborted = False

    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        print(f"  [yahoo] batch {i // batch_size + 1}/{n_batches}: "
              f"{', '.join(batch)}")
        try:
            data = _download_batch(batch, start, end, max_retries=max_retries,
                                   base_sleep=base_sleep, sleep_fn=sleep_fn,
                                   downloader=downloader)
        except RateLimited as exc:
            consecutive_walls += 1
            walled.extend(batch)
            print(f"  [yahoo] batch rate-limited ({exc})")
            if consecutive_walls >= max_consecutive_walls:
                print(f"  [yahoo] {consecutive_walls} batches walled in a row — "
                      "Yahoo is blocking wholesale, not throttling. Stopping.")
                aborted = True
                break
            if i + batch_size < len(todo):
                print(f"  [yahoo] cooling down {cooldown:.0f}s, then continuing "
                      "with the next batch (this one resumes on re-run)")
                sleep_fn(cooldown)
            continue

        consecutive_walls = 0
        for symbol in batch:
            frame = _extract_symbol(data, symbol, len(batch))
            if frame is None or frame.empty:
                failed.append(symbol)
                continue
            frame.to_csv(_price_path(symbol, out_dir))
            written.append(symbol)
        if i + batch_size < len(todo):
            sleep_fn(pause)

    remaining = sorted(set(walled) | set(failed))
    print(f"  [yahoo] wrote {len(written)}, rate-limited {len(walled)}, "
          f"empty/failed {len(failed)}")
    if remaining:
        print(f"  [yahoo] {len(remaining)} symbol(s) still missing — re-run "
              "--backfill-prices to pick them up (already-downloaded symbols "
              "are skipped).")
    if aborted:
        print("  [yahoo] tip: if this repeats, install curl_cffi (a dependency "
              "of this project) — Yahoo throttles on TLS fingerprint and a "
              "browser-impersonating session sidesteps the limit entirely.")
    return {"written": written, "failed": failed, "rate_limited": walled,
            "skipped": skipped, "aborted": aborted}


def _extract_symbol(data, symbol, batch_len):
    """Pull (close_adj, close_raw) for one symbol out of a yf.download frame."""
    try:
        if isinstance(data.columns, pd.MultiIndex):
            raw = data["Close"][symbol]
            adj = (data["Adj Close"][symbol]
                   if "Adj Close" in data.columns.get_level_values(0)
                   else raw)
        else:
            raw = data["Close"]
            adj = data["Adj Close"] if "Adj Close" in data.columns else raw
    except Exception:
        return None
    out = pd.DataFrame({"close_adj": adj, "close_raw": raw}).dropna(how="all")
    out.index.name = "date"
    return out


# -------------------------------------------------------------- verification
def _avanza_close(company_key, data_dir="data"):
    """Latest snapshot's Avanza close series for one company."""
    files = sorted(glob.glob(os.path.join(data_dir, f"{glob.escape(company_key)}_*.csv")))
    if not files:
        return pd.Series(dtype=float)
    try:
        raw = pd.read_csv(files[-1], usecols=["ohlc"])["ohlc"].iloc[0]
        ohlc = pd.DataFrame(json.loads(raw))
        ohlc["date"] = pd.to_datetime(ohlc["date"], errors="coerce")
        return ohlc.dropna(subset=["date"]).set_index("date")["close"].sort_index()
    except Exception:
        return pd.Series(dtype=float)


def load_symbol_prices(symbol, out_dir=PRICES_DIR):
    path = _price_path(symbol, out_dir)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col="date", parse_dates=True)
    except Exception:
        return None
    return df.sort_index() if not df.empty else None


def verify_symbol(company_key, symbol, data_dir="data", out_dir=PRICES_DIR):
    """Is this Yahoo series really this company? Checked, not assumed.

    Correlates *unadjusted* Yahoo closes against Avanza's (also unadjusted) over
    their overlap. Daily-return correlation is the primary test; a median
    relative level difference guards against a look-alike that merely co-moves.
    """
    prices = load_symbol_prices(symbol, out_dir)
    if prices is None or "close_raw" not in prices.columns:
        return {"company": company_key, "symbol": symbol, "ok": False,
                "reason": "no downloaded prices", "corr": None, "rel_diff": None,
                "overlap": 0}

    avanza = _avanza_close(company_key, data_dir)
    yahoo = prices["close_raw"]

    # Avanza and Yahoo disagree by one trading day on how a bar is dated: for
    # every company checked, yahoo[D] carries the price Avanza labels D-1
    # (level correlation 0.999, return correlation 0.15 at lag 0). Testing only
    # lag 0 would therefore reject every correct symbol. Search a +/-1 day
    # window and keep the best; the offset itself is harmless downstream
    # because the Yahoo series is used self-consistently for both the report
    # anchor and the +1y anchor, and a one-day-late label makes the anchor
    # price marginally stale rather than forward-looking.
    best = {"agree": -np.inf, "corr": None, "rel": None, "overlap": 0, "lag": 0}
    for lag in (0, -1, 1):
        shifted = yahoo.copy()
        shifted.index = shifted.index + pd.Timedelta(days=lag)
        joined = pd.concat(
            [avanza.rename("a"), shifted.rename("y")], axis=1
        ).dropna()
        if len(joined) < _MIN_OVERLAP_DAYS:
            best["overlap"] = max(best["overlap"], len(joined))
            continue
        diff = (joined["a"].pct_change() - joined["y"].pct_change()).abs().dropna()
        if diff.empty:
            continue
        agree = float((diff < _RETURN_AGREE_TOL).mean())
        if agree > best["agree"]:
            best = {
                "agree": agree,
                "corr": float(joined["a"].pct_change().corr(joined["y"].pct_change())),
                "rel": float(np.median(np.abs(joined["a"] / joined["y"] - 1.0))),
                "overlap": len(joined),
                "lag": lag,
            }

    if best["rel"] is None:
        return {"company": company_key, "symbol": symbol, "ok": False,
                "reason": f"overlap {best['overlap']}d < {_MIN_OVERLAP_DAYS}d",
                "agreement": None, "corr": None, "rel_diff": None,
                "overlap": best["overlap"], "lag": 0}

    agree, rel = best["agree"], best["rel"]
    ok = bool(agree >= _MIN_RETURN_AGREEMENT and rel <= _MAX_MEDIAN_REL_DIFF)
    reason = "ok" if ok else (
        f"daily-return agreement {agree:.1%} < {_MIN_RETURN_AGREEMENT:.0%}"
        if agree < _MIN_RETURN_AGREEMENT
        else f"median level diff {rel:.3f} > {_MAX_MEDIAN_REL_DIFF}"
    )
    return {"company": company_key, "symbol": symbol, "ok": ok, "reason": reason,
            "agreement": agree, "corr": best["corr"], "rel_diff": rel,
            "overlap": best["overlap"], "lag": best["lag"]}


def verify_all(symbol_map, data_dir="data", out_dir=PRICES_DIR, verbose=True):
    reports = [verify_symbol(k, s, data_dir, out_dir)
               for k, s in sorted(symbol_map.items())]
    good = [r for r in reports if r["ok"]]
    bad = [r for r in reports if not r["ok"]]
    if verbose:
        print(f"\n[yahoo] verification: {len(good)} verified, {len(bad)} rejected")
        for r in bad:
            print(f"  [REJECT] {r['company']:34s} {r['symbol']:12s} {r['reason']}")
    return {"verified": {r["company"]: r["symbol"] for r in good},
            "rejected": bad, "reports": reports}


def load_verified_closes(symbol_map, data_dir="data", out_dir=PRICES_DIR,
                         verbose=False):
    """{company_key: adjusted close Series} for every symbol that verifies.

    A company that fails verification is simply absent, so callers fall back to
    the Avanza series rather than silently adopting the wrong company's prices.
    """
    result = verify_all(symbol_map, data_dir, out_dir, verbose=verbose)
    closes = {}
    for company, symbol in result["verified"].items():
        prices = load_symbol_prices(symbol, out_dir)
        if prices is None or "close_adj" not in prices.columns:
            continue
        series = prices["close_adj"].dropna()
        if not series.empty:
            closes[company] = series
    return closes, result
