"""SEK conversion for the panel's forward-return target.

Why this exists
---------------
44% of the universe is not SEK-listed (76 ``.ST`` against 44 US, 5 ``.DE``,
3 ``.HE``, 3 ``.OL``, 2 ``.CO``, 2 ``.L``), yet the panel's forward return was
computed in each stock's own listing currency and then demeaned within the
fiscal year across that mixed-currency cross-section. USD/SEK went 7.74 -> 9.56
over 2015-2026, so in any single year every US name carries the same FX
tailwind or headwind, and the within-year demeaning hands that shared move to
whatever metric happens to correlate with being US-listed. For a SEK-based
investor the numeraire is SEK: the realised return on a US stock is its USD
return compounded with the USD/SEK move. Converting prices to SEK *before* the
return is taken removes that shared component.

Rates come from Frankfurter (ECB daily reference rates) -- free, no API key.
Yahoo's ``USDSEK=X`` symbols raise ``'str' object has no attribute 'name'`` on
the pinned yfinance 0.2.54 regardless of session; do not swap this back.

The cache is a manual, Mac-only step exactly like ``--backfill-prices``: when
``data/fx_sek.csv`` is absent (fresh clone, Pi, offline) every caller must go
on working unconverted, so nothing here is allowed to raise into the live path.
"""

import json
import os
import urllib.request

import pandas as pd

FX_CACHE_PATH = "data/fx_sek.csv"
_API_URL = "https://api.frankfurter.dev/v1/{start}..{end}?base={cur}&symbols=SEK"
_USER_AGENT = "tech-analyze/1.0"

# Yahoo exchange suffix -> the currency that exchange quotes in.
SUFFIX_CURRENCY = {
    ".ST": "SEK",
    ".CO": "DKK",
    ".OL": "NOK",
    ".DE": "EUR",
    ".HE": "EUR",
    ".L": "GBP",
}
# A bare symbol is a US listing (see main.to_yahoo_symbol -- US is the only
# no-suffix passthrough it emits).
DEFAULT_CURRENCY = "USD"

# Currencies this module only ever sees quoted in *minor* units. Yahoo prices
# London-listed names in pence (GBp), not pounds: GLEN.L quotes ~400 where the
# stock is ~4 GBP. Applying GBPSEK to a pence series overstates it 100x.
# Scoped deliberately to the Yahoo leg, which is the only series routed through
# here -- Avanza quotes the same names in pounds (its GLEN close is 5.129), so
# an Avanza series must never be passed to `to_sek`.
_MINOR_UNITS = {"GBP": 100.0}

# The currencies worth caching: everything the suffix map can yield, minus SEK
# (self-conversion is a no-op, and Frankfurter has no SEK->SEK series).
DEFAULT_CURRENCIES = sorted(
    {DEFAULT_CURRENCY} | {c for c in SUFFIX_CURRENCY.values() if c != "SEK"}
)

# Slack allowed at the recent end when deciding a cached range is complete.
# ECB publishes business days only and around 16:00 CET, so a cache fetched
# "through today" legitimately ends a few days short over a weekend/holiday;
# without this every invocation would refetch.
_END_TOLERANCE_DAYS = 4


def currency_for_symbol(symbol):
    """Yahoo symbol -> quote currency, or None when the suffix is unknown.

    None is the deliberate answer for an unrecognised suffix: guessing a
    currency is how a 10x price error gets in silently, so callers leave those
    series unconverted and say so.
    """
    if not symbol:
        return None
    text = str(symbol).strip()
    if not text:
        return None
    if "." not in text:
        return DEFAULT_CURRENCY
    suffix = "." + text.rsplit(".", 1)[1].upper()
    return SUFFIX_CURRENCY.get(suffix)


def load_sek_rates(path=FX_CACHE_PATH):
    """The cached rate frame (date index, one column per currency), or None.

    None means "no cache" and is a normal state, not an error -- callers
    degrade to unconverted prices.
    """
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return None
    if df.empty:
        return None
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()
    return df if not df.empty else None


def _fetch_pair(currency, start, end):
    """One currency's SEK-per-unit series from Frankfurter."""
    url = _API_URL.format(start=start, end=end, cur=currency)
    # Frankfurter 403s urllib's default User-Agent (measured); any named agent
    # is accepted.
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)
    rates = payload.get("rates") or {}
    series = pd.Series(
        {
            pd.Timestamp(day): float(vals["SEK"])
            for day, vals in rates.items()
            if isinstance(vals, dict) and "SEK" in vals
        },
        dtype=float,
    )
    if series.empty:
        raise RuntimeError(f"no {currency}/SEK rates returned for {start}..{end}")
    return series.sort_index()


def _covers(cached, currencies, start, end):
    """Does ``cached`` already span [start, end] for every currency asked for?"""
    if cached is None:
        return False
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    for cur in currencies:
        if cur not in cached.columns:
            return False
        have = cached[cur].dropna()
        if have.empty:
            return False
        if have.index.min() > start:
            return False
        if have.index.max() < end - pd.Timedelta(days=_END_TOLERANCE_DAYS):
            return False
    return True


def fetch_sek_rates(start, end, path=FX_CACHE_PATH, currencies=None,
                    fetch_fn=None):
    """Fetch SEK rates for ``currencies`` over [start, end] and cache them.

    Idempotent: a cache that already spans the range for every currency is
    returned untouched, without a single request. Otherwise the fetched frame
    is merged *over* whatever is already on disk (``combine_first``), so a
    partial cache is widened rather than replaced.

    Values are SEK per 1 unit of the column's currency.
    """
    currencies = sorted(currencies or DEFAULT_CURRENCIES)
    cached = load_sek_rates(path)
    if _covers(cached, currencies, start, end):
        print(f"[fx] {path} already covers {start}..{end} for "
              f"{', '.join(currencies)} — nothing to fetch")
        return cached

    fetch_fn = fetch_fn or _fetch_pair
    columns = {}
    for cur in currencies:
        print(f"[fx] fetching {cur}/SEK {start}..{end}")
        columns[cur] = fetch_fn(cur, start, end)

    fresh = pd.DataFrame(columns).sort_index()
    fresh.index = pd.to_datetime(fresh.index)
    if cached is not None:
        fresh = fresh.combine_first(cached).sort_index()
    fresh.index.name = "date"

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fresh.to_csv(path)
    print(f"[fx] wrote {path}: {len(fresh)} days "
          f"{fresh.index.min().date()}..{fresh.index.max().date()}, "
          f"columns {', '.join(fresh.columns)}")
    return fresh


def to_sek(close_series, currency, rates):
    """``close_series`` restated in SEK.

    Three properties that are each a way to get this silently wrong:

    * **SEK is a no-op.** A ``.ST`` series is already SEK; multiplying it by
      anything is a bug, so it is returned unchanged (same object semantics as
      the input, no reindexing).
    * **Forward-fill only, never back-fill or interpolate.** ECB quotes
      business days, and exchanges keep different holidays, so a trading day
      routinely has no same-day rate. The last *published* rate is the only
      honest stand-in -- inferring one from a later publication would put
      tomorrow's FX into today's price.
    * **Dates before the rate series starts are dropped**, not pinned to the
      earliest rate. A missing return is a missing observation; a return
      computed against a rate from years later is a fabricated one.

    Raises ``KeyError`` when ``currency`` has no column, so a partial cache
    surfaces at the call site instead of quietly leaving one company in the
    wrong numeraire.
    """
    if close_series is None or len(close_series) == 0 or currency == "SEK":
        return close_series
    if rates is None or currency not in getattr(rates, "columns", ()):
        raise KeyError(f"no {currency}/SEK rates available")

    series = close_series.copy()
    series.index = pd.to_datetime(series.index, errors="coerce")
    series = series[series.index.notna()].sort_index()

    fx = pd.to_numeric(rates[currency], errors="coerce").dropna().sort_index()
    if fx.empty:
        raise KeyError(f"no {currency}/SEK rates available")

    aligned = (
        fx.reindex(fx.index.union(series.index))
        .ffill()
        .reindex(series.index)
    )
    converted = series / _MINOR_UNITS.get(currency, 1.0) * aligned
    # NaNs here are exactly the pre-series-start dates: ffill leaves the head
    # of the union empty and nothing else can be missing.
    return converted.dropna()
