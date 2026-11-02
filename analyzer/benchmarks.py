"""Index series the evaluation compares the screener against.

Two benchmarks, both total return and both restated in SEK, because the
target they are compared with (``fwd_total_return_1y``) is total return in
SEK:

* **OMXS30 GI** from Avanza (orderbook 1002994). The GI variant reinvests
  dividends; the plain OMXS30 (19002) is a price index and would hand the
  screener ~3%/yr for free. Avanza only serves ~5 years of daily index data
  (``ALL_TIME`` is a 400), so the cache is *merged* on every fetch and grows
  forward from 2021-11. Merging is safe here, unlike the Yahoo adjusted closes
  in yahoo_prices.py: an index level is never restated retroactively.
  Yahoo's ``^OMXS30GI`` carries no history at all (measured), so it is not an
  alternative.
* **MSCI World** via the ``URTH`` ETF on Yahoo -- adjusted close, so total
  return, converted USD->SEK. Downloaded by ``--backfill-prices`` alongside
  the stock series, into the same cache.

Mac-only and manual, like the rest of the price caches: the evaluation must
work without either file and report "n/a" for that comparison.
"""

import os

import pandas as pd

BENCH_DIR = "data/benchmarks"
OMXS30GI_ID = "1002994"
OMXS30GI_PATH = os.path.join(BENCH_DIR, "OMXS30GI.csv")
WORLD_SYMBOL = "URTH"

OMXS30_LABEL = "OMXS30 GI"
WORLD_LABEL = "World (URTH)"


def fetch_omxs30gi(avanza, path=OMXS30GI_PATH):
    """Fetch Avanza's 5-year daily OMXS30 GI and merge it into the cache.

    Returns the merged series. Fresh values win on overlapping dates.
    """
    from avanza.avanza import Resolution, TimePeriod

    raw = avanza.get_chart_data(order_book_id=OMXS30GI_ID,
                                period=TimePeriod.FIVE_YEARS,
                                resolution=Resolution.DAY)["ohlc"]
    fresh = pd.Series(
        [float(e["close"]) for e in raw],
        index=pd.to_datetime([e["timestamp"] for e in raw], unit="ms").normalize(),
        name="close",
    ).sort_index()
    fresh = fresh[~fresh.index.duplicated(keep="last")]
    cached = _load_series(path)
    merged = fresh.combine_first(cached) if cached is not None else fresh
    os.makedirs(os.path.dirname(path), exist_ok=True)
    merged.rename_axis("date").to_frame("close").to_csv(path)
    print(f"[bench] {OMXS30_LABEL}: {len(merged)} days "
          f"{merged.index.min().date()}..{merged.index.max().date()} -> {path}")
    return merged


def _load_series(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return None
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().sort_index()
    return s if not s.empty else None


def load_benchmarks(rates=None, omx_path=OMXS30GI_PATH):
    """``{label: SEK close series}`` for whichever benchmarks are cached."""
    out = {}
    omx = _load_series(omx_path)
    if omx is not None:
        out[OMXS30_LABEL] = omx
    try:
        from analyzer.fx import load_sek_rates, to_sek
        from analyzer.yahoo_prices import load_symbol_prices
        world = load_symbol_prices(WORLD_SYMBOL)
        if world is not None and "close_adj" in world:
            rates = rates if rates is not None else load_sek_rates()
            if rates is not None:
                out[WORLD_LABEL] = to_sek(world["close_adj"].dropna(), "USD", rates)
    except Exception as exc:
        print(f"[bench] {WORLD_LABEL} unavailable ({exc})")
    return out


def window_return(series, start, end):
    """Return over ``(start, end]`` from the last close on or before each
    date, or None when the series does not cover the whole window."""
    if series is None or series.empty:
        return None
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if series.index.min() > start or series.index.max() < end:
        return None
    p0 = series[series.index <= start]
    p1 = series[series.index <= end]
    if p0.empty or p1.empty or p0.iloc[-1] <= 0:
        return None
    return float(p1.iloc[-1] / p0.iloc[-1] - 1.0)
