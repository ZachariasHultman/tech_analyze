"""Backfill machinery: rate-limit survival and symbol verification.

Two failure modes matter here and neither is hypothetical.

Rate limiting: Yahoo's search endpoint rate-limits on the first call and the
limit then spills onto the bulk download endpoint (observed directly while
building this). A backfill that loses everything on the first wall is useless,
so the download must be batched, backed off and resumable.

Wrong symbol: a mis-resolved ticker does not error -- it quietly fills the
panel with a *different company's* returns, which is worse than missing data
because nothing downstream can detect it. Every symbol is therefore checked
against the Avanza prices already in the snapshots before use.
"""

import json

import numpy as np
import pandas as pd
import pytest

from analyzer import yahoo_prices as yp


# ------------------------------------------------------------------ fixtures
def _yf_frame(symbols, n=400, seed=0, scale=None):
    """A yf.download-shaped MultiIndex frame."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    cols, data = [], {}
    for s in symbols:
        base = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        base = base * (scale.get(s, 1.0) if scale else 1.0)
        data[("Close", s)] = base
        data[("Adj Close", s)] = base * 0.9
        cols += [("Close", s), ("Adj Close", s)]
    return pd.DataFrame(data, index=idx, columns=pd.MultiIndex.from_tuples(cols))


def _write_snapshot(tmp_path, key, closes, dates):
    ohlc = json.dumps([{"date": d.strftime("%Y-%m-%d"), "open": c, "high": c,
                        "low": c, "close": c, "totalVolumeTraded": 1}
                       for d, c in zip(dates, closes)])
    pd.DataFrame([{"asof": "2026-07-10", "ohlc": ohlc}]).to_csv(
        tmp_path / f"{key}_2026-07-10.csv", index=False)


# ------------------------------------------------------------- symbol keys
def test_orderbook_id_extraction():
    assert yp.company_orderbook_id("Volvo B 5269") == "5269"
    assert yp.company_orderbook_id("Novo Nordisk B  52300") == "52300"
    assert yp.company_orderbook_id("no-id-here") is None


def test_company_keys_ignores_panel_outputs(tmp_path):
    for name in ["ABB 5447_2026-07-10.csv", "ABB 5447_2026-02-18.csv",
                 "Volvo B 5269_2026-07-10.csv", "panel_scores.csv"]:
        (tmp_path / name).write_text("asof\n2026-07-10\n")
    assert yp.company_keys(str(tmp_path)) == ["ABB 5447", "Volvo B 5269"]


def test_symbol_map_round_trips(tmp_path):
    path = str(tmp_path / "syms.json")
    yp.save_symbol_map({"B 2": "B.ST", "A 1": "A.ST"}, path)
    assert yp.load_symbol_map(path) == {"A 1": "A.ST", "B 2": "B.ST"}
    assert yp.load_symbol_map(str(tmp_path / "missing.json")) == {}


def test_corrupt_symbol_map_is_not_fatal(tmp_path):
    path = tmp_path / "syms.json"
    path.write_text("{not json")
    assert yp.load_symbol_map(str(path)) == {}


# -------------------------------------------------------------- rate limits
class _FakeRateLimit(Exception):
    def __str__(self):
        return "Too Many Requests. Rate limited. Try after a while."


def test_backoff_retries_then_succeeds():
    slept, calls = [], {"n": 0}

    def downloader(symbols, **kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _FakeRateLimit()
        return _yf_frame(list(symbols))

    out = yp._download_batch(["A.ST"], "2023-01-01", "2024-01-01",
                             sleep_fn=slept.append, downloader=downloader)
    assert calls["n"] == 3
    assert slept == [30.0, 60.0], "expected exponential backoff"
    assert not out.empty


def test_backoff_gives_up_as_rate_limited():
    def downloader(symbols, **kw):
        raise _FakeRateLimit()

    with pytest.raises(yp.RateLimited):
        yp._download_batch(["A.ST"], "2023-01-01", "2024-01-01", max_retries=2,
                           sleep_fn=lambda s: None, downloader=downloader)


def test_non_rate_limit_errors_are_not_retried():
    calls = {"n": 0}

    def downloader(symbols, **kw):
        calls["n"] += 1
        raise ValueError("bad ticker")

    with pytest.raises(ValueError):
        yp._download_batch(["A.ST"], "2023-01-01", "2024-01-01",
                           sleep_fn=lambda s: None, downloader=downloader)
    assert calls["n"] == 1


def test_partial_progress_survives_a_rate_limit_wall(tmp_path):
    """The whole point of resumability: batch 1 must stay on disk."""
    seen = {"n": 0}

    def downloader(symbols, **kw):
        seen["n"] += 1
        if seen["n"] > 1:
            raise _FakeRateLimit()
        return _yf_frame(list(symbols))

    res = yp.backfill_prices(["A.ST", "B.ST", "C.ST", "D.ST"],
                             out_dir=str(tmp_path), batch_size=2, max_retries=2,
                             sleep_fn=lambda s: None, downloader=downloader)
    assert sorted(res["written"]) == ["A.ST", "B.ST"]
    assert sorted(res["rate_limited"]) == ["C.ST", "D.ST"]
    assert (tmp_path / "A.ST.csv").exists()


def test_resume_skips_what_is_already_downloaded(tmp_path):
    calls = []

    def downloader(symbols, **kw):
        calls.append(list(symbols))
        return _yf_frame(list(symbols))

    yp.backfill_prices(["A.ST", "B.ST"], out_dir=str(tmp_path), batch_size=2,
                       sleep_fn=lambda s: None, downloader=downloader)
    calls.clear()
    res = yp.backfill_prices(["A.ST", "B.ST", "C.ST"], out_dir=str(tmp_path),
                             batch_size=2, sleep_fn=lambda s: None,
                             downloader=downloader)
    assert res["skipped"] == 2
    assert calls == [["C.ST"]], "already-downloaded symbols must not be refetched"


def test_downloaded_file_keeps_both_adjusted_and_raw(tmp_path):
    yp.backfill_prices(["A.ST"], out_dir=str(tmp_path), batch_size=1,
                       sleep_fn=lambda s: None,
                       downloader=lambda s, **kw: _yf_frame(list(s)))
    df = yp.load_symbol_prices("A.ST", str(tmp_path))
    # raw is needed to verify identity against Avanza; adjusted is the return
    # basis. Losing either breaks a different half of the pipeline.
    assert list(df.columns) == ["close_adj", "close_raw"]
    assert (df["close_adj"] < df["close_raw"]).all()


# ------------------------------------------------------------ verification
def test_matching_series_verifies(tmp_path):
    prices_dir, data_dir = tmp_path / "p", tmp_path / "d"
    prices_dir.mkdir(); data_dir.mkdir()
    frame = _yf_frame(["A.ST"], n=300)
    yp.backfill_prices(["A.ST"], out_dir=str(prices_dir), batch_size=1,
                       sleep_fn=lambda s: None, downloader=lambda s, **kw: frame)
    raw = frame[("Close", "A.ST")]
    _write_snapshot(data_dir, "Alpha 1", raw.to_numpy(), raw.index)

    r = yp.verify_symbol("Alpha 1", "A.ST", str(data_dir), str(prices_dir))
    assert r["ok"], r
    assert r["agreement"] > 0.99


def test_a_different_company_is_rejected(tmp_path):
    """The failure this exists to catch: a plausible but wrong ticker."""
    prices_dir, data_dir = tmp_path / "p", tmp_path / "d"
    prices_dir.mkdir(); data_dir.mkdir()
    frame = _yf_frame(["WRONG.ST"], n=300, seed=1)
    yp.backfill_prices(["WRONG.ST"], out_dir=str(prices_dir), batch_size=1,
                       sleep_fn=lambda s: None, downloader=lambda s, **kw: frame)
    other = _yf_frame(["X"], n=300, seed=99)[("Close", "X")]
    _write_snapshot(data_dir, "Alpha 1", other.to_numpy(), other.index)

    r = yp.verify_symbol("Alpha 1", "WRONG.ST", str(data_dir), str(prices_dir))
    assert not r["ok"]
    assert "agreement" in r["reason"]


def test_wrong_share_class_is_rejected_on_level(tmp_path):
    """Co-moving but priced differently -- an A/B share-class mix-up."""
    prices_dir, data_dir = tmp_path / "p", tmp_path / "d"
    prices_dir.mkdir(); data_dir.mkdir()
    frame = _yf_frame(["A-A.ST"], n=300)
    yp.backfill_prices(["A-A.ST"], out_dir=str(prices_dir), batch_size=1,
                       sleep_fn=lambda s: None, downloader=lambda s, **kw: frame)
    raw = frame[("Close", "A-A.ST")]
    _write_snapshot(data_dir, "Alpha 1", (raw * 1.15).to_numpy(), raw.index)

    r = yp.verify_symbol("Alpha 1", "A-A.ST", str(data_dir), str(prices_dir))
    assert not r["ok"]
    assert "level diff" in r["reason"]


def test_short_overlap_is_rejected(tmp_path):
    prices_dir, data_dir = tmp_path / "p", tmp_path / "d"
    prices_dir.mkdir(); data_dir.mkdir()
    frame = _yf_frame(["A.ST"], n=300)
    yp.backfill_prices(["A.ST"], out_dir=str(prices_dir), batch_size=1,
                       sleep_fn=lambda s: None, downloader=lambda s, **kw: frame)
    raw = frame[("Close", "A.ST")].iloc[:20]
    _write_snapshot(data_dir, "Alpha 1", raw.to_numpy(), raw.index)

    r = yp.verify_symbol("Alpha 1", "A.ST", str(data_dir), str(prices_dir))
    assert not r["ok"] and "overlap" in r["reason"]


def test_unverified_company_is_absent_not_wrong(tmp_path):
    """A rejected company must fall back to Avanza, never adopt bad prices."""
    prices_dir, data_dir = tmp_path / "p", tmp_path / "d"
    prices_dir.mkdir(); data_dir.mkdir()
    frame = _yf_frame(["GOOD.ST", "BAD.ST"], n=300)
    yp.backfill_prices(["GOOD.ST", "BAD.ST"], out_dir=str(prices_dir),
                       batch_size=2, sleep_fn=lambda s: None,
                       downloader=lambda s, **kw: frame)
    good = frame[("Close", "GOOD.ST")]
    _write_snapshot(data_dir, "Good 1", good.to_numpy(), good.index)
    unrelated = _yf_frame(["Z"], n=300, seed=42)[("Close", "Z")]
    _write_snapshot(data_dir, "Bad 2", unrelated.to_numpy(), unrelated.index)

    closes, result = yp.load_verified_closes(
        {"Good 1": "GOOD.ST", "Bad 2": "BAD.ST"}, str(data_dir), str(prices_dir))
    assert "Good 1" in closes
    assert "Bad 2" not in closes
    assert len(result["rejected"]) == 1


def test_missing_download_is_reported_not_crashed(tmp_path):
    r = yp.verify_symbol("Alpha 1", "NOPE.ST", str(tmp_path), str(tmp_path))
    assert not r["ok"] and r["reason"] == "no downloaded prices"


def test_one_day_date_offset_still_verifies(tmp_path):
    """Avanza and Yahoo date the same bar one trading day apart.

    Measured on every real company checked: yahoo[D] holds the price Avanza
    labels D-1, giving level correlation 0.999 but return correlation 0.15 at
    lag 0. A lag-0-only verifier rejects 30 of 30 correct symbols.
    """
    prices_dir, data_dir = tmp_path / "p", tmp_path / "d"
    prices_dir.mkdir(); data_dir.mkdir()
    frame = _yf_frame(["A.ST"], n=300)
    yp.backfill_prices(["A.ST"], out_dir=str(prices_dir), batch_size=1,
                       sleep_fn=lambda s: None, downloader=lambda s, **kw: frame)
    raw = frame[("Close", "A.ST")]
    # Avanza sees the same prices, dated one day earlier.
    _write_snapshot(data_dir, "Alpha 1", raw.to_numpy(),
                    raw.index - pd.Timedelta(days=1))

    r = yp.verify_symbol("Alpha 1", "A.ST", str(data_dir), str(prices_dir))
    assert r["ok"], r
    assert r["lag"] == -1
    assert r["agreement"] > 0.99


def test_lag_search_does_not_rescue_a_wrong_company(tmp_path):
    """Tolerating a one-day shift must not become tolerating anything."""
    prices_dir, data_dir = tmp_path / "p", tmp_path / "d"
    prices_dir.mkdir(); data_dir.mkdir()
    frame = _yf_frame(["WRONG.ST"], n=300, seed=3)
    yp.backfill_prices(["WRONG.ST"], out_dir=str(prices_dir), batch_size=1,
                       sleep_fn=lambda s: None, downloader=lambda s, **kw: frame)
    other = _yf_frame(["X"], n=300, seed=77)[("Close", "X")]
    _write_snapshot(data_dir, "Alpha 1", other.to_numpy(), other.index)

    r = yp.verify_symbol("Alpha 1", "WRONG.ST", str(data_dir), str(prices_dir))
    assert not r["ok"]


def test_isolated_corporate_action_days_do_not_reject(tmp_path):
    """Real case: Hexagon, Tele2 and Electrolux each match Avanza on 99.5%+ of
    1257 days, but two corporate-action days apiece drop Pearson to 0.82-0.90.
    Verification must survive that without loosening enough to admit a
    genuinely different company."""
    prices_dir, data_dir = tmp_path / "p", tmp_path / "d"
    prices_dir.mkdir(); data_dir.mkdir()
    frame = _yf_frame(["A.ST"], n=400)
    yp.backfill_prices(["A.ST"], out_dir=str(prices_dir), batch_size=1,
                       sleep_fn=lambda s: None, downloader=lambda s, **kw: frame)
    closes = frame[("Close", "A.ST")].to_numpy().copy()
    closes[200:] *= 1.45   # one unadjusted corporate action, everything else identical
    _write_snapshot(data_dir, "Alpha 1", closes, frame.index)

    r = yp.verify_symbol("Alpha 1", "A.ST", str(data_dir), str(prices_dir))
    assert r["agreement"] > 0.99
    assert r["corr"] < 0.95, "Pearson should indeed be wrecked by that one day"
    # rejected on the level check, not the (robust) return check
    assert "level diff" in r["reason"]


def test_one_walled_batch_does_not_abandon_the_rest(tmp_path):
    """A single rate-limited batch must not cost the batches after it.

    135 symbols at batch size 8 is 17 batches; ending the run on the first
    wall would leave everything downstream unattempted even when the block was
    transient.
    """
    def downloader(symbols, **kw):
        if "B.ST" in symbols:
            raise _FakeRateLimit()
        return _yf_frame(list(symbols))

    res = yp.backfill_prices(["A.ST", "B.ST", "C.ST", "D.ST", "E.ST", "F.ST"],
                             out_dir=str(tmp_path), batch_size=2, max_retries=1,
                             sleep_fn=lambda s: None, downloader=downloader)
    # A.ST shares the walled batch with B.ST, so it is reported missing too --
    # the next run picks both up. The point is that C-F still got fetched.
    assert sorted(res["written"]) == ["C.ST", "D.ST", "E.ST", "F.ST"]
    assert res["rate_limited"] == ["A.ST", "B.ST"]
    assert not res["aborted"]


def test_sustained_walls_abort_rather_than_grind(tmp_path):
    """Consecutive walls mean wholesale blocking, not throttling -- stop."""
    def downloader(symbols, **kw):
        raise _FakeRateLimit()

    res = yp.backfill_prices([f"S{i}.ST" for i in range(20)],
                             out_dir=str(tmp_path), batch_size=2, max_retries=1,
                             max_consecutive_walls=3, sleep_fn=lambda s: None,
                             downloader=downloader)
    assert res["aborted"]
    assert len(res["rate_limited"]) == 6, "should stop after 3 walled batches"
    assert res["written"] == []


def test_wall_then_recovery_resets_the_counter(tmp_path):
    """Intermittent throttling must not accumulate toward the abort limit."""
    calls = {"n": 0}

    def downloader(symbols, **kw):
        calls["n"] += 1
        if calls["n"] % 2 == 1:          # every other batch walls
            raise _FakeRateLimit()
        return _yf_frame(list(symbols))

    res = yp.backfill_prices([f"S{i}.ST" for i in range(12)],
                             out_dir=str(tmp_path), batch_size=2, max_retries=1,
                             max_consecutive_walls=3, sleep_fn=lambda s: None,
                             downloader=downloader)
    assert not res["aborted"]
    assert len(res["written"]) == 6


def test_cooldown_is_slept_between_walled_batches(tmp_path):
    slept = []

    def downloader(symbols, **kw):
        if "A.ST" in symbols:
            raise _FakeRateLimit()
        return _yf_frame(list(symbols))

    yp.backfill_prices(["A.ST", "B.ST"], out_dir=str(tmp_path), batch_size=1,
                       max_retries=1, cooldown=600.0, sleep_fn=slept.append,
                       downloader=downloader)
    assert 600.0 in slept, "a walled batch should trigger the cooldown"


def test_rerun_picks_up_only_what_is_missing(tmp_path):
    """The recovery path: what a walled run leaves behind is exactly what the
    next invocation fetches."""
    state = {"wall": True}

    def downloader(symbols, **kw):
        if state["wall"] and "B.ST" in symbols:
            raise _FakeRateLimit()
        return _yf_frame(list(symbols))

    first = yp.backfill_prices(["A.ST", "B.ST", "C.ST"], out_dir=str(tmp_path),
                               batch_size=1, max_retries=1,
                               sleep_fn=lambda s: None, downloader=downloader)
    assert first["rate_limited"] == ["B.ST"]

    state["wall"] = False
    second = yp.backfill_prices(["A.ST", "B.ST", "C.ST"], out_dir=str(tmp_path),
                                batch_size=1, max_retries=1,
                                sleep_fn=lambda s: None, downloader=downloader)
    assert second["written"] == ["B.ST"]
    assert second["skipped"] == 2
