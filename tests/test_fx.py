"""Tests for analyzer/fx.py — SEK conversion of the panel's return target.

No network: every ``fetch_sek_rates`` call is given a stub ``fetch_fn``. The
cases below are the five ways this can be silently wrong (pence, converting
SEK, back-filling a rate, inventing a pre-series-start rate, and hard-failing
when the cache is absent), not a tour of the happy path.
"""
import math

import pandas as pd
import pytest

import analyzer.fx as fx


def _rates(rows, columns=("USD",)):
    """rows: {date: (v1, v2, ...)} -> a cache-shaped frame."""
    idx = pd.to_datetime(list(rows))
    return pd.DataFrame(list(rows.values()), index=idx, columns=list(columns))


def _closes(pairs):
    return pd.Series(
        [v for _, v in pairs], index=pd.to_datetime([d for d, _ in pairs])
    )


# ---------------------------------------------------------- symbol -> currency
@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("VOLV-B.ST", "SEK"),
        ("NOVO-B.CO", "DKK"),
        ("EQNR.OL", "NOK"),
        ("SIE.DE", "EUR"),
        ("NOKIA.HE", "EUR"),
        ("GLEN.L", "GBP"),
        ("AAPL", "USD"),
        ("BRK-B", "USD"),
    ],
)
def test_currency_for_symbol(symbol, expected):
    assert fx.currency_for_symbol(symbol) == expected


@pytest.mark.parametrize("symbol", [None, "", "  ", "FOO.XYZ"])
def test_currency_for_unknown_symbol_is_none_not_a_guess(symbol):
    # A guessed currency is a silent 10x price error; None makes the caller
    # leave the series alone and say so.
    assert fx.currency_for_symbol(symbol) is None


# ------------------------------------------------------------------- to_sek
def test_sek_series_is_returned_untouched():
    ser = _closes([("2024-01-02", 100.0), ("2024-01-03", 101.0)])
    out = fx.to_sek(ser, "SEK", _rates({"2024-01-02": (10.0,)}))
    pd.testing.assert_series_equal(out, ser)


def test_usd_series_is_multiplied_by_the_same_day_rate():
    ser = _closes([("2024-01-02", 100.0), ("2024-01-03", 100.0)])
    rates = _rates({"2024-01-02": (10.0,), "2024-01-03": (11.0,)})
    out = fx.to_sek(ser, "USD", rates)
    assert list(out) == [1000.0, 1100.0]
    # the FX move alone is the whole return, which is the point of the fix
    assert math.isclose(out.iloc[-1] / out.iloc[0] - 1.0, 0.10)


def test_london_prices_are_pence_and_get_divided_by_100():
    # GLEN.L quotes ~400 for a ~4 GBP stock. Without the /100 this is 100x.
    ser = _closes([("2024-01-02", 400.0)])
    out = fx.to_sek(ser, "GBP", _rates({"2024-01-02": (13.0,)}, columns=("GBP",)))
    assert math.isclose(out.iloc[0], 4.0 * 13.0)


def test_missing_rate_is_forward_filled_never_back_filled():
    # 2024-01-03 is a market holiday for the ECB but a trading day here.
    ser = _closes([("2024-01-02", 100.0), ("2024-01-03", 100.0),
                   ("2024-01-04", 100.0)])
    rates = _rates({"2024-01-02": (10.0,), "2024-01-04": (12.0,)})
    out = fx.to_sek(ser, "USD", rates)
    # the gap day takes the last PUBLISHED rate (10.0), not the next one (12.0)
    assert list(out) == [1000.0, 1000.0, 1200.0]


def test_dates_before_the_rate_series_starts_are_dropped():
    ser = _closes([("2013-06-03", 100.0), ("2024-01-02", 100.0)])
    out = fx.to_sek(ser, "USD", _rates({"2024-01-02": (10.0,)}))
    # dropped, not pinned to the earliest rate — a return computed against a
    # rate published years later is fabricated, not merely stale.
    assert list(out.index) == [pd.Timestamp("2024-01-02")]


def test_pre_start_drop_kills_the_forward_return_for_that_row():
    ser = _closes([("2013-06-03", 100.0), ("2014-06-03", 200.0),
                   ("2024-01-02", 100.0)])
    out = fx.to_sek(ser, "USD", _rates({"2024-01-02": (10.0,)}))
    # nothing at or before an early report date survives, so build_scores_panel
    # finds no anchor price and the row's return is NaN rather than wrong.
    assert out[out.index <= pd.Timestamp("2014-06-03")].empty


def test_missing_currency_column_raises_rather_than_silently_passing_through():
    ser = _closes([("2024-01-02", 100.0)])
    with pytest.raises(KeyError):
        fx.to_sek(ser, "NOK", _rates({"2024-01-02": (10.0,)}))


def test_empty_series_is_a_noop():
    empty = pd.Series(dtype=float)
    assert fx.to_sek(empty, "USD", _rates({"2024-01-02": (10.0,)})).empty
    assert fx.to_sek(None, "USD", None) is None


# ------------------------------------------------------------ cache lifecycle
def _stub_fetch(calls, value=10.0):
    def _fetch(currency, start, end):
        calls.append((currency, start, end))
        idx = pd.bdate_range(start, end)
        return pd.Series([value] * len(idx), index=idx)
    return _fetch


def test_load_sek_rates_returns_none_when_absent(tmp_path):
    assert fx.load_sek_rates(str(tmp_path / "nope.csv")) is None


def test_fetch_writes_cache_then_second_call_makes_no_requests(tmp_path):
    path = str(tmp_path / "fx_sek.csv")
    calls = []
    first = fx.fetch_sek_rates("2024-01-01", "2024-03-01", path=path,
                               currencies=["USD", "EUR"],
                               fetch_fn=_stub_fetch(calls))
    assert sorted(first.columns) == ["EUR", "USD"]
    assert len(calls) == 2

    reloaded = fx.load_sek_rates(path)
    assert reloaded is not None and len(reloaded) == len(first)

    fx.fetch_sek_rates("2024-01-01", "2024-03-01", path=path,
                       currencies=["USD", "EUR"], fetch_fn=_stub_fetch(calls))
    assert len(calls) == 2  # idempotent: covered range, zero fetches


def test_a_range_the_cache_does_not_cover_is_refetched_and_merged(tmp_path):
    path = str(tmp_path / "fx_sek.csv")
    calls = []
    fx.fetch_sek_rates("2024-02-01", "2024-03-01", path=path,
                       currencies=["USD"], fetch_fn=_stub_fetch(calls, 10.0))
    merged = fx.fetch_sek_rates("2024-01-01", "2024-03-01", path=path,
                                currencies=["USD"],
                                fetch_fn=_stub_fetch(calls, 11.0))
    assert len(calls) == 2
    assert merged.index.min() == pd.Timestamp("2024-01-01")
    # fresh values win over cached ones where they overlap
    assert merged.loc[pd.Timestamp("2024-02-01"), "USD"] == 11.0


def test_a_currency_missing_from_the_cache_forces_a_refetch(tmp_path):
    path = str(tmp_path / "fx_sek.csv")
    calls = []
    fx.fetch_sek_rates("2024-01-01", "2024-03-01", path=path,
                       currencies=["USD"], fetch_fn=_stub_fetch(calls))
    fx.fetch_sek_rates("2024-01-01", "2024-03-01", path=path,
                       currencies=["USD", "NOK"], fetch_fn=_stub_fetch(calls))
    assert [c for c, _, _ in calls] == ["USD", "NOK", "USD"]
