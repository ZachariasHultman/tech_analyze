"""Momentum needs 200 trading days before the report date.

Avanza serves a *rolling* ~5-year OHLC window, so for the early fiscal years
the panel scores there is simply no pre-report price history: measured on the
real snapshots, `price momentum status` was 0% populated for FY2019/2020/2021
and 10% for FY2022 -- absent from four of the seven years the challenger gate
scores, while being the strongest single predictor in the set.

The verified Yahoo backfill reaches back to 2015, so it is preferred where it
exists and Avanza remains the fallback. The union matters in both directions:
Yahoo verifies for fewer companies than Avanza covers, so replacing rather
than falling back would buy the early years by losing companies in the late
ones.
"""
import numpy as np
import pandas as pd
import pytest

from analyzer.panel import MOMENTUM_SMA_DAYS, momentum_as_of


def _closes(n, start="2015-01-01", value=100.0, step=0.0):
    idx = pd.bdate_range(start, periods=n)
    return pd.Series([value + step * i for i in range(n)], index=idx)


def test_flat_series_has_zero_momentum():
    s = _closes(300)
    assert momentum_as_of(s, s.index[-1]) == pytest.approx(0.0)


def test_rising_series_has_positive_momentum():
    s = _closes(300, step=1.0)
    # last price is above the trailing 200-day mean of a rising series
    assert momentum_as_of(s, s.index[-1]) > 0


def test_falling_series_has_negative_momentum():
    s = _closes(300, value=400.0, step=-1.0)
    assert momentum_as_of(s, s.index[-1]) < 0


def test_returns_none_below_the_window_rather_than_shortening_it():
    """A 60-day SMA is a different statistic, not a noisy 200-day one."""
    s = _closes(MOMENTUM_SMA_DAYS - 1)
    assert momentum_as_of(s, s.index[-1]) is None
    assert momentum_as_of(_closes(MOMENTUM_SMA_DAYS), s.index[-1] + pd.Timedelta(days=7)) is not None


def test_uses_only_closes_up_to_the_report_date():
    """The look-ahead guard: a spike after report_date must not move it."""
    s = _closes(400)
    report_date = s.index[250]
    before = momentum_as_of(s, report_date)
    spiked = s.copy()
    spiked.iloc[260:] = 10_000.0
    assert momentum_as_of(spiked, report_date) == pytest.approx(before)


def test_is_numeraire_free():
    """A ratio within one series -- rescaling the whole series changes
    nothing. This is what lets the unconverted Yahoo leg sit alongside a
    SEK-converted target without a currency inconsistency."""
    s = _closes(300, step=0.5)
    d = s.index[-1]
    assert momentum_as_of(s * 11.7, d) == pytest.approx(momentum_as_of(s, d))


def test_degrades_on_missing_or_junk_input():
    assert momentum_as_of(None, pd.Timestamp("2024-01-01")) is None
    assert momentum_as_of(pd.Series(dtype=float), pd.Timestamp("2024-01-01")) is None
    assert momentum_as_of(_closes(300) * np.nan, pd.Timestamp("2024-01-01")) is None


def test_nonpositive_sma_is_rejected():
    s = pd.Series([0.0] * 300, index=pd.bdate_range("2015-01-01", periods=300))
    assert momentum_as_of(s, s.index[-1]) is None


# --------------------------------------------------------------------------
# the union: Yahoo preferred, Avanza fallback
# --------------------------------------------------------------------------

def _panel_with(monkeypatch, avanza_close, yahoo_close, report_date):
    """Drive build_fundamentals_panel over one synthetic company."""
    from analyzer import panel

    ohlc = pd.DataFrame({"close": avanza_close})
    row = {
        "sector": "Industrials",
        "ohlc": ohlc,
        "revenue_year": pd.DataFrame({"date": [report_date], "value": [100.0]}),
    }
    df = pd.DataFrame([row], index=["ACME"])
    monkeypatch.setattr(panel, "get_hist_data", lambda _d: df)
    closes = {"ACME": yahoo_close} if yahoo_close is not None else {}
    return panel.build_fundamentals_panel("ignored", yahoo_closes=closes)


def test_yahoo_fills_a_year_avanza_cannot_reach(monkeypatch):
    report_date = pd.Timestamp("2019-03-01")
    # Avanza's rolling window starts well after the report date
    avanza = _closes(300, start="2022-01-01", step=1.0)
    yahoo = _closes(1200, start="2015-01-01", step=1.0)

    without = _panel_with(monkeypatch, avanza, None, report_date)
    assert without["price momentum status"].isna().all(), (
        "precondition: Avanza alone cannot supply this fiscal year")

    with_yahoo = _panel_with(monkeypatch, avanza, yahoo, report_date)
    assert with_yahoo["price momentum status"].notna().all()


def test_avanza_still_used_when_the_company_has_no_verified_yahoo(monkeypatch):
    """The union's other half: 129 Yahoo companies vs 133 on Avanza, so a
    replacement would drop companies out of the late years."""
    report_date = pd.Timestamp("2024-03-01")
    avanza = _closes(600, start="2022-01-01", step=1.0)
    out = _panel_with(monkeypatch, avanza, None, report_date)
    assert out["price momentum status"].notna().all()


def test_yahoo_takes_precedence_when_both_are_available(monkeypatch):
    report_date = pd.Timestamp("2024-03-01")
    avanza = _closes(600, start="2022-01-01", step=1.0)     # rising
    yahoo = _closes(1200, start="2015-01-01", value=900.0, step=-0.5)  # falling
    out = _panel_with(monkeypatch, avanza, yahoo, report_date)
    assert out["price momentum status"].iloc[0] < 0, "should reflect Yahoo, not Avanza"
