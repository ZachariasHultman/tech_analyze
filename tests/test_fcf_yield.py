"""calculate_free_cashflow_yield: the largest untested block under the score.

115 lines, 93% of them uncovered before this file, feeding `fcfy_pe` -- one of
the five VALUE_METRICS. It is also wrapped end to end in a bare
`except Exception` that returns `(None, None, None, None)`, so every internal
failure is indistinguishable from "this company has no FCF data": the metric
just goes missing, drops the company below the cross-sectional coverage floor,
and scores on absolute thresholds instead. Nothing raises and nothing is
logged except one print. A happy-path test that proves it actually *computes*
is therefore the point of this file, not an afterthought.

No live API: `yahoo_ticker` is faked down to the three attributes the function
touches (`.cashflow`, `.ticker`, `.info`), `stock_info` is the plain Avanza
dict it already is, and `df_hist` is a close-price frame.
"""
import pandas as pd
import pytest

from analyzer.config import PRICE_SMOOTHING_WINDOW
from analyzer.financial_metrics import calculate_free_cashflow_yield


class _FakeTicker:
    """Only what calculate_free_cashflow_yield actually reads."""

    def __init__(self, cashflow, ticker="TEST.ST", info=None):
        self.cashflow = cashflow
        self.ticker = ticker
        self.info = info or {}


def _cashflow(values, label="Free Cash Flow", dates=None):
    """yfinance hands back most-recent-year-FIRST columns."""
    dates = dates or ["2024-12-31", "2023-12-31"][: len(values)]
    return pd.DataFrame([values], index=[label], columns=pd.to_datetime(dates))


def _stock_info(market_cap=1_000_000.0, currency="SEK"):
    return {"keyIndicators": {"marketCapital": {"value": market_cap,
                                                "currency": currency}}}


def _hist(closes):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"close": closes}, index=idx)


# --- the happy path --------------------------------------------------------

def test_yield_is_latest_fcf_over_market_cap():
    """With no price history there is nothing to smooth, so the quoted market
    cap is used directly -- the documented fallback."""
    fcf_yield, latest_fcf, _, _ = calculate_free_cashflow_yield(
        _FakeTicker(_cashflow([50_000.0, 40_000.0])), _stock_info(1_000_000.0)
    )
    assert latest_fcf == 50_000.0
    assert fcf_yield == pytest.approx(0.05)


def test_latest_fcf_is_the_first_column_not_the_last():
    """yfinance orders columns most-recent-first, the opposite of Avanza's
    yearly blocks. Reading `.iloc[-1]` here would silently score every company
    on its oldest cash-flow year."""
    _, latest_fcf, _, _ = calculate_free_cashflow_yield(
        _FakeTicker(_cashflow([50_000.0, 10_000.0])), _stock_info()
    )
    assert latest_fcf == 50_000.0


def test_market_cap_is_repriced_on_the_smoothed_close():
    """Shares are backed out of the quoted cap on the *last* close, then
    re-priced on the 60-day mean (config.PRICE_SMOOTHING_WINDOW).

    Held flat at 100 for the whole window except a final spike to 200: shares
    = cap/200, and the smoothed cap is mean(closes) * shares -- roughly half
    the quoted cap, so the yield roughly doubles. This is the churn damping
    the whole smoothing exists for; on spot price the yield would read 0.05.
    """
    closes = [100.0] * (PRICE_SMOOTHING_WINDOW - 1) + [200.0]
    fcf_yield, _, _, _ = calculate_free_cashflow_yield(
        _FakeTicker(_cashflow([50_000.0])), _stock_info(1_000_000.0), _hist(closes)
    )
    shares = 1_000_000.0 / 200.0
    expected = 50_000.0 / (sum(closes) / len(closes) * shares)
    assert fcf_yield == pytest.approx(expected)
    assert fcf_yield > 0.09  # ~2x the un-smoothed 0.05


def test_short_history_falls_back_to_the_quoted_market_cap():
    """smoothed_close returns None below a full window rather than a shorter
    mean, and the yield must then fall back rather than go missing."""
    fcf_yield, _, _, _ = calculate_free_cashflow_yield(
        _FakeTicker(_cashflow([50_000.0])), _stock_info(1_000_000.0),
        _hist([100.0] * 5),
    )
    assert fcf_yield == pytest.approx(0.05)


# --- the historical dict the panel reads -----------------------------------

def test_history_is_keyed_by_reporting_date():
    _, _, _, fcf_hist = calculate_free_cashflow_yield(
        _FakeTicker(_cashflow([50_000.0, 40_000.0])), _stock_info()
    )
    assert fcf_hist == {"2024-12-31": 50_000.0, "2023-12-31": 40_000.0}


def test_historical_yield_cuts_the_price_series_at_each_reporting_date():
    """`fcf_yield_hist` is what the backtest panel reads as `fcfy`, so each
    year's denominator may only use closes up to that year's report date.
    Prices here step up sharply after the 2024 cut; if post-report closes
    leaked in, the 2024 yield would be computed against the higher level.
    """
    closes = [100.0] * PRICE_SMOOTHING_WINDOW + [1000.0] * PRICE_SMOOTHING_WINDOW
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    df_hist = pd.DataFrame({"close": closes}, index=idx)
    cutoff = idx[PRICE_SMOOTHING_WINDOW - 1].strftime("%Y-%m-%d")

    _, _, yield_hist, _ = calculate_free_cashflow_yield(
        _FakeTicker(_cashflow([50_000.0], dates=[cutoff])),
        _stock_info(1_000_000.0), df_hist,
    )
    shares = 1_000_000.0 / closes[-1]
    assert yield_hist[cutoff] == pytest.approx(50_000.0 / (100.0 * shares))


# --- the guards, which are the difference between "missing" and "wrong" ----

@pytest.mark.parametrize("market_cap", [0, -1, None, "1000"],
                         ids=["zero", "negative", "missing", "string"])
def test_unusable_market_cap_returns_all_none(market_cap):
    assert calculate_free_cashflow_yield(
        _FakeTicker(_cashflow([50_000.0])), _stock_info(market_cap)
    ) == (None, None, None, None)


def test_missing_cashflow_frame_returns_all_none():
    assert calculate_free_cashflow_yield(
        _FakeTicker(None), _stock_info()
    ) == (None, None, None, None)


def test_empty_cashflow_frame_returns_all_none():
    assert calculate_free_cashflow_yield(
        _FakeTicker(pd.DataFrame()), _stock_info()
    ) == (None, None, None, None)


def test_all_nan_fcf_returns_all_none():
    assert calculate_free_cashflow_yield(
        _FakeTicker(_cashflow([float("nan"), float("nan")])), _stock_info()
    ) == (None, None, None, None)


def test_stock_info_without_key_indicators_returns_all_none():
    assert calculate_free_cashflow_yield(
        _FakeTicker(_cashflow([50_000.0])), {}
    ) == (None, None, None, None)


@pytest.mark.parametrize("label", ["Free Cash Flow", "FreeCashFlow",
                                   "free_cash_flow", "Levered Free Cash Flow"])
def test_fcf_row_is_found_under_every_yfinance_label(label):
    """yfinance renamed this row across versions; the last one is only caught
    by the case-insensitive 'free'+'cash' fallback."""
    _, latest_fcf, _, _ = calculate_free_cashflow_yield(
        _FakeTicker(_cashflow([50_000.0], label=label)), _stock_info()
    )
    assert latest_fcf == 50_000.0


def test_unrecognised_row_label_returns_all_none():
    assert calculate_free_cashflow_yield(
        _FakeTicker(_cashflow([50_000.0], label="Operating Income")), _stock_info()
    ) == (None, None, None, None)
