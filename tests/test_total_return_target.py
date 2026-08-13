"""The forward-return target must include dividends.

Avanza's OHLC close is NOT dividend-adjusted. Verified against a real snapshot:
Handelsbanken A closes drop ~12.5% in a single day every March/April, which is
the ex-dividend date, not a price move (Yahoo's adjusted close for the same day
moves ~1.2% and reports a 15.00 SEK dividend).

Scoring against a price-only target therefore penalises exactly the stocks the
dividend-yield metric rewards. Measured on the real panel, switching to a
total-return target moved `dividend yield status` from IC +0.011 to +0.074 and
the value sleeve's spread from +1.2% to +4.1%.

`dividend_per_share` is already in every snapshot, so this needs no new data
source. Its dates are calendar-year labels for the *payment* year (verified:
Handelsbanken's `2025-12-31: 15.0` is the 15.00 SEK paid 2025-03-27), which is
why the window weighting below prorates between label years rather than
treating the label as an ex-date.
"""

import numpy as np
import pandas as pd
import pytest

from analyzer.panel import forward_dividend_yield


def _dps(mapping):
    return pd.DataFrame(
        {"date": [f"{y}-12-31" for y in mapping], "value": list(mapping.values())}
    )


def test_non_payer_yields_zero():
    assert forward_dividend_yield(None, pd.Timestamp("2024-02-01"), 100.0) == 0.0
    assert forward_dividend_yield(
        pd.DataFrame(columns=["date", "value"]), pd.Timestamp("2024-02-01"), 100.0
    ) == 0.0


def test_early_year_report_weights_mostly_its_own_label_year():
    # A 1 Feb report: ~92% of the following 12 months falls in label year Y,
    # ~8% in Y+1. Nordic spring dividends are paid inside that first stretch.
    dps = _dps({2024: 10.0, 2025: 0.0})
    y = forward_dividend_yield(dps, pd.Timestamp("2024-02-01"), 100.0)
    assert y == pytest.approx(0.0915, abs=0.005)


def test_late_year_report_weights_mostly_the_next_label_year():
    dps = _dps({2024: 0.0, 2025: 10.0})
    y = forward_dividend_yield(dps, pd.Timestamp("2024-11-15"), 100.0)
    assert y > 0.08


def test_constant_dividend_is_scale_invariant_to_report_month():
    # When DPS is flat year over year the prorating must wash out entirely:
    # any 12-month window collects one annual dividend.
    dps = _dps({2023: 8.0, 2024: 8.0, 2025: 8.0})
    for month in (1, 4, 7, 11):
        y = forward_dividend_yield(dps, pd.Timestamp(f"2024-{month:02d}-10"), 200.0)
        assert y == pytest.approx(0.04, abs=1e-9)


def test_missing_label_year_falls_back_to_the_other():
    dps = _dps({2024: 5.0})
    y = forward_dividend_yield(dps, pd.Timestamp("2024-06-01"), 100.0)
    assert y == pytest.approx(0.05, abs=1e-9)


@pytest.mark.parametrize("price", [0.0, -1.0, np.nan, None])
def test_unusable_price_gives_nan_not_a_bogus_yield(price):
    dps = _dps({2024: 5.0})
    assert np.isnan(forward_dividend_yield(dps, pd.Timestamp("2024-06-01"), price))


def test_build_scores_panel_columns_and_semantics(monkeypatch):
    """fwd_excess_return_1y must now be demeaned TOTAL return, and the
    price-only version must survive under its own explicit name."""
    from analyzer import panel as panel_mod

    dates = pd.date_range("2022-01-01", "2026-01-01", freq="D")
    # Two companies: one flat price + fat dividend, one rising price + none.
    ohlc_a = pd.DataFrame({"close": np.full(len(dates), 100.0)}, index=dates)
    ohlc_b = pd.DataFrame({"close": np.linspace(100.0, 200.0, len(dates))}, index=dates)
    hist = pd.DataFrame(
        {
            "ohlc": [ohlc_a, ohlc_b],
            "dividend_per_share": [_dps({2024: 10.0, 2025: 10.0}), None],
        },
        index=["A", "B"],
    )
    monkeypatch.setattr(panel_mod, "get_hist_data", lambda _d: hist)
    monkeypatch.setattr(panel_mod, "_all_scored_metrics", lambda: [])
    monkeypatch.setattr(panel_mod, "_score_snapshot",
                        lambda *a, **k: pd.DataFrame(index=["A", "B"]))

    fundamentals = pd.DataFrame([
        {"company_id": "A", "company": "A", "fiscal_year": 2024,
         "report_date": "2024-02-01", "sector": "X"},
        {"company_id": "B", "company": "B", "fiscal_year": 2024,
         "report_date": "2024-02-01", "sector": "X"},
    ])
    out = panel_mod.build_scores_panel(fundamentals, "data").set_index("company_id")

    for col in ("fwd_return_1y", "fwd_dividend_yield_1y", "fwd_total_return_1y",
                "fwd_excess_price_return_1y", "fwd_excess_return_1y"):
        assert col in out.columns, f"missing {col}"

    # A: flat price, ~10% dividend. B: ~+25% price over the year, no dividend.
    assert out.loc["A", "fwd_return_1y"] == pytest.approx(0.0, abs=1e-9)
    assert out.loc["A", "fwd_dividend_yield_1y"] == pytest.approx(0.10, abs=0.01)
    assert out.loc["B", "fwd_dividend_yield_1y"] == 0.0
    assert out.loc["A", "fwd_total_return_1y"] == pytest.approx(
        out.loc["A", "fwd_return_1y"] + out.loc["A", "fwd_dividend_yield_1y"]
    )

    # The headline target is total-return based; the old one still available.
    tot = out["fwd_total_return_1y"]
    assert out["fwd_excess_return_1y"].tolist() == pytest.approx(
        (tot - tot.mean()).tolist()
    )
    pr = out["fwd_return_1y"]
    assert out["fwd_excess_price_return_1y"].tolist() == pytest.approx(
        (pr - pr.mean()).tolist()
    )
    # A's dividend must close part of the gap to B versus the price-only view.
    assert (out.loc["A", "fwd_excess_return_1y"]
            > out.loc["A", "fwd_excess_price_return_1y"])
