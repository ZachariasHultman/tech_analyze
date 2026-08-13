"""How much of the score's apparent edge is stock selection vs currency?

Converting the forward-return target to SEK is correct -- it is what a SEK
investor earns -- but it hands every USD-listed name the same USD/SEK move, so
a currency cohort becomes a shared factor inside the target. Within-year
demeaning removes the universe mean, not a group effect.

Measured on the real panel, currency-within-year explains 5.3% of return
variance against year's own 6.7%, and demeaning it away drops the pooled
quintile spread from +0.97% to +0.23% -- i.e. most of the headline spread was
currency exposure rather than company picking.

So the SEK target stays (it is what you earn) and this reports the
currency-neutral view alongside it (what you are actually skilled at). Both
numbers, always, rather than a choice between them.
"""

import numpy as np
import pandas as pd
import pytest

from analyzer.validation import currency_neutral_ic


def _panel(n_years=4, n_per_ccy=30, fx_shift=0.0, seed=0, signal=0.0,
           alternate=True):
    """Half SEK, half USD. `fx_shift` is a per-year return bump applied to
    every USD name -- exactly the cohort effect this diagnostic isolates.

    `alternate` flips the bump's sign each year, which is realistic (FX goes
    both ways) but averages a constant currency tilt to zero across years. Set
    it False to model a currency that trends one way over the whole sample --
    the case where a pure currency bet actually looks like skill.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for k, fy in enumerate(range(2022, 2022 + n_years)):
        bump = fx_shift * ((1 if k % 2 == 0 else -1) if alternate else 1)
        for ccy in ("SEK", "USD"):
            for i in range(n_per_ccy):
                score = float(rng.normal())
                ret = signal * score + float(rng.normal(0, 0.2))
                if ccy == "USD":
                    ret += bump
                rows.append({
                    "company_id": f"{ccy}{i}", "fiscal_year": fy,
                    "composite_score_equal": score,
                    "fwd_total_return_1y": ret,
                    "currency": ccy,
                })
    df = pd.DataFrame(rows)
    df["fwd_excess_return_1y"] = df.fwd_total_return_1y - df.groupby(
        "fiscal_year").fwd_total_return_1y.transform("mean")
    return df


def _ccy_map(df):
    return dict(zip(df.company_id, df.currency))


def test_reports_currency_variance_share():
    df = _panel(fx_shift=0.25)
    res = currency_neutral_ic(df, currency_map=_ccy_map(df))
    # A large per-cohort bump must show up as a large currency variance share.
    # (0.25 bump against 0.2 residual sd lands around 0.17.)
    assert res["currency_variance_share"] > 0.12
    assert _panel(fx_shift=0.0, seed=1).pipe(
        lambda d: currency_neutral_ic(d, currency_map=_ccy_map(d))
    )["currency_variance_share"] < res["currency_variance_share"]


def test_no_cohort_effect_leaves_ic_essentially_unchanged():
    df = _panel(fx_shift=0.0, signal=0.5, seed=3)
    res = currency_neutral_ic(df, currency_map=_ccy_map(df))
    assert abs(res["mean_ic"] - res["mean_ic_raw"]) < 0.02
    assert res["currency_variance_share"] < 0.05


def test_a_pure_currency_bet_survives_raw_but_dies_neutralised():
    """The failure this exists to catch: a score that only knows a stock's
    currency looks skilful against the SEK target and must not survive here."""
    df = _panel(fx_shift=0.30, seed=5, alternate=False)
    # score encodes nothing but "is this USD?" -- zero company information
    df["composite_score_equal"] = (df.currency == "USD").astype(float)
    res = currency_neutral_ic(df, currency_map=_ccy_map(df))
    assert abs(res["mean_ic_raw"]) > 0.2, "the fake bet should look real raw"
    assert abs(res["mean_ic"]) < 0.05, "and must vanish once neutralised"


def test_real_company_signal_survives_neutralisation():
    df = _panel(fx_shift=0.20, signal=0.6, seed=7)
    res = currency_neutral_ic(df, currency_map=_ccy_map(df))
    assert res["mean_ic"] > 0.2


def test_single_currency_universe_is_a_no_op():
    df = _panel(fx_shift=0.0, signal=0.4, seed=11)
    df["currency"] = "SEK"
    res = currency_neutral_ic(df, currency_map=_ccy_map(df))
    assert res["currency_variance_share"] == pytest.approx(0.0, abs=1e-9)
    assert res["mean_ic"] == pytest.approx(res["mean_ic_raw"], abs=1e-9)


def test_missing_currency_map_degrades_to_none():
    """No Yahoo symbol map (fresh clone) must not break --validate."""
    df = _panel()
    assert currency_neutral_ic(df, currency_map={}) is None
    assert currency_neutral_ic(pd.DataFrame(), currency_map={"a": "SEK"}) is None


def test_cohorts_below_the_floor_are_excluded():
    """A 1-company currency cohort demeans to exactly zero, which would be a
    fake perfect neutralisation rather than a measurement."""
    df = _panel(fx_shift=0.1, signal=0.5, seed=13)
    solo = df[df.fiscal_year == 2022].iloc[[0]].copy()
    solo["company_id"] = "GBP0"
    solo["currency"] = "GBP"
    df = pd.concat([df, solo], ignore_index=True)
    res = currency_neutral_ic(df, currency_map=_ccy_map(df))
    assert "GBP" not in res["cohorts_used"]
    assert "SEK" in res["cohorts_used"] and "USD" in res["cohorts_used"]
