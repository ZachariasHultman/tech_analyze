"""Investment companies are out of the universe entirely.

They were removed rather than fixed. The reason is not the ~250 lines of
NAV-scraping and dual-template machinery they carried, it is that they were
scored on 8 metrics where every other company is scored on ~17 -- and `points`,
a *sum* of weighted metric scores, was then ranked across both pools for the
same 25 watchlist slots. Summing 8 things and 17 things and comparing the
totals is the same two-ranking-regimes defect that scoring the two pools
separately used to cause (see the history in test_unified_pooling.py), and it
had never been measured because these names produce zero panel rows:
`_iter_fiscal_years` derives report dates from `revenue_year`, which their
snapshots do not carry.

So every backtest number in this repo described the other companies only,
while Investor, Kinnevik, Latour and friends competed for slots on an
unvalidated basis. Excluding them makes the running system the one that was
actually measured.

Excluded, not silently mis-scored: their snapshots also lack `de_ratio`,
`eps_quarter`, `revenue_quarter` and `netDebtEbitdaRatio`, so routing them
through the regular template would score them on a handful of metrics and let
them keep competing -- a quieter version of the same defect.
"""
import importlib

import pytest

from analyzer.config import EXCLUDED_SECTORS, QUALITY_METRICS, VALUE_METRICS
from analyzer.main import _is_excluded
from analyzer.summary_manager import SummaryManager


def _info(sector_name, ticker_id="123"):
    return {"orderbookId": ticker_id, "name": "Test AB",
            "sectors": [{"sectorId": "51", "sectorName": sector_name}]}


def test_investment_sector_is_excluded_from_the_loop():
    assert _is_excluded("5247", _info("Investmentbolag")) is True


def test_ordinary_sectors_are_not_excluded():
    for sector in ("Industri", "Teknologi", "Bank", "Fastigheter & Utveckling"):
        assert _is_excluded("123", _info(sector)) is False, sector


def test_excluded_ticker_ids_still_apply():
    """The id-based exclusion predates this and must keep working."""
    assert _is_excluded("1640718", _info("Industri")) is True


def test_a_ticker_without_sectors_is_excluded():
    assert _is_excluded("123", {"orderbookId": "123", "sectors": []}) is True


def test_excluded_sectors_is_not_empty():
    assert "Investmentbolag" in EXCLUDED_SECTORS


def test_manager_has_a_single_pool():
    """One summary and one template. The dual-pool split is what made two
    ranking regimes possible in the first place."""
    sm = SummaryManager()
    assert not hasattr(sm, "summary_investment")
    assert not hasattr(sm, "template_investment")


def test_no_nav_metrics_remain_in_either_sleeve():
    for metric in QUALITY_METRICS + VALUE_METRICS:
        assert "nav" not in metric.lower(), metric


def test_no_nav_metrics_remain_in_the_template():
    sm = SummaryManager()
    assert not [k for k in sm.template if "nav" in str(k).lower()]


def test_the_nav_scraper_is_gone():
    """get_NAV_data hit ibindex.se -- a second unofficial dependency that could
    break independently of Avanza, kept alive for six companies."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("analyzer.get_NAV_data")


def test_nav_discount_calculation_is_gone():
    import analyzer.financial_metrics as fm
    assert not hasattr(fm, "calculate_NAV_discount")
