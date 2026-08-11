"""Tests for item 1: Piotroski F-Score criterion #4 (accruals quality) disabled.

Criterion #4 (operating cash flow > net income) has no historical fallback
(operatingCashFlow is Nordic-only live, never in the CSV snapshot schema), so it
was permanently disabled to keep live and backfilled Piotroski on an identical
0-8 definition. These tests use a hand-built fixture that fires all 8 remaining
criteria and would have fired criterion #4 under the old logic.
"""
from analyzer.financial_metrics import calculate_piotroski_f_score


def _vals(seq):
    """Wrap a plain list into the {value: x} dict shape _as_vals expects."""
    return [{"value": v} for v in seq]


def _fixture(op_cf):
    """A company that satisfies criteria 1,2,3,5,6,7,8,9 (all 8 kept ones).

    Two-year series, latest last:
      1. net income > 0              netProfit[-1] = 200 > 0
      2. op_cf > 0                   op_cf passed in
      3. ROA improving               np/ta: 0.1 -> 0.2
      5. D/E decreased               1.0 -> 0.5
      6. equity ratio improved       0.5 -> 0.6
      7. equity/share not declining  10 -> 12
      8. gross margin improved       0.1 -> 0.2
      9. asset turnover improved     2.0 -> 3.0
    """
    ticker_analysis = {
        "companyFinancialsByYear": {
            "netProfit": _vals([100, 200]),
            "totalAssets": _vals([1000, 1000]),
            "totalLiabilities": _vals([500, 400]),
            "sales": _vals([2000, 3000]),
            "profitMargin": _vals([0.1, 0.2]),
            "debtToEquityRatio": _vals([1.0, 0.5]),
        },
        "companyKeyRatiosByYear": {
            "equityPerShare": _vals([10, 12]),
        },
    }
    ticker_info = {"keyIndicators": {"operatingCashFlow": op_cf}}
    return ticker_analysis, ticker_info


def test_score_tops_out_at_8():
    # op_cf = 250 > net_income 200: criterion 4 WOULD have fired under old logic,
    # pushing the score to 9. With #4 disabled it caps at 8.
    ta, ti = _fixture(op_cf=250)
    assert calculate_piotroski_f_score(ta, ti) == 8


def test_criterion_4_fully_inert():
    # Flipping op_cf vs net_income must not change the score at all. Both op_cf
    # values are > 0 so criterion 2 fires in both cases; only the #4 relationship
    # flips (250 > 200 vs 150 < 200). Equal scores prove #4 is inert, not capped.
    ta_high, ti_high = _fixture(op_cf=250)  # op_cf > net_income
    ta_low, ti_low = _fixture(op_cf=150)  # op_cf < net_income, still > 0
    assert calculate_piotroski_f_score(ta_high, ti_high) == 8
    assert calculate_piotroski_f_score(ta_low, ti_low) == 8
