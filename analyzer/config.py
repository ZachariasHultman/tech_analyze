# ----------------------------------------------------------------------
#  Centralized constants (formerly scattered magic numbers)
# ----------------------------------------------------------------------

# Two-sleeve scoring: a stock qualifies for the buy watchlist only when it
# ranks at least this high (cross-sectional percentile, 0–1) in BOTH the
# quality and value sleeves. Replaces the old single-spearman qualify gate.
SLEEVE_GATE_MIN = 0.4

# Metric-to-sleeve assignment. Every status column present in either
# SummaryManager template (regular + investment) must appear in exactly one
# sleeve — enforced by tests/test_sleeves.py. QUALITY = "is this a solid,
# well-run business"; VALUE = "is it cheaply priced / shareholder-friendly".
# NAV-discount metrics (investment companies) are a cheapness signal → VALUE.
#
# Four metrics were removed outright rather than down-weighted:
#   dividend yield          - measured IC -0.026 over 7 fiscal years (p=0.76),
#                             i.e. nothing, even after its look-ahead leak was
#                             fixed. Note the forward target already includes
#                             dividends, so this says yield does not predict
#                             *total* return -- not that dividends don't count.
#   earnings quality        - Avanza exposes no historical OCF series, so the
#                             column is permanently all-NaN in the panel.
#   revenue/eps yoy hit-rate- 0% coverage in the panel.
#
# The last three are the important case: all are ~100% populated *live*, so
# they were shifting the live ranking while contributing exactly nothing to the
# backtest that validated it. Removing them makes the running system the one
# that was actually measured, rather than a superset of it.
QUALITY_METRICS = [
    "piotroski f-score status", "roe_de ratio status",
    "net debt - ebitda status", "net margin vs avg status", "roe vs avg status",
    "gross margin stability status", "revenue y cagr status", "eps y cagr status",
    "revenue trend year status",
]
VALUE_METRICS = [
    "fcfy_pe ratio status", "cagr_pe ratio status", "roe_pe ratio status",
    "dividend growth status", "price momentum status",
    "nav discount status", "calculated nav discount status",
    "nav discount trend status",
]

# Minimum companies a fiscal year must have (with a usable forward return)
# before its cross-section counts as evidence. Deliberately the same number
# _quintile_spread uses to switch from terciles to quintiles: below it, a
# "quintile spread" is really a 3-vs-3 comparison.
#
# Concretely: FY2021 in the real panel has 9 companies, because the live OHLC
# window starts mid-2021 and only off-cycle reporters got a price anchor. That
# year scored IC=+0.867 / spread=+41.6% off three US mega-caps beating two
# Nordic small-caps, and dragged the reported mean IC from +0.038 to +0.203.
MIN_CROSS_SECTION = 25

# A fiscal year must also cover this fraction of the companies that *have*
# fundamentals that year. Size alone is not enough: a partial Yahoo price
# backfill gives the earliest years a forward return for only the backfilled
# subset, which clears MIN_CROSS_SECTION while being a biased sample rather
# than the universe (observed: FY2019 had 27 companies, all of them large-cap
# names from a 30-symbol partial backfill, against 87 with fundamentals).
MIN_YEAR_COVERAGE = 0.60

# Trading days of close history averaged before a price is used as the
# denominator of a valuation ratio (FCF yield's market cap, dividend yield).
# Spot price made those two metrics the only fast-moving inputs in an otherwise
# annual-report-driven score, and with ranks 19-25 of the watchlist packed
# inside 0.65 points, that noise alone reshuffled the bottom of the list every
# week. 60 days ~ one quarter: enough to damp the week-to-week move, short
# enough that a real repricing still shows up within a rebalance cycle.
#
# Deliberately NOT applied to `price momentum status` (which must track spot
# against its own SMA200 -- that comparison *is* the metric) nor to the NAV
# discounts (already a 30-day mean, and left alone by decision).
PRICE_SMOOTHING_WINDOW = 60

# Share of the final ranking key taken by price momentum, the rest going to the
# fundamental composite. Both sides enter as within-cross-section percentile
# ranks, because `points` spans roughly -6..+14 while momentum is a fraction in
# -0.08..+0.24 -- averaging the raw numbers would hand the ranking to whichever
# happens to have the wider numeric spread.
#
# 0.5 is not fitted, and deliberately so: with 7 annual observations any fitted
# split is noise (the weight optimizer has been rejected three times, the
# permutation p climbing 0.199 -> 0.408 -> 0.512). Equal weight between two
# comparably predictive, weakly correlated signals is the defensible default.
#
# Measured on the panel, top-25 basket mean excess return: fundamentals alone
# +2.77%/yr (p=0.171), momentum alone +6.82%/yr (p=0.144, and the metric most
# flattered by survivorship bias), the 50/50 blend +5.90%/yr (p=0.053) and
# positive in 6 of 7 years -- the most consistent of the three.
#
# Momentum stays inside the fundamental composite as well, so its effective
# weight is 0.5 + 0.5*(1/21) ~ 0.52. Stripping it out scored the same on IC
# but worse on the basket (+4.65 vs +5.90%/yr), and would have forced
# `price momentum status` out of the VALUE sleeve for no measured gain.
MOMENTUM_BLEND_WEIGHT = 0.5

# Tickers excluded from the main analysis loop.
# 1640718 — unidentified; original reason for exclusion unknown (not
# recoverable from the repo or git history), kept from prior behavior.
EXCLUDED_TICKER_IDS = {"1640718"}
