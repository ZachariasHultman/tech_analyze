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
QUALITY_METRICS = [
    "piotroski f-score status", "earnings quality status", "roe_de ratio status",
    "net debt - ebitda status", "net margin vs avg status", "roe vs avg status",
    "gross margin stability status", "revenue y cagr status", "eps y cagr status",
    "revenue yoy hit-rate status", "eps yoy hit-rate status",
    "revenue trend year status",
]
VALUE_METRICS = [
    "fcfy_pe ratio status", "cagr_pe ratio status", "roe_pe ratio status",
    "dividend yield status", "dividend growth status", "price momentum status",
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

# Tickers excluded from the main analysis loop.
# 1640718 — unidentified; original reason for exclusion unknown (not
# recoverable from the repo or git history), kept from prior behavior.
EXCLUDED_TICKER_IDS = {"1640718"}
