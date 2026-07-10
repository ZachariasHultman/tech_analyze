# ----------------------------------------------------------------------
#  Centralized constants (formerly scattered magic numbers)
# ----------------------------------------------------------------------

# Reliability thresholds (Spearman correlation of score vs. forward return)
RELIABILITY_DEFAULT_CUTOFF = 0.4   # _load_reliability_map: fallback "reliable" flag when CSV lacks one
RELIABILITY_ESTABLISHED = 0.2      # _compute_sell_signals: spearman needed to trust a negative score
RELIABILITY_INVERSE = -0.15        # _compute_sell_signals: spearman below this means score moves opposite to returns

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

# Tickers excluded from the main analysis loop.
# 1640718 — unidentified; original reason for exclusion unknown (not
# recoverable from the repo or git history), kept from prior behavior.
EXCLUDED_TICKER_IDS = {"1640718"}
