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

# Tickers excluded from the main analysis loop.
# 1640718 — unidentified; original reason for exclusion unknown (not
# recoverable from the repo or git history), kept from prior behavior.
EXCLUDED_TICKER_IDS = {"1640718"}
