# ----------------------------------------------------------------------
#  Centralized constants (formerly scattered magic numbers)
# ----------------------------------------------------------------------

# Reliability thresholds (Spearman correlation of score vs. forward return)
RELIABILITY_DEFAULT_CUTOFF = 0.4   # _load_reliability_map: fallback "reliable" flag when CSV lacks one
RELIABILITY_MIN_QUALIFY = 0.1      # _update_watchlist: minimum spearman to qualify for the buy watchlist
RELIABILITY_ESTABLISHED = 0.2      # _compute_sell_signals: spearman needed to trust a negative score
RELIABILITY_INVERSE = -0.15        # _compute_sell_signals: spearman below this means score moves opposite to returns
