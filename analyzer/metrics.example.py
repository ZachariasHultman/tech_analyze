# ----------------------------------------------------------------------
#  Portability template for analyzer/metrics.py
#
#  metrics.py is gitignored and load-bearing — every module imports
#  RATIO_SPECS, DIRECTION_OVERRIDES, weight tiers, and thresholds from it.
#  To set up on a new machine: `cp analyzer/metrics.example.py analyzer/metrics.py`
#  and tune the placeholder numbers below to taste (or rely on the JSON
#  optimization_results_*.json overrides loaded at runtime by main.py).
#
#  All metric *names* here are real (already public via summary_manager.py's
#  templates); all thresholds/weights/currency-map entries are placeholders.
# ----------------------------------------------------------------------
import pandas as pd

# =========================
# Static ticker currency maps
# =========================
ticker_reporting_currency_map = {}

# =========================
# Weights
# =========================
# Tune by moving metric names between these three tiers (weight 2 / 1.5 / 1).
# Metrics in none of the three tiers score with weight 0 (excluded).
HIGHEST_WEIGHT_METRICS = []

HIGH_WEIGHT_METRICS = []

LOW_WEIGHT_METRICS = [
    "roe_pe ratio status",
    "cagr_pe ratio status",
    "fcfy_pe ratio status",
    "roe_de ratio status",
    "net debt - ebitda status",
    "revenue trend year status",
    "nav discount status",
    "calculated nav discount status",
    "nav discount trend status",
    "revenue y cagr status",
    "eps y cagr status",
    "net margin vs avg status",
    "roe vs avg status",
    "revenue yoy hit-rate status",
    "eps yoy hit-rate status",
    "gross margin stability status",
    "piotroski f-score status",
    "dividend yield status",
    "price momentum status",
    "earnings quality status",
    "dividend growth status",
]

# =========================
# Sector-agnostic ratio specs
# Direction: +1 => higher is better
# num_is_rate=True => treat ROE/CAGR/FCFY as fractions, convert to percent before dividing
# thr=(NOK, OK) global bands used by SummaryManager._assign_points — PLACEHOLDER VALUES
# =========================
RATIO_SPECS = {
    "roe_pe ratio status": {
        "num": "roe",
        "den": "pe",
        "dir": +1,
        "num_is_rate": True,
        "thr": (0.0, 1.0),
    },
    "cagr_pe ratio status": {
        "num": "cagr",
        "den": "pe",
        "dir": +1,
        "num_is_rate": True,
        "thr": (0.0, 1.0),
    },
    "fcfy_pe ratio status": {
        "num": "fcfy",
        "den": "pe",
        "dir": +1,
        "num_is_rate": True,
        "thr": (0.0, 1.0),
    },
    "roe_de ratio status": {
        "num": "roe",
        "den": "de",
        "dir": +1,
        "num_is_rate": True,
        "thr": (0.0, 1.0),
        "den_floor": 0.0,  # clamp D/E to prevent blow-up with near-zero debt
    },
    "net debt - ebitda status": {
        "num": "net_debt",
        "den": "ebitda",
        "dir": -1,
        "num_is_rate": False,
        "thr": (0.0, 1.0),
    },
}

# =========================
# Direction overrides for non-ratio metrics where lower is better.
# Default direction for non-ratio metrics is +1 (higher is better).
# =========================
DIRECTION_OVERRIDES = {
    "gross margin stability status": -1,  # lower CV = more stable = better
    "net debt - ebitda status": -1,        # lower leverage = better
}

# =========================
# Global, sector-agnostic thresholds for non-ratio metrics
# IMPORTANT: all values are (NOK, OK) — PLACEHOLDER VALUES
# (use None to skip)
# =========================
GLOBAL_THRESHOLDS = {
    "revenue trend year status": (0.0, 1.0),
    "nav discount status": (0.0, 1.0),
    "calculated nav discount status": (0.0, 1.0),
    "nav discount trend status": (0.0, 1.0),
    "revenue y cagr status": (0.0, 1.0),
    "eps y cagr status": (0.0, 1.0),
    "net margin vs avg status": (0.0, 1.0),
    "roe vs avg status": (0.0, 1.0),
    "revenue yoy hit-rate status": (0.0, 1.0),
    "eps yoy hit-rate status": (0.0, 1.0),
    "gross margin stability status": (0.0, 1.0),
    "piotroski f-score status": (0.0, 1.0),
    "dividend yield status": (0.0, 1.0),
    "price momentum status": (0.0, 1.0),
    "earnings quality status": (0.0, 1.0),
    "dividend growth status": (0.0, 1.0),
}


# =========================
# Threshold accessor (sector-agnostic)
# SummaryManager still calls get_metrics_threshold(metric, sector),
# so we keep the signature but ignore sector.
# Returns: dict like {"nok": x, "ok": y} OR None
# =========================
def get_metrics_threshold(metric, sector=None):
    m = str(metric)

    # Ratios are handled in SummaryManager using RATIO_SPECS (dir + thr).
    # We return None here for ratios so the manager uses its own path.
    if m in RATIO_SPECS:
        return {"nok": RATIO_SPECS[m]["thr"][0], "ok": RATIO_SPECS[m]["thr"][1]}

    if m in GLOBAL_THRESHOLDS:
        nok, ok = GLOBAL_THRESHOLDS[m]
        # Allow None bands to propagate as "no scoring"
        if nok is None or ok is None:
            return None
        return {"nok": nok, "ok": ok}

    return None


# =========================
# Sector utilities (kept for compatibility; no longer used by thresholds)
# =========================
possible_sectors = [
    "Fordonsindustri",
    "Teknologi",
    "Industri",
    "Konsumentvaror & Tjänster",
    "Investmentbolag",
    "Energi",
    "Läkemedel",
    "Metall & Gruvdrift",
    "Papper & Skogsindustri",
    "Fastigheter & Utveckling",
    "Bank",
    "Försäkring",
    "Börs & Dataleverantörer",
    "Betalningslösningar & Transaktioner",
    "Bioteknik & Läkemedel",
    "Skog & Massaindustri",
]


def extract_sector(obj):
    """
    Return a canonical sector string from various shapes.
    Retained for compatibility; thresholds no longer depend on sector.
    """
    # STRING
    if isinstance(obj, str):
        s = obj.strip()
        return s if s in possible_sectors else "Unknown"

    # SINGLE DICT (accept sectorName or sector)
    if isinstance(obj, dict):
        for key in ("sectorName", "sector"):
            v = obj.get(key)
            if isinstance(v, str):
                v = v.strip()
                return v if v in possible_sectors else "Unknown"
        return "Unknown"

    # DATAFRAME with sectorName column → first matching value
    if isinstance(obj, pd.DataFrame):
        if "sectorName" in obj.columns:
            for name in obj["sectorName"]:
                name = str(name).strip()
                if name in possible_sectors:
                    return name
        return "Unknown"

    # SERIES → apply row-wise, return first non-Unknown
    if isinstance(obj, pd.Series):
        extracted = obj.apply(extract_sector)
        for v in extracted:
            if isinstance(v, str) and v in possible_sectors:
                return v
        return "Unknown"

    # LIST or nested LIST → list of dicts with sectorName
    if isinstance(obj, list) and len(obj) > 0:
        lst = obj[0] if isinstance(obj[0], list) else obj
        for entry in lst:
            if isinstance(entry, dict):
                v = entry.get("sectorName")
                if isinstance(v, str):
                    v = v.strip()
                    if v in possible_sectors:
                        return v
        return "Unknown"

    return "Unknown"
