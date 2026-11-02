from analyzer.metrics import (
    HIGHEST_WEIGHT_METRICS,
    HIGH_WEIGHT_METRICS,
    LOW_WEIGHT_METRICS,
    RATIO_SPECS,
    DIRECTION_OVERRIDES,
    get_metrics_threshold,
    extract_sector,
)
from analyzer.data_processing import _to_pct, _unwrap
from tabulate import tabulate
import math
import pandas as pd

# Cross-sectional rank thresholds used by _assign_points_rank.
# Bottom 35% of the peer group → negative score, top 35% → positive.
_RANK_NOK = 0.35
_RANK_OK = 0.65

# base fields needed to compute ratios (always accepted by _update)
REQUIRED_BASE_COLS = {"roe", "pe", "cagr", "fcfy", "de"}

# Ranking columns in fallback order. Both sort sites -- the printed/exported
# table (SummaryManager._display) and the pushed watchlist
# (main._update_watchlist) -- walk this one list, so they cannot drift apart.
RANK_COLUMNS = ("blend_score", "points", "combined_score")


def rank_key(df):
    """Sort key: blend_score, falling back to points, then combined_score.

    Returns None when the frame carries none of RANK_COLUMNS with a usable
    value, so callers can tell "nothing to rank on" apart from "everything
    ties at zero" -- _update_watchlist must not push an arbitrary order.

    This is the single implementation behind both sort sites. It used to be a
    closure inside _display, duplicated by hand in main._update_watchlist, and
    the two drifted twice: first when the ranking key moved from `points` to
    `blend_score` and only the watchlist side was updated, then again on the
    third rung, where _update_watchlist's intake filter dropped any summary
    without a `points` column outright -- so a combined_score-only
    cross-section printed a full table and pushed an empty watchlist. Keeping
    one function is what makes that class of drift unrepresentable rather than
    merely tested for.

    On why blend_score is the key at all, see config.MOMENTUM_BLEND_WEIGHT:
    over 7 fiscal years it carries the best ordering signal in the system
    (IC +0.118, positive 7 of 7, p=0.052) against points' +0.081 and
    combined_score's, which multiplies two noisy percentiles and amplifies the
    noise.
    """
    for col in RANK_COLUMNS:
        if col not in df.columns:
            continue
        base = pd.to_numeric(df[col], errors="coerce")
        if base.notna().any():
            return base.fillna(float("-inf"))
    return None


def _is_nan(x):
    try:
        return math.isnan(float(x))
    except Exception:
        return False


def _safe_div(a, b):
    try:
        if a is None or b is None:
            return None
        a = float(a)
        b = float(b)
        if _is_nan(a) or _is_nan(b) or b == 0.0:
            return None
        return a / b
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Display helpers.
#
# These were seven closures nested inside SummaryManager._display, which made
# that one method 191 lines of which 177 were function definitions. Six of the
# seven captured nothing from the enclosing scope and the seventh captured only
# `save_df`, now an explicit parameter -- so nesting them bought no state, only
# invisibility: a closure is not a node any static tool can see, which is
# exactly how the duplicated ranking logic in the old `_rank_key` drifted twice
# without anything noticing. Hoisted, each one is independently testable and
# shows up in the call graph.
# ---------------------------------------------------------------------------

def _unwrap_deep(v):
    while isinstance(v, (list, tuple)) and len(v) == 1:
        v = v[0]
    try:
        import numpy as np

        if isinstance(v, np.generic):
            v = v.item()
    except Exception:
        pass
    return v

def _fmt2(v):
    if v is None:
        return v
    try:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return f"{v:.2f}"
    except Exception:
        pass
    return v

def _colorize_row(row, template):
    colored = row.copy()
    for col in template:
        score_col = col + "_score"
        if score_col in row:
            v = row[col]
            uv = _unwrap_deep(v)
            if isinstance(uv, (int, float)) and uv is not None:
                if row[score_col] > 0:
                    colored[col] = f"\033[92m{uv:.2f}\033[0m"
                elif row[score_col] < 0:
                    colored[col] = f"\033[91m{uv:.2f}\033[0m"
                else:
                    colored[col] = f"{uv:.2f}"
            else:
                colored[col] = v
    return colored

def _build_export(df, template):
    """
    Build a CSV-friendly DataFrame:
    - unwrap lists/tuples
    - add '<metric> (flag)' columns using *_score (GREEN/RED/NEUTRAL)
    - drop *_score, sector, and base fields
    - round numeric to 2 decimals (leave flags as text)
    """
    import pandas as pd

    if df.empty:
        return df

    df2 = df.map(_unwrap_deep)

    # Create flags from *_score while we still have them
    def _flags_row(row):
        out = {}
        for col in template:
            sc = col + "_score"
            if sc in row:
                s = row[sc]
                flag = "GREEN" if s > 0 else ("RED" if s < 0 else "NEUTRAL")
                out[f"{col} (flag)"] = flag
        return pd.Series(out)

    flags = df2.apply(_flags_row, axis=1)
    export = pd.concat([df2, flags], axis=1)

    # Drop helper/base columns. combined_score is kept explicitly —
    # it ends in "_score" but is a headline sleeve output, not a
    # per-metric score column.
    drop_cols = [
        c
        for c in export.columns
        if c != "combined_score"
        and (
            str(c).endswith("_score")
            or c == "sector"
            or str(c).strip().lower() in {"pe", "cagr", "fcfy", "de", "roe"}
        )
    ]
    export = export.drop(columns=drop_cols, errors="ignore")

    # Round numeric to 2 dp (keep flags and strings as-is)
    def _round2(x):
        return round(x, 2) if isinstance(x, (int, float)) else x

    export = export.map(_round2)

    # Nice NA
    export = export.where(export.notna(), other="N/A")
    return export

def _process_dataframe(df, template):
    """For terminal view (with ANSI colors)."""
    if df.empty:
        return df
    df2 = df.map(_unwrap_deep)
    df2 = df2.apply(_colorize_row, axis=1, args=(template,))

    def _fmt_cell(x):
        if isinstance(x, str):  # leave colored strings
            return x
        return _fmt2(x)

    df2 = df2.map(_fmt_cell)

    drop_cols = [
        c
        for c in df2.columns
        if c != "combined_score"
        and (
            str(c).endswith("_score")
            or c == "sector"
            or str(c).strip().lower() in {"pe", "cagr", "fcfy", "de", "roe"}
        )
    ]
    df2 = df2.drop(columns=drop_cols, errors="ignore")

    if "points" in df2.columns:
        import pandas as pd

        df2["points"] = pd.to_numeric(df2["points"], errors="coerce")

    df2 = df2.where(df2.notna(), other="N/A")
    return df2

def _reorder_columns(df, weight_of):
    """Reorder metric columns by weight (highest first)."""
    # Separate special cols from metric cols
    metric_cols = [
        c for c in df.columns
        if c.endswith(" status") and not c.endswith("_score")
    ]
    other_cols = [c for c in df.columns if c not in metric_cols]

    # Sort metric columns by weight descending, then alphabetical
    metric_cols.sort(key=lambda m: (-weight_of(m), m))

    # Build final order: points, then the sleeve columns, then metrics,
    # then rest.
    ordered = []
    if "points" in other_cols:
        ordered.append("points")
        other_cols.remove("points")
    for sleeve in ("quality_pct", "value_pct", "combined_score"):
        if sleeve in other_cols:
            ordered.append(sleeve)
            other_cols.remove(sleeve)
    ordered.extend(metric_cols)
    ordered.extend(other_cols)

    # Only keep columns that exist
    ordered = [c for c in ordered if c in df.columns]
    return df[ordered]


def _sort_and_print(df, csv_name, save_df=False, *, template, weight_of):
    if df.empty:
        return

    # rank_key returns None when the frame carries no ranking column;
    # the table is still worth printing, just in its incoming order.
    key = rank_key(df)
    order = df.index if key is None else key.sort_values(ascending=False).index

    # CSV export first (clean + flags)
    if save_df:
        export_df = _build_export(df, template)
        export_df = _reorder_columns(export_df, weight_of)
        export_df = export_df.reindex(order)
        export_df.to_csv(csv_name)

    # Terminal view (colored)
    proc = _process_dataframe(df, template)
    proc = proc.reindex(order)
    proc = _reorder_columns(proc, weight_of)

    from tabulate import tabulate

    print(tabulate(proc, headers="keys", tablefmt="fancy_grid", showindex=True))


class SummaryManager:
    def __init__(self):
        # Initialize the attributes
        self.summary = {}
        self._threshold_overrides = {}  # dict to hold custom thresholds
        # Live-path daily closes and listing currency per company, kept for
        # analyzer/frontier.py's portfolio weights. Not scored, not saved.
        self.price_hist = {}
        self.price_ccy = {}

        # ---- STANDARD TEMPLATE (sector-agnostic) ----
        self.template = {
            "sector": None,
            "points": 0,
            # ratios
            "roe_pe ratio status": None,
            "cagr_pe ratio status": None,
            "fcfy_pe ratio status": None,
            "roe_de ratio status": None,
            "net debt - ebitda status": None,
            # growth
            "revenue y cagr status": None,
            "eps y cagr status": None,
            # quality
            "net margin vs avg status": None,
            "roe vs avg status": None,
            "gross margin stability status": None,
            # consistency
            "revenue trend year status": None,
            # shareholder return
            "dividend growth status": None,
            # composite quality
            "piotroski f-score status": None,
            # momentum
            "price momentum status": None,
        }

    def _initialize_template(self, key, sector):
        """
        Ensure the given key exists in the specified sector and initialize it with the template.
        """
        if key not in self.summary:
            self.summary[key] = self.template.copy()  # fresh copy for the key

    def _is_summary_initialized(self, sector=""):
        """
        Checks if self.summary contains at least one company key.
        Returns False if empty, True otherwise.
        """
        return bool(self.summary)

    def _update(self, ticker, sector, metric, value):
        """
        Updates the summary for the given key with calculated metrics.
        Accepts template metrics AND base ratio inputs (roe, pe, cagr, fcfy, de).
        """
        target = self.summary
        template = self.template

        # Ensure the metric is valid (template metric or required base input)
        if metric not in template and metric not in REQUIRED_BASE_COLS:
            raise KeyError(f"The metric '{metric}' is not valid for the {sector}.")

        # Ensure the ticker exists in the target dictionary
        if ticker not in target:
            raise KeyError(
                f"The ticker '{ticker}' does not exist in the {sector}. Did you forget to initialize it?"
            )

        self._perform_update(ticker, metric, value, sector)
        return True

    def _perform_update(self, ticker, metric, value, sector):
        # Update the metric in the summary
        self.summary[ticker][metric] = [value]
        self.summary[ticker]["sector"] = [sector]
        return True

    def _assign_weight(self, metric):
        # Live and the panel set equal weights here (correlation.equal_metric_weights)
        if hasattr(self, "_weight_overrides") and self._weight_overrides:
            return self._weight_overrides.get(metric, 0)
        if metric in HIGHEST_WEIGHT_METRICS:
            return 2
        if metric in HIGH_WEIGHT_METRICS:
            return 1.5
        if metric in LOW_WEIGHT_METRICS:
            return 1
        return 0

    def _assign_points(self, row, metric, threshold_override=None):
        """Score a metric using continuous linear interpolation.

        Instead of the old {-weight, 0, +weight} bucketing, scores are
        interpolated linearly between NOK and OK so that a value halfway
        between the thresholds gets ~0 rather than being lumped with the
        worst or best values.

        Returns a float in [-weight, +weight].
        """
        weight = self._assign_weight(metric)
        if weight == 0:
            return 0

        # Direction: RATIO_SPECS for ratios, DIRECTION_OVERRIDES for others, default +1
        if metric in RATIO_SPECS:
            direction = RATIO_SPECS[metric]["dir"]
        else:
            direction = DIRECTION_OVERRIDES.get(metric, +1)

        # Thresholds: explicit override -> per-run override -> ratio bands -> global bands
        if threshold_override is not None:
            cfg = threshold_override
        elif metric in self._threshold_overrides:
            cfg = self._threshold_overrides[metric]
        elif metric in RATIO_SPECS:
            nok, ok = RATIO_SPECS[metric]["thr"]
            cfg = {"nok": nok, "ok": ok}
        else:
            sector = row.get("sector") or row.get("sectorName") or extract_sector(row)
            cfg = get_metrics_threshold(metric, sector)

        if cfg is None:
            return 0

        def _normalize_pair(nok, ok):
            nok, ok = float(nok), float(ok)
            if direction == +1 and nok > ok:
                nok, ok = min(nok, ok), max(nok, ok)
            elif direction == -1 and nok < ok:
                nok, ok = max(nok, ok), min(nok, ok)
            return nok, ok

        def _score_scalar(x, nok, ok):
            """Continuous scoring with linear interpolation between NOK and OK.

            For higher-is-better (direction == +1):
              x >= ok      → +weight
              x <= nok     → -weight
              nok < x < ok → linearly interpolated from -weight to +weight

            For lower-is-better (direction == -1):
              x <= ok      → +weight
              x >= nok     → -weight
              ok < x < nok → linearly interpolated from +weight to -weight
            """
            if direction == +1:
                if x >= ok:
                    return +weight
                if x <= nok:
                    return -weight
                # linear interpolation: nok → -weight, ok → +weight
                span = ok - nok
                if span == 0:
                    return 0
                t = (x - nok) / span  # 0 at nok, 1 at ok
                return weight * (2 * t - 1)  # -weight at t=0, +weight at t=1
            else:
                if x <= ok:
                    return +weight
                if x >= nok:
                    return -weight
                # linear interpolation: ok → +weight, nok → -weight
                span = nok - ok
                if span == 0:
                    return 0
                t = (x - ok) / span  # 0 at ok, 1 at nok
                return weight * (1 - 2 * t)  # +weight at t=0, -weight at t=1

        def _extract_pair(cfg_obj):
            if isinstance(cfg_obj, (list, tuple)) and len(cfg_obj) == 2:
                a, b = cfg_obj[0], cfg_obj[1]
                if (
                    isinstance(a, (list, tuple))
                    and isinstance(b, (list, tuple))
                    and len(a) == 2
                    and len(b) == 2
                ):
                    return ((a[0], a[1]), (b[0], b[1])), True
                if all(isinstance(t, (int, float)) for t in (a, b)):
                    return (a, b), False
                return None, False
            if isinstance(cfg_obj, dict):
                if "thresholds" in cfg_obj:
                    return _extract_pair(cfg_obj["thresholds"])
                if {"nok", "ok"} <= set(cfg_obj.keys()):
                    return (cfg_obj["nok"], cfg_obj["ok"]), False
                if metric in cfg_obj:
                    return _extract_pair(cfg_obj[metric])
            return None, False

        # value to score (already computed ratios stored under status fields)
        v = row.get(metric)
        if isinstance(v, (list, tuple)) and len(v) >= 1:
            v = v[0]
        if v is None:
            return 0
        try:
            x = float(v)
        except Exception:
            return 0

        pair, is_composite = _extract_pair(cfg)
        if pair is None:
            return 0

        if not is_composite:
            nok, ok = _normalize_pair(pair[0], pair[1])
            return _score_scalar(x, nok, ok)

        (nok1, nok2), (ok1, ok2) = pair
        nok1, ok1 = _normalize_pair(nok1, ok1)
        nok2, ok2 = _normalize_pair(nok2, ok2)
        xv1, xv2 = x, x
        rv = row.get(metric)
        if (
            isinstance(rv, (list, tuple))
            and len(rv) >= 1
            and isinstance(rv[0], (list, tuple))
            and len(rv[0]) >= 2
        ):
            xv1, xv2 = rv[0][0], rv[0][1]
        s1 = _score_scalar(float(xv1), nok1, ok1)
        s2 = _score_scalar(float(xv2), nok2, ok2)
        # average the two component scores for composite metrics
        return (s1 + s2) / 2.0

    def _assign_points_rank(self, row, metric: str, rank_val: float) -> float:
        """Score a pre-computed cross-sectional percentile rank (0–1, 1=best).

        rank_val should already be direction-corrected so that 1.0 always
        means "best in the peer group" regardless of whether the raw metric
        is higher-is-better or lower-is-better.

        Symmetric thresholds: top 35% → +weight, bottom 35% → −weight,
        middle 30% → linear interpolation near zero.
        """
        weight = self._assign_weight(metric)
        if weight == 0:
            return 0.0
        if pd.isna(rank_val):
            return 0.0
        if rank_val >= _RANK_OK:
            return float(weight)
        if rank_val <= _RANK_NOK:
            return -float(weight)
        span = _RANK_OK - _RANK_NOK
        t = (rank_val - _RANK_NOK) / span  # 0 at NOK, 1 at OK
        return float(weight) * (2 * t - 1)

    def _display(self, save_df=False):
        """Displays DataFrame with tabulate formatting for a cleaner output."""
        if not self.summary.empty:
            _sort_and_print(self.summary, "summary.csv", save_df=save_df,
                            template=self.template, weight_of=self._assign_weight)

    def process_historical(
        self, historical_df, metrics_to_use, thresholds=None, sector_column="sector"
    ):
        """
        Ingest historical data:
        - Store base fields (roe, pe, cagr, fcfy, de) so ratios can be computed.
        - Store template metrics (trends/NAV).
        - Compute and store ratio status values alongside other fields.
        thresholds: optional dict {metric: threshold or {'nok': x, 'ok': y}}
        """
        self._threshold_overrides = thresholds if thresholds else {}

        def _get_val_from_row_or_summary(row, ticker, sector, field):
            v = row.get(field)
            if v is not None:
                return v
            stored = self.summary.get(ticker, {}).get(field)
            if isinstance(stored, (list, tuple)) and stored:
                return stored[0]
            return stored

        for _, row in historical_df.iterrows():
            ticker = row.get("company", "UNKNOWN")
            sector = row.get(sector_column, "UNKNOWN")

            sector_info = [{"sectorName": sector}]
            self._initialize_template(ticker, sector_info)

            # 1) base columns for ratios
            for base in REQUIRED_BASE_COLS:
                base_val = (
                    row.get(base) or row.get(base.strip()) or row.get(base.strip().lower())
                )
                if base_val is not None:
                    self._update(ticker, sector_info, base, base_val)

            # 2) template metrics (trends etc.)
            template = self.template
            for metric in metrics_to_use:
                if metric not in template:
                    continue
                value = (
                    row.get(metric)
                    or row.get(metric.strip())
                    or row.get(metric.strip().lower())
                )
                self._update(ticker, sector_info, metric, value)

            # 3) ratio values -> status fields
            for out_col, spec in RATIO_SPECS.items():
                num_name, den_name = spec["num"], spec["den"]
                num_is_rate = spec.get("num_is_rate", False)
                den_floor = spec.get("den_floor")

                num = _get_val_from_row_or_summary(row, ticker, sector, num_name)
                den = _get_val_from_row_or_summary(row, ticker, sector, den_name)

                num = _to_pct(num, force_convert=True) if num_is_rate else _unwrap(num)
                # Clamp denominator to floor to prevent blow-up (e.g. ROE/DE when DE≈0)
                if den_floor is not None and den is not None:
                    try:
                        den = float(_unwrap(den))
                        if abs(den) < den_floor:
                            den = den_floor if den >= 0 else -den_floor
                    except (TypeError, ValueError):
                        pass
                ratio_val = _safe_div(num, den)

                if ratio_val is not None:
                    self._update(ticker, sector_info, out_col, ratio_val)
                else:
                    if out_col not in self.summary[ticker]:
                        self.summary[ticker][out_col] = [None]
