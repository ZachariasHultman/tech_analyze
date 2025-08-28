from analyzer.metrics import (
    HIGHEST_WEIGHT_METRICS,
    HIGH_WEIGHT_METRICS,
    LOW_WEIGHT_METRICS,
    RATIO_SPECS,
    get_metrics_threshold,
    extract_sector,
)
from tabulate import tabulate
import math


# base fields needed to compute ratios (always accepted by _update)
REQUIRED_BASE_COLS = {"roe", "pe", "cagr", "fcfy", "de"}


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


class SummaryManager:
    def __init__(self):
        # Initialize the attributes
        self.summary = {}
        self._threshold_overrides = {}  # dict to hold custom thresholds

        # ---- STANDARD TEMPLATE (sector-agnostic) ----
        self.template = {
            "sector": None,
            "points": 0,
            # ratios (computed and stored during ingestion)
            "roe_pe ratio status": None,
            "cagr_pe ratio status": None,
            "fcfy_pe ratio status": None,
            "roe_de ratio status": None,
            # sector-agnostic trends/technical
            "revenue trend quarter status": None,
            "revenue trend year status": None,
            "profit margin trend status": None,
            "profit per share trend status": None,
            "sma200 slope status": None,
        }

        self.summary_investment = {}
        # ---- INVESTMENT TEMPLATE (keeps NAV items) ----
        self.template_investment = {
            "sector": None,
            "points": 0,
            # ratios (computed and stored during ingestion)
            "roe_pe ratio status": None,
            "cagr_pe ratio status": None,
            "fcfy_pe ratio status": None,
            "roe_de ratio status": None,
            # investment-specific items
            "nav discount status": None,
            "calculated nav discount status": None,
            "nav discount trend status": None,
            # trends/technical
            "profit per share trend status": None,
            "sma200 slope status": None,
        }

    def _initialize_template(self, key, sector):
        """
        Ensure the given key exists in the specified sector and initialize it with the template.
        """
        target = (
            self.summary
            if sector[0]["sectorName"] != "Investmentbolag"
            else self.summary_investment
        )
        template = (
            self.template
            if sector[0]["sectorName"] != "Investmentbolag"
            else self.template_investment
        )

        if key not in target:
            target[key] = template.copy()  # Create a fresh copy for the key

    def _is_summary_initialized(self, sector=""):
        """
        Checks if self.summary contains at least one company key.
        Returns False if empty, True otherwise.
        """
        summary = (
            self.summary if sector != "Investmentbolag" else self.summary_investment
        )
        return bool(summary)

    def _update(self, ticker, sector, metric, value):
        """
        Updates the summary for the given key with calculated metrics.
        Accepts template metrics AND base ratio inputs (roe, pe, cagr, fcfy, de).
        """
        target = (
            self.summary
            if sector[0]["sectorName"] != "Investmentbolag"
            else self.summary_investment
        )
        template = (
            self.template
            if sector[0]["sectorName"] != "Investmentbolag"
            else self.template_investment
        )

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
        if sector[0]["sectorName"] != "Investmentbolag":
            self.summary[ticker][metric] = [value]
            self.summary[ticker]["sector"] = [sector]
        else:
            self.summary_investment[ticker][metric] = [value]
            self.summary_investment[ticker]["sector"] = [sector]
        return True

    def _assign_weight(self, metric):
        if metric in HIGHEST_WEIGHT_METRICS:
            return 2
        if metric in HIGH_WEIGHT_METRICS:
            return 1.5
        if metric in LOW_WEIGHT_METRICS:
            return 1
        return 0

    def _assign_points(self, row, metric, threshold_override=None):
        weight = self._assign_weight(metric)
        if weight == 0:
            return 0

        # Direction: use RATIO_SPECS for ratios; default higher-better otherwise
        direction = RATIO_SPECS[metric]["dir"] if metric in RATIO_SPECS else +1

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
            if direction == +1:
                if x >= ok:
                    return +weight
                if x <= nok:
                    return -weight
                return 0
            else:
                if x <= ok:
                    return +weight
                if x >= nok:
                    return -weight
                return 0

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
        if s1 > 0 and s2 > 0:
            return +weight
        if s1 < 0 and s2 < 0:
            return -weight
        return 0

    def _display(self, save_df=False):
        """Displays DataFrame with tabulate formatting for a cleaner output."""

        def colorize_row(row):
            """Applies color to metric values based on their *_score column."""
            colored_row = row.copy()
            for col in (
                self.template
                if "nav discount status" not in row
                else self.template_investment
            ):
                score_col = col + "_score"
                if score_col in row:
                    if row[score_col] > 0:
                        colored_row[col] = (
                            f"\033[92m{row[col]}\033[0m"  # Green for positive score
                        )
                    elif row[score_col] < 0:
                        colored_row[col] = (
                            f"\033[91m{row[col]}\033[0m"  # Red for negative score
                        )

            return colored_row

        def process_dataframe(df):
            """Applies row-wise coloring, then drops raw/base columns and helper scores."""

            if df.empty:
                return df

            df_colored = df.apply(colorize_row, axis=1)

            # Columns to drop
            drop_cols = [
                col
                for col in df_colored.columns
                if col.endswith("_score")
                or col == "sector"
                or col.strip().lower() in {"pe", "cagr", "fcfy", "de", "roe"}
            ]

            df_colored = df_colored.drop(columns=drop_cols, errors="ignore")
            return df_colored

        # Process and print `summary`
        if not self.summary.empty:
            summary_colored = process_dataframe(self.summary).sort_values(
                by="points", ascending=False
            )
            if save_df:
                # Save the summary to a CSV file
                self.summary.to_csv("summary.csv")

            print(
                tabulate(
                    summary_colored,
                    headers="keys",
                    tablefmt="fancy_grid",
                    showindex=True,
                )
            )

        # Process and print `summary_investment`
        if not self.summary_investment.empty:
            summary_investment_colored = process_dataframe(
                self.summary_investment
            ).sort_values(by="points", ascending=False)
            if save_df:
                # Save the investment summary to a CSV file
                self.summary_investment.to_csv("summary_investment.csv")

            print(
                tabulate(
                    summary_investment_colored,
                    headers="keys",
                    tablefmt="fancy_grid",
                    showindex=True,
                )
            )


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
        target = (
            self.summary if sector != "Investmentbolag" else self.summary_investment
        )
        stored = target.get(ticker, {}).get(field)
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

        # 2) template metrics (trends/NAV/etc.)
        template = (
            self.template if sector != "Investmentbolag" else self.template_investment
        )
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

            num = _get_val_from_row_or_summary(row, ticker, sector, num_name)
            den = _get_val_from_row_or_summary(row, ticker, sector, den_name)

            num = _to_pct(num) if num_is_rate else _unwrap(num)
            ratio_val = _safe_div(num, den)

            if ratio_val is not None:
                self._update(ticker, sector_info, out_col, ratio_val)
            else:
                target = (
                    self.summary
                    if sector != "Investmentbolag"
                    else self.summary_investment
                )
                if out_col not in target[ticker]:
                    target[ticker][out_col] = [None]
