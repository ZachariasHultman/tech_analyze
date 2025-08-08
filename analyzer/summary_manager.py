from analyzer.metrics import (
    HIGHEST_WEIGHT_METRICS,
    HIGH_WEIGHT_METRICS,
    LOW_WEIGHT_METRICS,
    get_metrics_threshold,
    extract_sector,
)
from tabulate import tabulate
from itertools import product


class SummaryManager:
    def __init__(self):
        # Initialize the attributes
        self.summary = {}
        self._threshold_overrides = {}  # dict to hold custom thresholds
        self.template = {
            "sector": None,
            "points": 0,
            "cagr-pe compare status": None,
            "roe status": None,
            "de status": None,
            "peg status": None,
            "fcfy status": None,
            "net debt - ebitda status": None,
            "revenue trend quarter status": None,
            "revenue trend year status": None,
            "profit margin status": None,
            "profit per share trend status": None,
            "profit margin trend status": None,
            "sma200 slope status": None,
        }
        self.summary_investment = {}
        self.template_investment = {
            "sector": None,
            "points": 0,
            "cagr-pe compare status": None,
            "roe status": None,
            "nav discount status": None,
            "calculated nav discount status": None,
            "nav discount trend status": None,
            "net debt - ebit status": None,
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

        return bool(summary)  # Returns False if empty, True if it has keys

    def _update(self, ticker, sector, metric, value):
        """
        Updates the summary for the given key with calculated metrics.

        :param ticker: The ticker that is associated with the values
        :param sector: The sector ('summary' or 'summary_investment') to update
        :param metric: The metric to update (e.g., 'sma200status')
        :param value: The value that is used for deciding the metric status, given by ticker
        :param compare_value: The value that is used for decision, set manually
        :param threshold: The calculated threshold for metric, set manually
        """
        # print(value,compare_value)
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

        # Ensure the metric is valid
        if metric not in template:
            raise KeyError(f"The metric '{metric}' is not valid for the {sector}.")

        # Ensure the ticker exists in the target dictionary
        if ticker not in target:
            raise KeyError(
                f"The ticker '{ticker}' does not exist in the {sector}. Did you forget to initialize it?"
            )

        # print("ticker, metric, value, compare_value",ticker, metric, value, compare_value)
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
        weight = 0

        if metric in HIGHEST_WEIGHT_METRICS:
            weight = 2
        elif metric in HIGH_WEIGHT_METRICS:
            weight = 1.5
        elif metric in LOW_WEIGHT_METRICS:
            weight = 1
        return weight

    def _assign_points(self, row, metric, threshold_override=None):
        """
        Score one metric for one row using canonical (NOK, OK) thresholds.

        Conventions:
        - Threshold tuple is ALWAYS (NOK, OK).
        - Boundary is inclusive on the 'good' side:
            higher-is-better:  x >= OK  => +weight;  x <= NOK => -weight
            lower-is-better:   x <= OK  => +weight;  x >= NOK => -weight
        - Neutral zone between NOK and OK (strictly between).
        """
        # --- weight ---
        weight = self._assign_weight(metric)
        if weight == 0:
            return 0

        # --- direction map ---
        BAD_IF_HIGH = {
            "de status",
            "peg status",
            "net debt - ebitda status",
        }  # lower is better
        direction = (
            -1 if any(k in metric for k in BAD_IF_HIGH) else +1
        )  # +1 => higher-better, -1 => lower-better

        # --- thresholds source ---
        sector = row.get("sector") or row.get("sectorName") or extract_sector(row)
        cfg = (
            threshold_override
            if threshold_override is not None
            else get_metrics_threshold(metric, sector)
        )
        if cfg is None:
            return 0

        # --- helpers ---
        def _normalize_pair(nok, ok):
            nok, ok = float(nok), float(ok)
            if direction == +1 and nok > ok:
                nok, ok = min(nok, ok), max(nok, ok)
            elif direction == -1 and nok < ok:
                nok, ok = max(nok, ok), min(nok, ok)
            return nok, ok

        def _score_scalar(x, nok, ok):
            if direction == +1:  # higher is better
                if x >= ok:
                    return +weight
                if x <= nok:
                    return -weight
                return 0
            else:  # lower is better
                if x <= ok:
                    return +weight
                if x >= nok:
                    return -weight
                return 0

        def _extract_pair(cfg_obj):
            """Return (pair, is_composite) where:
            - pair is (nok, ok) or ((nok1,nok2),(ok1,ok2))
            - is_composite is True for the latter
            """
            # tuple/list (simple)
            if isinstance(cfg_obj, (list, tuple)) and len(cfg_obj) == 2:
                a, b = cfg_obj[0], cfg_obj[1]
                # composite if both sides are pairs
                if (
                    isinstance(a, (list, tuple))
                    and isinstance(b, (list, tuple))
                    and len(a) == 2
                    and len(b) == 2
                ):
                    return ((a[0], a[1]), (b[0], b[1])), True
                # simple numeric
                if all(isinstance(t, (int, float)) for t in (a, b)):
                    return (a, b), False
                return None, False

            # dict shapes
            if isinstance(cfg_obj, dict):
                if "thresholds" in cfg_obj:
                    return _extract_pair(cfg_obj["thresholds"])
                if {"nok", "ok"} <= set(cfg_obj.keys()):
                    return (cfg_obj["nok"], cfg_obj["ok"]), False
                if metric in cfg_obj:
                    return _extract_pair(cfg_obj[metric])

            # anything else -> not supported
            return None, False

        # --- get the value x to score ---
        v = row.get(metric)
        if isinstance(v, (list, tuple)) and len(v) >= 1:
            v = v[0]
        if v is None:
            return 0
        try:
            x = float(v)
        except Exception:
            return 0

        # --- extract thresholds (handles dict or tuple) ---
        pair, is_composite = _extract_pair(cfg)
        if pair is None:
            return 0

        # --- simple (nok, ok) ---
        if not is_composite and isinstance(pair, (list, tuple)) and len(pair) == 2:
            nok, ok = _normalize_pair(pair[0], pair[1])
            return _score_scalar(x, nok, ok)

        # --- composite: ((nok_a, nok_b), (ok_a, ok_b)) ---
        if is_composite:
            (nok1, nok2), (ok1, ok2) = pair
            nok1, ok1 = _normalize_pair(nok1, ok1)
            nok2, ok2 = _normalize_pair(nok2, ok2)

            # try to pull a 2-tuple value, else reuse x for both
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
            """Applies row-wise coloring, then drops `_score` and 'sector' columns."""

            if df.empty:
                return df
            df_colored = df.apply(colorize_row, axis=1)
            df_colored = df_colored.drop(
                columns=[
                    col
                    for col in df_colored.columns
                    if col.endswith("_score") or col == "sector"
                ],
                errors="ignore",
            )
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
        Process historical data into summary using optional thresholds.
        thresholds: dict of {metric: threshold}
        """
        # Store threshold overrides for use in _assign_points
        self._threshold_overrides = thresholds if thresholds else {}

        for _, row in historical_df.iterrows():
            ticker = row.get("company", "UNKNOWN")
            sector = row.get(sector_column, "UNKNOWN")

            sector_name = row.get(sector_column, "")
            sector_info = [{"sectorName": sector_name}]
            self._initialize_template(ticker, sector_info)
            for metric in metrics_to_use:
                template = (
                    self.template
                    if sector != "Investmentbolag"
                    else self.template_investment
                )
                if metric not in template:
                    continue  # skip metrics not valid for this sector

                value = (
                    row.get(metric)
                    or row.get(metric.strip())
                    or row.get(metric.strip().lower())
                )

                self._update(ticker, sector_info, metric, value)
