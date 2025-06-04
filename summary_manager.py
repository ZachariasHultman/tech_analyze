from metrics import (
    HIGHEST_WEIGHT_METRICS,
    HIGH_WEIGHT_METRICS,
    LOW_WEIGHT_METRICS,
    get_metrics_threshold,
)
from tabulate import tabulate


class SummaryManager:
    def __init__(self):
        # Initialize the attributes
        self.summary = {}
        self.template = {
            "sector": None,
            "points": 0,
            "cagr-pe compare status": None,
            "fcf status": None,
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
            "fcf status": None,
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

    def _assign_points(self, row, metric):
        """
        Compute the score (−weight, 0, +weight) for *metric* on one DataFrame row.

        `row[metric]` is expected to be
            • a scalar, or
            • a tuple/list whose element-0 is the *latest* value and element-1 the history.
        """
        weight = self._assign_weight(metric)

        # ── Thresholds & direction ────────────────────────────────────────────────
        cfg = get_metrics_threshold(metric, row["sector"])
        if not cfg:  # no rule defined → no points
            return 0

        ok, nok = cfg["thresholds"]  # upper / lower band
        good_if_high = cfg["good_if_high"]

        value = row[metric]
        latest = value[0] if isinstance(value, (tuple, list)) else value

        # ── Metric-specific rules ────────────────────────────────────────────────
        if metric == "cagr-pe compare status":
            # here `value` is (cagr, pe)
            cagr, pe = value[0]
            if cagr is None or pe is None:
                return 0

            good = (cagr >= ok[0]) or (pe <= ok[1])
            bad = (cagr <= nok[0]) and (pe >= nok[1])

            if good and not bad:
                return weight
            elif bad:
                return -weight
            else:
                return 0

        elif metric == "net debt - ebitda status":
            if latest is None:
                return 0
            if latest <= 2:
                return weight
            if latest >= 2.5:
                return -weight
            return 0

        elif metric == "net debt - ebit status":
            if latest is None:
                return 0
            if latest <= 8:
                return weight
            if latest >= 12:
                return -weight
            return 0

        elif metric == "sma200 status":
            # value = (last_close, sma200)
            last_close, sma200 = value
            if last_close is None or sma200 is None:
                return -weight  # treat missing SMA as a negative signal
            return weight if last_close >= sma200 else -weight

        # ── Generic rule ─────────────────────────────────────────────────────────
        if latest is None:
            return 0

        if good_if_high:
            if latest >= ok:
                return weight
            elif latest <= nok:
                return -weight
        else:
            if latest <= ok:
                return weight
            elif latest >= nok:
                return -weight
        return 0

    def _display(self):
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
            print(
                tabulate(
                    summary_investment_colored,
                    headers="keys",
                    tablefmt="fancy_grid",
                    showindex=True,
                )
            )
