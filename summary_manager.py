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

    def _assign_points(self, df, metric):
        """Assigns points based on thresholds and updates the existing DataFrame in place."""
        weight = self._assign_weight(metric)

        # Get threshold values using the first row's sector
        result = get_metrics_threshold(metric, df["sector"])

        if result:
            [OK_threshold, NOK_threshold] = result["thresholds"]
            good_if_high = result["good_if_high"]
        else:
            # df[metric + "_score"] = 0
            return 0

        # print(
        #     f"Processing {metric}: OK={OK_threshold}, NOK={NOK_threshold}, , good if high = {good_if_high}, sector ={df['sector']}"
        # )

        value = df[metric]
        # Apply scoring logic and store it in a new column
        if metric == "cagr-pe compare status":
            score = (
                (1 * weight)
                if (
                    (value[0][0] is not None and value[0][0] >= OK_threshold[0])
                    or (value[0][1] is not None and value[0][1] <= OK_threshold[1])
                )
                and not (
                    (value[0][0] is not None and value[0][0] <= NOK_threshold[0])
                    and (value[0][1] is not None and value[0][1] >= NOK_threshold[1])
                )
                else (
                    -1 * weight
                    if (
                        (value[0][0] is not None and value[0][0] <= NOK_threshold[0])
                        and (
                            value[0][1] is not None and value[0][1] >= NOK_threshold[1]
                        )
                    )
                    else 0
                )
            )
        elif metric == "net debt - ebitda status" or metric == "net debt - ebit status":
            score = (
                1 * weight
                if value[0][0]
                and value[0][1] is not None
                and value[0][0] <= 2 * value[0][1]
                else (
                    -1 * weight
                    if value[0][0]
                    and value[0][1] is not None
                    and value[0][0] >= 2.5 * value[0][1]
                    else 0
                )
            )
        elif metric == "sma200 status":
            score = (
                1 * weight
                if value[0][0]
                and value[0][1] is not None
                and value[0][0] >= value[0][1]
                else (-1 * weight)
            )
        else:
            if good_if_high:
                score = (
                    (1 * weight)
                    if (value[0] is not None and value[0] >= OK_threshold)
                    else (
                        (-1 * weight)
                        if (value[0] is not None and value[0] <= NOK_threshold)
                        else 0
                    )
                )
            else:
                score = (
                    1 * weight
                    if value[0] is not None and value[0] <= OK_threshold
                    else (
                        -1 * weight
                        if value[0] is not None and value[0] >= NOK_threshold
                        else 0
                    )
                )
        return score

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
