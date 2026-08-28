#!/usr/bin/env python3
"""Connect the README concept nodes to the code symbols they describe.

graphify extracts README.md semantically and the repo structurally, and the two
passes mint their own node ids -- `readme_sek_target_conversion` on one side,
`analyzer_panel_convert_closes_to_sek` on the other. Nothing reconciles them, so
the 58 README nodes come out as a fully disconnected component: asking "why does
the panel convert to SEK" and "what calls convert_closes_to_sek" hits two
separate graphs that share no edge.

The mapping below is curated by hand and every target is validated against the
graph before an edge is written, so an id that stops existing drops the edge
instead of creating a dangling one. Nothing here is fuzzy-matched: a label
match would connect `combined_score` correctly and then quietly invent
something for `Panel Challenger Gate`, which is exactly the failure the
EXTRACTED/INFERRED split exists to prevent.

Confidence follows that split honestly:

  EXTRACTED (1.0)  the README names the identifier itself -- a CLI flag, a
                   column name, a filename -- and it resolves to that symbol.
  INFERRED (0.85)  the README names a *concept* and this is the function that
                   implements it. A real reading, but a reading.

Nine README nodes are deliberately left unconnected, and that is the correct
result rather than a gap -- see INTENTIONALLY_UNLINKED. Six describe code that
no longer exists (the removed metrics, the NAV template, the dropped
reliability score); three are methodology caveats about the pipeline as a
whole. Linking either group to a surviving function would assert a
relationship the codebase does not have, purely to drive an orphan count to
zero.

    python3 scripts/graph_readme_links.py --dry-run
    python3 scripts/graph_readme_links.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ORIGIN = "readme-link"

# (readme node id, target symbol label, target source_file, confidence)
# "E" = the README names this identifier verbatim; "I" = concept -> implementation.
MAPPING: list[tuple[str, str, str, str]] = [
    # --- CLI flags: the README names the flag, main() dispatches to these ----
    ("readme_backfill_prices_flag", "backfill_prices()", "analyzer/yahoo_prices.py", "E"),
    ("readme_fetch_fx_flag", "fetch_sek_rates()", "analyzer/fx.py", "E"),
    ("readme_backfill_panel_flag", "build_fundamentals_panel()", "analyzer/panel.py", "E"),
    ("readme_backfill_panel_flag", "build_scores_panel()", "analyzer/panel.py", "E"),
    ("readme_validate_flag", "run_validation_battery()", "analyzer/validation.py", "E"),
    ("readme_optimize_flag", "optimize_panel_combo()", "analyzer/correlation.py", "E"),
    ("readme_optimize_flag", "gate_optimized_weights()", "analyzer/correlation.py", "E"),
    ("readme_save_snapshots", "save_snapshot()", "analyzer/helper.py", "E"),
    ("readme_email_report", "_send_email()", "analyzer/main.py", "E"),
    ("readme_sell_signals", "_compute_sell_signals()", "analyzer/main.py", "E"),
    ("readme_push_watchlist", "_update_watchlist()", "analyzer/main.py", "E"),
    ("readme_prices_refresh_after", "_is_stale()", "analyzer/yahoo_prices.py", "E"),
    ("readme_universe_presets", "_resolve_universe()", "analyzer/main.py", "E"),
    ("readme_universe_presets", "_search_preset()", "analyzer/main.py", "I"),
    ("readme_weight_variants", "_load_optimized_params()", "analyzer/main.py", "I"),

    # --- scoring columns named verbatim in the README -----------------------
    ("readme_points", "calculate_score()", "analyzer/data_processing.py", "E"),
    ("readme_combined_score", "calculate_score()", "analyzer/data_processing.py", "E"),
    ("readme_quality_sleeve", "calculate_score()", "analyzer/data_processing.py", "E"),
    ("readme_value_sleeve", "calculate_score()", "analyzer/data_processing.py", "E"),
    ("readme_fundamental_pct", "calculate_score()", "analyzer/data_processing.py", "E"),
    ("readme_blend_score", "calculate_score()", "analyzer/data_processing.py", "E"),
    ("readme_blend_score", "rank_key()", "analyzer/summary_manager.py", "I"),
    ("readme_cross_sectional_scoring", "calculate_score()", "analyzer/data_processing.py", "I"),
    ("readme_sleeve_gate_min", "_update_watchlist()", "analyzer/main.py", "I"),
    ("readme_momentum_pct", "momentum_as_of()", "analyzer/panel.py", "E"),
    ("readme_momentum_pct", "calc_sma200_metrics()", "analyzer/financial_metrics.py", "I"),

    # --- statistics ---------------------------------------------------------
    ("readme_deflated_sharpe_ratio", "deflated_sharpe_ratio()", "analyzer/stats_utils.py", "E"),
    ("readme_fama_macbeth", "fama_macbeth()", "analyzer/stats_utils.py", "E"),
    ("readme_quintile_spread", "_quintile_spread()", "analyzer/correlation.py", "E"),
    ("readme_permutation_test", "permutation_benchmark()", "analyzer/correlation.py", "E"),
    ("readme_information_coefficient", "ic_time_series()", "analyzer/validation.py", "I"),
    ("readme_information_coefficient", "per_metric_ic()", "analyzer/validation.py", "I"),
    ("readme_challenger_gate", "gate_optimized_weights()", "analyzer/correlation.py", "I"),
    ("readme_confidence_low_caveat", "print_caveat_header()", "analyzer/validation.py", "I"),

    # --- panel construction and its invariants ------------------------------
    ("readme_fiscal_year_panel", "build_fundamentals_panel()", "analyzer/panel.py", "I"),
    ("readme_iter_fiscal_years", "_iter_fiscal_years()", "analyzer/panel.py", "E"),
    ("readme_sek_target_conversion", "convert_closes_to_sek()", "analyzer/panel.py", "I"),
    ("readme_dividend_inclusive_target", "forward_dividend_yield()", "analyzer/panel.py", "I"),
    ("readme_lookahead_bug", "slice_df_upto()", "analyzer/historical_calc.py", "I"),
    ("readme_turnover_damping", "smoothed_close()", "analyzer/financial_metrics.py", "I"),

    # --- data sources and caches -------------------------------------------
    ("readme_avanza_api", "setup_env()", "analyzer/main.py", "I"),
    ("readme_yfinance_source", "_download_batch()", "analyzer/yahoo_prices.py", "I"),
    ("readme_yahoo_price_cache", "load_symbol_prices()", "analyzer/yahoo_prices.py", "I"),
    ("readme_yahoo_price_cache", "load_verified_closes()", "analyzer/yahoo_prices.py", "I"),
    ("readme_fx_sek_cache", "load_sek_rates()", "analyzer/fx.py", "I"),
    ("readme_symbol_verification", "verify_symbol()", "analyzer/yahoo_prices.py", "E"),
    ("readme_full_refetch_policy", "backfill_prices()", "analyzer/yahoo_prices.py", "I"),
    ("readme_metrics_py", "get_metrics_threshold()", "analyzer/metrics.example.py", "I"),

    # --- artifacts, deployment, universe ------------------------------------
    ("readme_optimization_results_panel_json", "save_panel_optimization_results()",
     "analyzer/correlation.py", "E"),
    ("readme_optimization_results_panel_json", "_load_optimized_params()",
     "analyzer/main.py", "E"),
    ("readme_system_status_block", "_format_optimizer_status()", "analyzer/main.py", "I"),
    ("readme_cron_pi_sh", "cron_pi.sh", "cron_pi.sh", "E"),
    ("readme_pi_deployment", "cron_pi.sh", "cron_pi.sh", "I"),
    ("readme_excluded_investment_companies", "_is_excluded()", "analyzer/main.py", "I"),
    ("readme_stock_screener", "main()", "analyzer/main.py", "I"),
    ("readme_breadth_information_ratio", "_update_watchlist()", "analyzer/main.py", "I"),
    ("readme_equal_weight_default", "config.py", "analyzer/config.py", "I"),
]

# README nodes describing code that no longer exists. Asserted here so a future
# reader can see they were considered and deliberately left alone.
INTENTIONALLY_UNLINKED = {
    # Describe code that no longer exists.
    "readme_dividend_yield_status",
    "readme_earnings_quality_status",
    "readme_yoy_hit_rate_metrics",
    "readme_nav_discount_template",
    "readme_reliability_score_dropped",
    "readme_removed_metrics",
    # Methodology caveats about the whole pipeline. No single symbol implements
    # "the sample is survivorship-biased" or "keep mildly negative metrics";
    # picking a plausible-looking function for them would be inventing an edge
    # to make a number go to zero, which is the one thing this file must not do.
    "readme_survivorship_bias",
    "readme_nordic_only_constraint",
    "readme_keep_mildly_negative_metrics",
}

_CONF = {"E": ("EXTRACTED", 1.0), "I": ("INFERRED", 0.85)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default="graphify-out/graph.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = Path(args.graph)
    if not path.exists():
        print(f"error: {path} not found — run /graphify first", file=sys.stderr)
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in data["nodes"]}
    by_label: dict[tuple[str, str], str] = {}
    for n in data["nodes"]:
        if n.get("_origin") == "ast":
            by_label[(n["label"], n.get("source_file") or "")] = n["id"]

    existing = {(e["source"], e["target"], e.get("relation")) for e in data["links"]}
    added, skipped = [], []

    for readme_id, label, src_file, conf in MAPPING:
        target = by_label.get((label, src_file))
        if readme_id not in nodes:
            skipped.append(f"{readme_id} (no such README node)")
            continue
        if target is None:
            skipped.append(f"{label} in {src_file} (no such code node)")
            continue
        confidence, score = _CONF[conf]
        if (readme_id, target, "documents") in existing:
            continue
        data["links"].append({
            "source": readme_id,
            "target": target,
            "relation": "documents",
            "_origin": ORIGIN,
            "confidence": confidence,
            "confidence_score": score,
            "context": "README concept -> implementation",
            "source_file": "README.md",
            "source_location": None,
            "weight": 1.0,
        })
        existing.add((readme_id, target, "documents"))
        added.append(f"{nodes[readme_id]['label']}  --documents-->  {label}  [{confidence}]")

    for line in added:
        print(f"  + {line}")
    for line in skipped:
        print(f"  ! skipped: {line}")

    readme_ids = {n["id"] for n in data["nodes"]
                  if (n.get("source_file") or "") == "README.md"}
    linked = {e["source"] for e in data["links"] if e.get("_origin") == ORIGIN}
    orphans = readme_ids - linked - INTENTIONALLY_UNLINKED

    print(f"\n{len(added)} edge(s) added, {len(skipped)} skipped.")
    print(f"README nodes: {len(readme_ids)} total, {len(linked)} linked to code, "
          f"{len(INTENTIONALLY_UNLINKED)} intentionally unlinked (removed features), "
          f"{len(orphans)} still unlinked.")
    if orphans:
        print("  still unlinked: " + ", ".join(sorted(
            nodes[o]["label"] for o in orphans)))

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0
    if added:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n{path} updated. Re-run `graphify export html` to refresh.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
