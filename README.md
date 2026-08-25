# Stock Screener

A fundamental stock screener that pulls data from the unofficial Avanza API and yfinance, scores companies on ~20 quality/value metrics, and checks whether those scores have historically predicted returns.

> **Disclaimer:** Uses an unofficial, unsupported Avanza API. No affiliation with Avanza. Not financial advice. I have no formal economics education — this is an experiment based on things I've read from smarter people.

---

## Setup

Requires a `.env` file with your Avanza credentials and (optionally) Gmail for email reports:

```
AVANZA_USERNAME=your_avanza_username
AVANZA_PASSWORD=your_avanza_password
AVANZA_TOTP_SECRET=your_totp_secret

# Legacy names (USERNAME / PASSWORD / MY_TOTP_SECRET) are still read as a
# fallback if the AVANZA_* vars above aren't set.

# Optional — for monthly email reports (same vars as stryket)
SMTP_USER=you@gmail.com
SMTP_PASSWORD=xxxx        # Gmail app password (not your login password)
                          # Generate at: myaccount.google.com/apppasswords
EMAIL_TO=you@gmail.com
EMAIL_FROM=you@gmail.com  # optional, defaults to SMTP_USER
```

Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

After syncing, `uv run tech-analyze` is available as a shortcut for `uv run python3 main.py`.

### New machine setup

Cloning the repo alone isn't enough to run it — the following are gitignored and must be transferred manually (scp from another machine that has them, or created fresh):

| File | Required? | Notes |
|---|---|---|
| `.env` | Yes | Avanza credentials (see above); nothing runs without it |
| `analyzer/metrics.py` | Yes | Scoring weights/thresholds — load-bearing, every module imports from it. Doesn't exist in a fresh clone. Either `cp analyzer/metrics.example.py analyzer/metrics.py` to bootstrap with placeholder values, or scp a tuned copy from an existing machine |
| `optimization_results_panel.json` | Optional | Optimized weights/thresholds loaded at runtime. Used when present, else `metrics.py`'s hardcoded defaults — see "Weight variants" below |
| `data/yahoo_prices/` + `data/yahoo_symbols.json` | Mac only | Dividend-adjusted daily closes, 2015→today. Rebuildable with `--backfill-prices`, but that's a slow rate-limited run — scp it if you have it. Without it the panel drops from 7 fiscal years to 4 |
| `data/fx_sek.csv` | Mac only | ECB daily rates, so the panel's forward return is in SEK. Rebuildable in seconds with `--fetch-fx` |

Neither cache is needed on the Pi — they only matter when building the panel. None of these are ever committed (`*.json`/`*.csv` are blanket-ignored, and `metrics.py` is explicitly excluded).

---

## How it scores stocks

Each stock is scored on metrics like Piotroski F-Score, revenue and EPS CAGR, ROE/PE ratio, net debt/EBITDA, gross-margin stability, dividend growth, and price momentum.

Scoring is **cross-sectional** — each metric is ranked relative to all other stocks in the current run (percentile 0–1), so a utility with solid-for-utilities ROE ranks in the top half just like a tech company with high ROE. No sector-specific thresholds needed.

Each metric feeds into two "sleeves" — **quality** (business health: Piotroski F-Score, margin stability, ROE, leverage, etc.) and **value** (cheapness: ROE/PE, CAGR/PE, FCF yield/PE, dividend growth, momentum). `combined_score = quality_pct × value_pct`, and both sleeves must clear `SLEEVE_GATE_MIN` (0.40) for a stock to qualify at all.

The **ranking key within that qualifying set is `blend_score`**: half the stock's price-vs-SMA200 percentile, half its `points` percentile. See "The ranking key" below.

It's a cross-sectional tool — good for ranking a basket of stocks against each other, not for predicting any single stock's outcome.

### The ranking key

```
blend_score = 0.5 × momentum_pct + 0.5 × fundamental_pct
```

Both halves are percentile ranks (0–1) within the current cross-section. Ranks rather than raw values because `points` spans roughly −6…+14 while momentum is a fraction in −0.08…+0.24 — averaging the raw numbers would hand the ranking to whichever has the wider numeric spread.

Momentum enters as the **raw** `(price − SMA200) / SMA200`, not as its banded metric score. Ranking on the banded score measures IC +0.092 (p=0.117, positive 5 of 7 years); ranking on the underlying ratio measures +0.112 (p=0.045, 7 of 7). The ok/nok banding discards roughly a fifth of the signal of the strongest single predictor in the system.

The 0.5 is not fitted. With 7 annual observations any fitted split is noise — the weight optimizer has been rejected three times, permutation p climbing 0.199 → 0.408 → 0.512. Equal weight between two comparably predictive, weakly correlated (~0.29) signals is the defensible default.

Momentum also stays inside `points`, so its effective weight is ≈0.52. Stripping it out scored the same on IC but worse on the basket (+4.65 vs +5.90%/yr, measured on the panel as it stood at the time — not re-run since the SEK/adjusted-price rebuild).

`points` is unchanged and still drives the sell-signal deterioration check.

### What is and isn't established

Measured on 7 fiscal years (2019–2025, 811 company-years, dividend-inclusive returns in SEK), last refreshed 2026-08-25:

| Claim | Evidence |
|---|---|
| The ranking beats chance | **Probably, but it no longer clears p<0.05.** `blend_score` IC +0.118, p=0.052, positive in **7 of 7** years. Fama-MacBeth coef +0.1547, t=+2.32, p=0.059 |
| The blend beats ranking on `points` alone | **On the measured window, yes.** Top-25 basket +6.36%/yr excess vs +2.59%/yr — but neither is significant (p=0.146 and p=0.495) and both are positive in only 4 of 7 years |
| The concentrated pick beats the universe | **No.** p=0.146. The consistent result is the *ordering* (7 of 7 years), not the tail |
| Optimized weights beat equal weight | **No.** Rejected again on 2026-08-25: out-of-sample spread +4.8%/yr against equal weight's +8.1%/yr, so it failed on the spread condition before the DSR (0.943) even mattered. The system runs equal weights |

**The two rows about the basket used to read better than they do now.** Earlier revisions of this table reported the top-25 pick at p=0.086 and 6 of 7 years positive, and Fama-MacBeth at p=0.029. Those came from a shallower, price-only, mixed-currency panel. Rebuilding it on Yahoo adjusted closes converted to SEK raised the mean IC and the mean basket return while *lowering* consistency and significance — a reminder that the tail statistics move a lot on 7 observations, and that the whole-ordering IC is the more stable of the two measurements.

IC measures the whole ordering; the basket measures only the tail. A 10-stock bucket at ~36% cross-sectional return dispersion has a standard error of ~11.5%/yr — **a 2%/yr edge there would take ~208 years to detect**. That is a permanent consequence of universe size, not a data gap to be closed, which is why `--push-top` defaults to a quintile: information ratio scales as IC × √breadth.

**Survivorship bias is not corrected for** (Avanza exposes no point-in-time index membership) and inflates all of the above — *and it inflates momentum more than any other metric here*, because the universe is today's list applied backwards and survivors are precisely the stocks that trended up. That is the main reason the ranking key is a blend rather than pure momentum, despite momentum alone measuring higher on the basket (+6.82%/yr, 5 of 7 years, p=0.144 — nominally better than the blend's 4 of 7 on the current panel).

### Metrics that were removed

Four metrics were deleted from scoring outright rather than down-weighted:

| Metric | Why |
|---|---|
| `dividend yield status` | Measured IC **−0.026** (p=0.76) — nothing, even after its look-ahead leak was fixed. Note the forward target already includes dividends, so this says *yield doesn't predict total return*, not that dividends don't matter |
| `earnings quality status` | Avanza exposes no historical OCF series, so the column was permanently all-NaN in the panel |
| `revenue yoy hit-rate status` | 0% coverage in the panel |
| `eps yoy hit-rate status` | 0% coverage in the panel |

The last three are the important case. All were ~100% populated **live**, so they were shifting the live ranking while contributing exactly nothing to the backtest that validated it. Removing them makes the running system the one that was actually measured, rather than a superset of it.

Still present and measuring *negative*, though none significantly so: `eps y cagr` (−0.071), `revenue trend year` (−0.055), `dividend growth` (−0.042). Left in — removing metrics because they measure mildly negative on 7 observations is fitting to noise.

### Investment companies are excluded

Öresund, Investor, Industrivärden, Kinnevik, Latour and Lundbergföretagen are **not scored and not pushed** (`config.EXCLUDED_SECTORS`).

They used to be, through a separate template carrying three NAV-discount metrics no other company had — which meant they were scored on **8 metrics against everyone else's ~17**, and `points`, a *sum* of weighted metric scores, was then ranked across both groups for the same 25 slots. Summing 8 things and 17 things and comparing the totals is the same two-ranking-regimes defect that scoring the two pools separately used to cause.

It had also never been measured. `_iter_fiscal_years` derives report dates from `revenue_year`, which their snapshots don't carry, so they produced **zero panel rows** — every number in the tables above always described the other 127 companies only. Removing them changed the panel not at all (811 company-years, 127 companies, IC +0.118 before and after); it just stopped the live list from ranking six names on a basis nothing had validated.

Routing them through the regular template instead was the obvious alternative and is worse: their snapshots also lack `de_ratio`, `eps_quarter`, `revenue_quarter` and `netDebtEbitdaRatio`, so they would score on a handful of metrics and go on competing — the same defect, quieter. Bringing them back properly means giving `_iter_fiscal_years` a dated fallback (`net_profit`/`profit_per_share` are present and FULL_YEAR-dated) *and* deciding whether partial-coverage companies belong in the same cross-section as fully-covered ones. Not attempted.

(An earlier per-company/per-sector "reliability" score, tracking whether an individual company's own score history predicted its own returns, was tried and dropped — the data doesn't have enough depth per company to support it, see git history if curious.)

---

## Daily use

```bash
# Score your stocks (default: watchlist named "Test")
uv run python3 main.py

# Use a different personal watchlist
uv run python3 main.py --watchlists Utdelare

# Use multiple watchlists at once (deduplicates)
uv run python3 main.py --watchlists Test Utdelare Äger

# Use the built-in OMXS30 preset (~30 stocks, searched by name)
uv run python3 main.py --preset omxs30

# Combine OMXS30 + Mid Cap for a broad universe (~55 stocks)
uv run python3 main.py --preset omxs30 omxs-mid

# Combine a personal watchlist with a preset
uv run python3 main.py --watchlists Test --preset omxs30

# Push top 25 scoring stocks to "Bör köpa" (default)
uv run python3 main.py --push

# Push top 5 to a different watchlist
uv run python3 main.py --push --push-top 5 --push-to "Min lista"
```

---

## Expanding the universe

The more stocks you include, the more reliable the correlation analysis becomes. Available presets:

| Preset | Contents | ~Size |
|---|---|---|
| `omxs30` | OMXS30 large caps | 30 |
| `omxs-mid` | OMXS Mid Cap selection | 25 |

```bash
# Broadest built-in universe
uv run python3 main.py --preset omxs30 omxs-mid
```

Presets use `search_for_stock` to look up each company by name, so they don't rely on hardcoded IDs that can go stale.

> **Note on S&P 500 / US stocks:** The `get_analysis()` endpoint only returns meaningful financial data for Nordic stocks. US stocks silently produce empty metrics. Stick to Nordic lists.

---

## Workflow

### One-time setup

**Step 1 — Save historical data for a broad universe**

Aim for 30+ companies. Each save pulls multi-year financial history and 5 years of daily prices from Avanza in one go — you don't need to collect snapshots over months.

```bash
uv run python3 main.py --save --preset omxs30 omxs-mid --watchlists Test Utdelare Äger
```

**Step 2 — Backfill adjusted prices and FX rates (Mac only, one-time)**

```bash
uv run python3 main.py --backfill-prices    # Yahoo adjusted closes back to 2015
uv run python3 main.py --fetch-fx           # ECB daily rates -> data/fx_sek.csv
```

Skipping these does not error — the panel silently falls back to Avanza's own prices in each stock's listing currency — but the panel it builds is **not** the one the numbers above describe. Both are worth understanding before you skip them:

- **`--backfill-prices`** fetches dividend-adjusted daily closes from Yahoo into `data/yahoo_prices/`, verifies each series against the Avanza prices already in your snapshots (an unverifiable symbol is dropped, not guessed), and caches the symbol map in `data/yahoo_symbols.json`. Avanza's OHLC is a rolling, unadjusted ~5-year window; Yahoo's goes back to 2015. This is what takes the panel from 4 usable fiscal years to 7. It is resumable and rate-limit-prone — re-run it after a wall, and lower `--prices-batch-size` (default 8) or raise `--prices-cooldown` (default 600s) if it keeps hitting one. `--prices-from` sets the start date (default 2015-01-01).

  Re-running it also **brings the cache forward**: any symbol whose series ends more than `--prices-refresh-after` days ago (default 30) is refetched, everything newer is skipped so you don't pay the rate limit for data you already have. Pass a large number for the old skip-if-present behavior, or `0` to refetch the lot.
- **`--fetch-fx`** caches ECB daily reference rates (via Frankfurter — free, no API key) so forward returns can be taken in SEK. 44% of the universe is not SEK-listed and USD/SEK moved roughly +24% across the sample, so without it every US name in a given year carries the same FX move, and the within-year demeaning hands that shared move to whatever metric happens to correlate with being US-listed. Idempotent: re-running when the cache already covers the range makes no requests. `--fx-from` mirrors `--prices-from`.

Only the *target* is converted to SEK; the predictors stay in listing currency (see the module docstring in `analyzer/panel.py`). The Pi never needs either command — both caches only matter when building the panel.

**Step 3 — Build the panel and optimize metric weights**

```bash
uv run python3 main.py --backfill-panel --validate
uv run python3 main.py --optimize --challenger-confidence 0.925
```

`--backfill-panel` builds a fiscal-year cross-sectional panel from your saved `data/*.csv` snapshots (`data/panel_fundamentals.csv` + `data/panel_scores.csv`); `--validate` runs a diagnostic battery against it (quintile sorts, Information Coefficient, Fama-MacBeth) — useful for sanity-checking before optimizing, not required for daily use.

`--optimize` runs the challenger gate against that panel: coordinate descent over per-metric weights and thresholds, ranked within each historical fiscal year, with the top-vs-bottom quintile spread compared against equal-weighting via leave-one-fiscal-year-out cross-validation and acceptance gated on a Deflated Sharpe Ratio bar (`--challenger-confidence`, default 0.925). The verdict is written to `optimization_results_panel.json` on **both** accept and reject — on a reject the file holds the gate's own equal-weight fallback, so it is always a defensible recommendation. That file is the weight source for every future run, no flag needed.

> **These are two separate commands, not one.** `--backfill-panel` (and `--validate`) return before the optimizer ever runs, so `main.py --backfill-panel --optimize` silently does only the backfill. Always run them as two invocations, in that order. `--optimize` needs `data/panel_scores.csv` + `data/panel_fundamentals.csv` to exist and exits with an error if they don't.

**`--permutations` (default 200).** The DSR bar needs to know how good a result the search reaches by luck alone. It used to approximate that from the spread of objective values across grid candidates — a number that moves with how the grid was configured rather than with the evidence (the panel search evaluates ~1200 candidates but produces only ~90 distinct values, so duplicates skewed it in both directions at once). Instead the target is now shuffled *within* each fiscal year and the optimizer refitted 200 times, which measures the null directly. Costs a few minutes on a Mac; `--permutations 0` skips it and falls back to the old approximation, which the output then explicitly labels as not meaningful.

**What the gate prints.** Alongside the DSR you get `beat equal weight in N of M held-out years` and a permutation p-value. At four usable fiscal years no statistic has real power, so that blunt count is often the most informative line in the report.

> **Note:** two past bugs invalidate older weight files. (1) The backtest had a look-ahead bug — predictors measured across the same window as the return. (2) The forward-return target was price-only, while Avanza's OHLC close is not dividend-adjusted, so every high-yield stock was penalised by roughly its own yield. Both are fixed, but any `optimization_results_*.json` from before must be regenerated — re-run this step at least once after upgrading. The inert `optimization_results_individual.json` / `_combo.json` leftovers have been deleted; nothing reads them.

---

### Daily use

Optimized weights are loaded automatically — no extra flags needed.

```bash
# Score a watchlist and push top 25 to "Bör köpa"
uv run python3 main.py --watchlists Äger --push

# Just score, no push
uv run python3 main.py --watchlists Äger
```

---

### Maintenance

| When | Command |
|---|---|
| **After new quarterly earnings** | `uv run python3 main.py --save --preset omxs30 omxs-mid ...` |
| **After re-saving** | `--backfill-panel --validate`, then `--optimize` |
| **If you only changed a threshold in code** | `uv run python3 main.py --backfill-panel --validate` (rebuilds the panel and prints the diagnostics, without re-fitting weights) |
| **If you add new stocks to universe** | `--save`, then `--backfill-prices` (the new names have no Yahoo cache yet), then re-optimize |
| **Before rebuilding the panel** | `--fetch-fx` then `--backfill-prices` — both skip what's already current, and the second brings stale series forward |

Re-saving quarterly is enough. Re-optimizing is only needed after a re-save or when you've significantly expanded the universe.

**On refreshing the price cache.** `--backfill-prices` used to only download symbols it had no file for, so the cache never moved forward and the panel's most recent fiscal year kept reading a truncated series. It now refetches anything staler than `--prices-refresh-after` days (default 30) and skips the rest.

A refresh **refetches a symbol's whole history and replaces the file** rather than appending the missing tail, and that is deliberate: Yahoo restates its adjusted close *retroactively* every time a dividend or split lands, so the entire series rescales. Splicing a freshly-adjusted tail onto a stale history would put a step the size of the dividend at the seam — inside `close_adj`, which is exactly the series the panel computes forward returns from. That is the bug this repo already fixed once when the target was price-only; the bandwidth saved isn't worth reintroducing it. If a refetch gets rate-limited, the existing file stays put, so a refresh is never worse than not refreshing.

---

## Raspberry Pi deployment

The Pi only runs the weekly scoring job — no `--save`, no `--optimize`. Those stay on your Mac. The optimized weights (`optimization_results_panel.json`) are gitignored (`*.json`/`*.csv` are blanket-ignored) — never committed, so the file must be copied to the Pi manually via `scp` (see below) after each re-optimize.

### First-time Pi setup

```bash
# On Pi
git clone <your-repo-url> ~/tech_analyze
cd ~/tech_analyze
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Copy credentials and output files from Mac
scp .env pi@raspberrypi:~/tech_analyze/.env
scp optimization_results_panel.json pi@raspberrypi:~/tech_analyze/
```

### Set up the cron job

```bash
# On Pi
crontab -e
```

Add this line (runs at 08:00 every Monday; matches the schedule documented at the top of `cron_pi.sh`):
```
0 8 * * 1 /home/pi/tech_analyze/cron_pi.sh >> /home/pi/tech_analyze/logs/cron.log 2>&1
```

The cron job runs — and rebalances — weekly.

Turnover was measured at ~16% of the list per week, but 7 of the 8 removals across three consecutive runs were names still inside rank 35, i.e. boundary noise rather than signal. That churn came mostly from the two inputs that tracked spot price; both now divide by a 60-day mean close, and the ranking key is half an SMA200 comparison, so it is damped at the source rather than by rebalancing less often.

### Workflow between Mac and Pi

| Where | When | Action |
|---|---|---|
| Mac | After a re-save | `--backfill-panel --validate`, then `--optimize` |
| Mac | After optimize | `scp optimization_results_panel.json pi@raspberrypi:~/tech_analyze/` |
| Pi | Automatic (cron) | `cron_pi.sh` — pulls code via git, scores, pushes to Avanza, sends email |

### Manual Pi run (test it first)

```bash
ssh pi@raspberrypi
cd ~/tech_analyze
./cron_pi.sh
```

---

## Sell signals

Add `--sell-from Äger` to flag stocks in your portfolio whose fundamentals have deteriorated. A stock is flagged when its points score (`pts`) is negative.

```bash
uv run python3 main.py --watchlists Äger --sell-from Äger
```

Sell signals are also included automatically in the email when `--email` is used.

---

## Email reports

```bash
uv run python3 main.py --watchlists Äger --push --sell-from Äger --email
```

Sends a plain-text email with the watchlist update and any sell signals. Requires `EMAIL_*` vars in `.env` (see Setup).

Every email opens with a **system status block** describing the weights it is recommending with:

```
SYSTEM STATUS  (weights fitted 2026-08-25, panel challenger: REJECTED (running equal weight))
  Out-of-sample, 7 fiscal years (2019-2025), total return (price + dividends), demeaned within year:
    IC by year:   2019 +0.289   2020 +0.082   2021 +0.025   2022 -0.034   2023 +0.057   2024 +0.067   2025 +0.077
    mean IC +0.080  |  top-bottom spread +8.1%/yr
    challenger beat equal weight in 3 of 7 held-out years but missed the significance bar
    permutation test (200 refits on shuffled targets): p=0.149
    deflated Sharpe 0.943 vs 0.925 bar (cleared)
  CONFIDENCE: LOW — 7 periods only (t=+2.42, p=0.05). Directional, not proof.
  RANKING KEY (blend_score = 50% momentum + 50% pts), same 7 years:
    IC +0.118 (p=0.052, positive 7/7)
    top-25 basket +6.36%/yr excess (p=0.146, positive 4/7)
    Momentum is the metric survivorship bias most inflates — treat this as the optimistic end.
    Investment companies are not in the panel at all, so their placement in the list is unvalidated.
    Survivorship bias is not corrected for and inflates all of the above.
  Universe: 127 stocks (today's list applied backwards).
```

That is the block as of the last real run, reproduced verbatim. Note the DSR **cleared** its bar (0.943 > 0.925) and the challenger was still rejected: acceptance needs the optimized spread to beat equal weight *as well*, and it didn't (+4.8%/yr against +8.1%/yr). Two conditions, both required.

Read `ACCEPTED` vs `REJECTED` first: on a reject the system silently runs equal weight, and the reported numbers then describe equal weight rather than the challenger that was turned down. The `CONFIDENCE: LOW` line is permanent, not a placeholder — four fiscal years cannot support a significance claim, and a tidy top-10 list otherwise implies more than the data does.

The block is read from the `validation` section of `optimization_results_panel.json`, so the one file you already copy to the Pi carries it. A Pi holding an older file, or none, prints a short "unknown" notice and still sends a complete email.

---

## Weight variants

```bash
uv run python3 main.py --no-opt     # Use hardcoded default weights (ignore optimizer output)
uv run python3 main.py --use-panel  # Force the panel challenger's out-of-sample-validated weights
```

Default (no flag): reads `optimization_results_panel.json` when it exists — the out-of-sample, Deflated-Sharpe-Ratio-gated result (run `--backfill-panel` then `--optimize` once, see Workflow above) — falling back to hardcoded `metrics.py` defaults. `--use-panel` is the same thing, stated explicitly.

---

## Full CLI reference

| Flag | Description |
|---|---|
| `--watchlists NAME ...` | Personal Avanza watchlists to analyze |
| `--preset NAME ...` | Built-in presets: `omxs30`, `omxs-mid` |
| `--save` | Save today's metric snapshot to `data/` |
| `--optimize` | Run the panel challenger gate and write `optimization_results_panel.json` (needs `--backfill-panel` first) |
| `--backfill-panel` | Build the fiscal-year panel from `data/*.csv` (writes `data/panel_fundamentals.csv` + `data/panel_scores.csv`) |
| `--backfill-prices` | Mac-only one-time step: fetch Yahoo dividend-adjusted closes into `data/yahoo_prices/`, verified against the Avanza prices in your snapshots. Lifts the panel from 4 fiscal years to 7. Resumable |
| `--prices-from YYYY-MM-DD` | Earliest date for `--backfill-prices` (default `2015-01-01`) |
| `--prices-batch-size N` | Symbols per Yahoo request (default 8). Lower it if you keep getting rate-limited |
| `--prices-refresh-after DAYS` | Refetch a cached symbol whose series ends more than DAYS ago (default 30); newer ones are skipped. A refresh replaces the full history, it does not append. `0` refetches everything |
| `--prices-cooldown SECONDS` | Pause after a batch exhausts its retries (default 600). Within a batch the backoff is already 30s doubling to a 900s cap |
| `--fetch-fx` | Mac-only manual step: cache ECB daily rates to `data/fx_sek.csv` so forward returns are computed in SEK. Idempotent |
| `--fx-from YYYY-MM-DD` | Earliest date for `--fetch-fx` (default `2015-01-01`) |
| `--validate` | Run the cross-sectional validation battery against `data/panel_scores.csv` |
| `--challenger-confidence X` | Deflated Sharpe Ratio bar the panel challenger must clear to accept optimized weights (default 0.925) |
| `--permutations N` | Refits on shuffled targets used to measure the gate's null distribution (default 200; 0 skips) |
| `--no-opt` | Ignore saved weights, use hardcoded defaults |
| `--use-panel` | Force the panel challenger's out-of-sample-validated weights |
| `--push` | Push top-scoring stocks to an Avanza watchlist |
| `--push-to NAME` | Which watchlist to push to (default: `Bör köpa`) |
| `--push-top N` | How many stocks to push (default: 25 — breadth, see below) |
| `--sell-from NAME` | Check this watchlist for sell signals |
| `--email` | Send email summary (requires `EMAIL_*` in `.env`) |

If neither `--watchlists` nor `--preset` is given, both default to the same universe `cron_pi.sh` uses: `--preset omxs30 omxs-mid --watchlists Test Utdelare Äger Berkshire`. Giving either flag explicitly uses exactly what you passed, with no default mixed in.
