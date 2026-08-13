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
| `optimization_results_panel.json` / `_individual.json` / `_combo.json` | Optional | Optimized weights/thresholds loaded at runtime. Default prefers `_panel.json` (out-of-sample validated) if present, else `_individual.json`, else `metrics.py`'s hardcoded defaults — see "Weight variants" below |

None of these are ever committed (`*.json`/`*.csv` are blanket-ignored, and `metrics.py` is explicitly excluded).

---

## How it scores stocks

Each stock is scored on metrics like Piotroski F-Score, earnings quality (OCF/net income), revenue CAGR, ROE/PE ratio, net debt/EBITDA, price momentum, and dividend yield.

Scoring is **cross-sectional** — each metric is ranked relative to all other stocks in the current run (percentile 0–1), so a utility with solid-for-utilities ROE ranks in the top half just like a tech company with high ROE. No sector-specific thresholds needed.

Each metric feeds into two "sleeves" — **quality** (business health: Piotroski F-Score, margin stability, ROE, leverage, momentum, etc.) and **value** (cheapness: P/E, FCF yield, dividend yield). `combined_score = quality_pct × value_pct` is the actual ranking key, favoring stocks that are both good *and* cheap, not just one or the other.

It's a cross-sectional tool — good for ranking a basket of stocks against each other, not for predicting any single stock's outcome.

### What is and isn't established

Measured on 7 fiscal years (2019–2025, 813 company-years, dividend-inclusive returns):

| Claim | Evidence |
|---|---|
| The ranking beats chance | **Yes.** IC +0.041, t=+3.29, p=0.017, positive in 6 of 7 years |
| The concentrated pick beats the universe | **Not established.** Top-10 excess return +0.45%/yr, t=0.12, p=0.91 |
| Optimized weights beat equal weight | **No.** Rejected twice; permutation p=0.41, and optimized OOS IC is negative in 5 of 7 years. The system runs equal weights |

Those first two are not in tension: IC measures the whole ordering, the pick measures only the tail. A 10-stock bucket at ~36% cross-sectional return dispersion has a standard error of ~11.5%/yr — **a 2%/yr edge there would take ~208 years to detect**. That is a permanent consequence of universe size, not a data gap to be closed, which is why `--push-top` defaults to a quintile: information ratio scales as IC × √breadth, so breadth is how a weak-but-real ranking signal is actually harvested.

Survivorship bias is not corrected for (Avanza exposes no point-in-time index membership) and inflates all of the above.

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

**Step 2 — Build the panel and optimize metric weights**

```bash
uv run python3 main.py --backfill-panel --validate
uv run python3 main.py --optimize --optimize-combo --challenger-confidence 0.925
```

`--backfill-panel` builds a fiscal-year cross-sectional panel from your saved `data/*.csv` snapshots (`data/panel_fundamentals.csv` + `data/panel_scores.csv`); `--validate` runs a diagnostic battery against it (quintile sorts, Information Coefficient, Fama-MacBeth) — useful for sanity-checking before optimizing, not required for daily use.

`--optimize`/`--optimize-combo` compute per-metric correlations against historical returns and save weights to `optimization_results_individual.json` / `_combo.json`. If `data/panel_scores.csv` already exists (i.e. you ran `--backfill-panel` first), this also automatically runs an out-of-sample challenger test: rank companies by the optimized weights within each historical fiscal year, compare the top-vs-bottom quintile spread against equal-weighting via leave-one-fiscal-year-out cross-validation, and gate acceptance on a Deflated Sharpe Ratio bar (`--challenger-confidence`, default 0.925). If it clears the bar, the result is saved to `optimization_results_panel.json` — the default weight source for every future run, no flag needed.

> **These are two separate commands, not one.** `--backfill-panel` (and `--validate`) return before the optimizer ever runs, so `main.py --backfill-panel --optimize` silently does only the backfill. Always run them as two invocations, in that order.

**`--permutations` (default 200).** The DSR bar needs to know how good a result the search reaches by luck alone. It used to approximate that from the spread of objective values across grid candidates — a number that moves with how the grid was configured rather than with the evidence (the panel search evaluates ~1200 candidates but produces only ~90 distinct values, so duplicates skewed it in both directions at once). Instead the target is now shuffled *within* each fiscal year and the optimizer refitted 200 times, which measures the null directly. Costs a few minutes on a Mac; `--permutations 0` skips it and falls back to the old approximation, which the output then explicitly labels as not meaningful.

**What the gate prints.** Alongside the DSR you get `beat equal weight in N of M held-out years` and a permutation p-value. At four usable fiscal years no statistic has real power, so that blunt count is often the most informative line in the report.

> **Note:** two past bugs invalidate older weight files. (1) The backtest had a look-ahead bug — predictors measured across the same window as the return. (2) The forward-return target was price-only, while Avanza's OHLC close is not dividend-adjusted, so every high-yield stock was penalised by roughly its own yield. Both are fixed, but any `optimization_results_*.json` from before must be regenerated — re-run this step at least once after upgrading.

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
| **After re-saving** | `--backfill-panel --validate`, then `--optimize --optimize-combo` |
| **If you only changed a threshold in code** | `uv run python3 main.py --correlate` (skips optimize, just shows the correlation report) |
| **If you add new stocks to universe** | Re-save then re-optimize |

Re-saving quarterly is enough. Re-optimizing is only needed after a re-save or when you've significantly expanded the universe.

---

## Raspberry Pi deployment

The Pi only runs the weekly scoring job — no `--save`, no `--optimize`. Those stay on your Mac. The optimized weights (`optimization_results_panel.json` and/or `_individual.json`) are gitignored (`*.json`/`*.csv` are blanket-ignored) — they're never committed, so they must be copied to the Pi manually via `scp` (see below) after each re-optimize.

### First-time Pi setup

```bash
# On Pi
git clone <your-repo-url> ~/tech_analyze
cd ~/tech_analyze
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Copy credentials and output files from Mac
scp .env pi@raspberrypi:~/tech_analyze/.env
scp optimization_results_panel.json optimization_results_individual.json pi@raspberrypi:~/tech_analyze/
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

### Workflow between Mac and Pi

| Where | When | Action |
|---|---|---|
| Mac | After a re-save | `--backfill-panel --validate`, then `--optimize --optimize-combo` |
| Mac | After optimize | `scp optimization_results_panel.json optimization_results_individual.json pi@raspberrypi:~/tech_analyze/` |
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
SYSTEM STATUS  (weights fitted 2026-08-12, panel challenger: ACCEPTED)
  Out-of-sample, 4 fiscal years (2022-2025), total return (price + dividends):
    IC by year:   2022 -0.050   2023 +0.081   2024 +0.119   2025 +0.136
    mean IC +0.072  |  top-bottom spread +5.5%/yr
    beat equal weight in 4 of 4 held-out years
    permutation test (200 refits on shuffled targets): p=0.015
  CONFIDENCE: LOW — 4 periods only (t=+1.70, p=0.19). Directional, not proof.
  Universe: 127 stocks, survivorship-biased (today's list applied backwards).
```

Read `ACCEPTED` vs `REJECTED` first: on a reject the system silently runs equal weight, and the reported numbers then describe equal weight rather than the challenger that was turned down. The `CONFIDENCE: LOW` line is permanent, not a placeholder — four fiscal years cannot support a significance claim, and a tidy top-10 list otherwise implies more than the data does.

The block is read from the `validation` section of `optimization_results_panel.json`, so the one file you already copy to the Pi carries it. A Pi holding an older file, or none, prints a short "unknown" notice and still sends a complete email.

---

## Weight variants

```bash
uv run python3 main.py --no-opt          # Use hardcoded default weights (ignore optimizer output)
uv run python3 main.py --use-individual  # Force the individual-correlation weights
uv run python3 main.py --use-combo       # Force the grid-sweep + cross-validation weights
uv run python3 main.py --use-panel       # Force the panel challenger's out-of-sample-validated weights
```

Default (no flag): prefers `optimization_results_panel.json` when it exists — the out-of-sample, Deflated-Sharpe-Ratio-gated result (run `--optimize` once with `data/panel_scores.csv` present, see Workflow above) — falling back to `optimization_results_individual.json`, then hardcoded `metrics.py` defaults.

---

## Full CLI reference

| Flag | Description |
|---|---|
| `--watchlists NAME ...` | Personal Avanza watchlists to analyze |
| `--preset NAME ...` | Built-in presets: `omxs30`, `omxs-mid` |
| `--save` | Save today's metric snapshot to `data/` |
| `--correlate` | Run baseline correlation report against historical snapshots |
| `--optimize` | Re-optimize metric weights from historical data |
| `--optimize-combo` | Optimize with grid sweep + cross-validation |
| `--backfill-panel` | Build the fiscal-year panel from `data/*.csv` (writes `data/panel_fundamentals.csv` + `data/panel_scores.csv`) |
| `--validate` | Run the cross-sectional validation battery against `data/panel_scores.csv` |
| `--challenger-confidence X` | Deflated Sharpe Ratio bar the panel challenger must clear to accept optimized weights (default 0.925) |
| `--no-opt` | Ignore saved weights, use hardcoded defaults |
| `--use-individual` | Force individual-correlation weights |
| `--use-combo` | Force combo-optimized weights |
| `--use-panel` | Force the panel challenger's out-of-sample-validated weights |
| `--push` | Push top-scoring stocks to an Avanza watchlist |
| `--push-to NAME` | Which watchlist to push to (default: `Bör köpa`) |
| `--push-top N` | How many stocks to push (default: 25 — breadth, see below) |
| `--sell-from NAME` | Check this watchlist for sell signals |
| `--email` | Send email summary (requires `EMAIL_*` in `.env`) |

If neither `--watchlists` nor `--preset` is given, both default to the same universe `cron_pi.sh` uses: `--preset omxs30 omxs-mid --watchlists Test Utdelare Äger Berkshire`. Giving either flag explicitly uses exactly what you passed, with no default mixed in.
