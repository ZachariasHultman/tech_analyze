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
| `optimization_results_individual.json` (or `_combo.json`) | Optional | Optimized weights/thresholds loaded at runtime; without it, `main.py` falls back to `metrics.py`'s hardcoded defaults |
| `company_reliability.csv` | Optional | Per-company reliability scores; without it, the watchlist push/sell-signal logic treats every company as reliability=unknown |

None of the four are ever committed (`*.json`/`*.csv` are blanket-ignored, and `metrics.py` is explicitly excluded).

---

## How it scores stocks

Each stock is scored on metrics like Piotroski F-Score, earnings quality (OCF/net income), revenue CAGR, ROE/PE ratio, net debt/EBITDA, price momentum, and dividend yield.

Scoring is **cross-sectional** — each metric is ranked relative to all other stocks in the current run (percentile 0–1), so a utility with solid-for-utilities ROE ranks in the top half just like a tech company with high ROE. No sector-specific thresholds needed.

A **reliability score** tracks whether each company's fundamental score has historically predicted its own price returns. Unreliable companies (good numbers, bad price response) are filtered out of the watchlist.

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

# Push top 10 scoring + reliable stocks to "Bör köpa" (default)
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

**Step 2 — Optimize metric weights and compute reliability**

```bash
uv run python3 main.py --correlate --optimize
```

This calculates which metrics historically predicted returns, saves optimized weights to `optimization_results_individual.json`, and saves per-company reliability scores to `company_reliability.csv`. Both files are picked up automatically on every future run.

> **Note:** the backtest previously had a look-ahead bug (predictors were measured across the same window as the return they were correlated against). This is now fixed, but it means any `optimization_results_*.json` / `company_reliability.csv` saved before the fix are invalid — re-run Step 2 (`--correlate --optimize`) at least once after upgrading.

---

### Daily use

Optimized weights and reliability are loaded automatically — no extra flags needed.

```bash
# Score a watchlist and push top 10 reliable stocks to "Bör köpa"
uv run python3 main.py --watchlists Äger --push

# Just score, no push
uv run python3 main.py --watchlists Äger
```

---

### Maintenance

| When | Command |
|---|---|
| **After new quarterly earnings** | `uv run python3 main.py --save --preset omxs30 omxs-mid ...` |
| **After re-saving** | `uv run python3 main.py --correlate --optimize` |
| **If you only changed a threshold in code** | `uv run python3 main.py --correlate` (skips optimize, just refreshes reliability) |
| **If you add new stocks to universe** | Re-save then re-optimize |

Re-saving quarterly is enough. Re-optimizing is only needed after a re-save or when you've significantly expanded the universe.

---

## Raspberry Pi deployment

The Pi only runs the monthly scoring job — no `--save`, no `--optimize`. Those stay on your Mac. The optimized weights (`optimization_results_individual.json`) and reliability scores (`company_reliability.csv`) are gitignored (`*.json`/`*.csv` are blanket-ignored) — they're never committed, so they must be copied to the Pi manually via `scp` (see below) after each re-optimize.

### First-time Pi setup

```bash
# On Pi
git clone <your-repo-url> ~/tech_analyze
cd ~/tech_analyze
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Copy credentials and output files from Mac
scp .env pi@raspberrypi:~/tech_analyze/.env
scp optimization_results_individual.json company_reliability.csv pi@raspberrypi:~/tech_analyze/
```

### Set up the monthly cron job

```bash
# On Pi
crontab -e
```

Add this line (runs at 08:00 on the 1st of each month):
```
0 8 1 * * /home/pi/tech_analyze/cron_pi.sh >> /home/pi/tech_analyze/cron.log 2>&1
```

### Workflow between Mac and Pi

| Where | When | Action |
|---|---|---|
| Mac | Quarterly | `--save --preset omxs30 ... --correlate --optimize` |
| Mac | After optimize | `scp optimization_results_individual.json company_reliability.csv pi@raspberrypi:~/tech_analyze/` |
| Pi | Automatic (cron) | `cron_pi.sh` — pulls code via git, scores, pushes to Avanza, sends email |

### Manual Pi run (test it first)

```bash
ssh pi@raspberrypi
cd ~/tech_analyze
./cron_pi.sh
```

---

## Sell signals

Add `--sell-from Äger` to flag stocks in your portfolio that have deteriorating fundamentals. A stock is flagged when:
- Its fundamental score is negative, or
- Its score doesn't reliably predict its returns (unreliable company), or
- Its combined score (pts × reliability) falls below 1.5

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

---

## Weight variants

```bash
uv run python3 main.py --no-opt          # Use hardcoded default weights (ignore optimizer output)
uv run python3 main.py --use-individual  # Use individual-correlation weights (default when file exists)
uv run python3 main.py --use-combo       # Use grid-sweep + cross-validation weights
```

`--use-individual` is the most trustworthy with a small universe. `--use-combo` is more likely to overfit until you have 50+ companies × 2+ years of data.

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
| `--no-opt` | Ignore saved weights, use hardcoded defaults |
| `--use-individual` | Use individual-correlation weights |
| `--use-combo` | Use combo-optimized weights |
| `--push` | Push top-scoring reliable stocks to an Avanza watchlist |
| `--push-to NAME` | Which watchlist to push to (default: `Bör köpa`) |
| `--push-top N` | How many stocks to push (default: 10) |
| `--sell-from NAME` | Check this watchlist for sell signals |
| `--email` | Send email summary (requires `EMAIL_*` in `.env`) |

If neither `--watchlists` nor `--preset` is given, both default to the same universe `cron_pi.sh` uses: `--preset omxs30 omxs-mid --watchlists Test Utdelare Äger Berkshire`. Giving either flag explicitly uses exactly what you passed, with no default mixed in.
