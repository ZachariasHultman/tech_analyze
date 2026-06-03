# Stock Screener

A fundamental stock screener that pulls data from the unofficial Avanza API and yfinance, scores companies on ~20 quality/value metrics, and checks whether those scores have historically predicted returns.

> **Disclaimer:** Uses an unofficial, unsupported Avanza API. No affiliation with Avanza. Not financial advice. I have no formal economics education — this is an experiment based on things I've read from smarter people.

---

## Setup

Requires a `.env` file with your Avanza credentials:

```
USERNAME=your_avanza_username
PASSWORD=your_avanza_password
MY_TOTP_SECRET=your_totp_secret
```

Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

After syncing, `uv run tech-analyze` is available as a shortcut for `uv run python3 main.py`.

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

# Push top 10 scoring + reliable stocks to "Bör köpa" watchlist
uv run python3 main.py --watchlist

# Top 5 instead
uv run python3 main.py --watchlist --watchlist-top 5
```

---

## Expanding the universe

The more stocks you include, the more reliable the correlation analysis becomes. Available presets:

| Preset | Contents | ~Size |
|---|---|---|
| `omxs30` | OMXS30 large caps | 30 |
| `omxs-mid` | OMXS Mid Cap selection | 26 |

```bash
# Broadest built-in universe
uv run python3 main.py --preset omxs30 omxs-mid
```

Presets use `search_for_stock` to look up each company by name, so they don't rely on hardcoded IDs that can go stale.

> **Note on S&P 500 / US stocks:** The `get_analysis()` endpoint only returns meaningful financial data for Nordic stocks. US stocks silently produce empty metrics. Stick to Nordic lists.

---

## Building a solid backtest dataset — step by step

The optimizer needs historical snapshots of each company's metrics *and* its subsequent price returns. Here's how to build that up properly:

### Step 1 — Choose a broad universe (do this once)

Pick enough stocks that the cross-sectional correlation is statistically meaningful. Aim for **30+ companies**. More is better.

```bash
# Run with a broad set and save today's snapshot
uv run python3 main.py --save --preset omxs30 omxs-mid
```

Each run with `--save` writes one CSV per company to `data/`. These accumulate over time.

### Step 2 — Keep saving snapshots over time

Run `--save` regularly — monthly or quarterly is enough. The optimizer needs snapshots from different points in time to compute forward returns (what happened to the price *after* the snapshot was taken).

```bash
# Add to a cron job or just run manually each month
uv run python3 main.py --save --preset omxs30 omxs-mid
```

You need **at least 1–2 years of snapshots** before the correlation analysis is meaningful. The more historical depth, the more reliable the weights.

### Step 3 — Run the correlation analysis

Once you have data from multiple time points, check which metrics actually predicted returns:

```bash
uv run python3 main.py --correlate
```

This prints a baseline report: Spearman correlation between score and return for each time window, top/bottom quintile spread, and per-metric correlations. Look for:
- Average Spearman > 0.2 → the scoring system has predictive value
- A clear spread between the top and bottom quintile return
- Warning if fewer than 30 companies — correlation results are noisy below that

### Step 4 — Optimize metric weights (do this occasionally, not every run)

```bash
uv run python3 main.py --correlate --optimize
```

This re-weights each metric proportional to how well it predicted returns in your historical data. The result is saved to `optimization_results_individual.json` and picked up automatically on future runs.

**Re-run this when:**
- You've expanded the universe significantly (more stocks)
- 6+ months have passed since the last run
- The correlation report shows the current weights are underperforming

**Do not re-run this every week** — the correlation doesn't shift that fast, and with a small universe you risk overfitting to noise.

### Step 5 — Normal use going forward

```bash
# Live scoring uses the optimized weights automatically
uv run python3 main.py --preset omxs30 omxs-mid
```

---

## Weight variants

```bash
python3 main.py --no-opt          # Use hardcoded default weights (ignore optimizer output)
python3 main.py --use-individual  # Use individual-correlation weights (default when file exists)
python3 main.py --use-combo       # Use grid-sweep + cross-validation weights
python3 main.py --use-stepwise    # Use scipy Nelder-Mead weights
```

`--use-individual` is the most trustworthy with a small universe. `--use-combo` and `--use-stepwise` are more likely to overfit until you have 50+ companies × 2+ years of data.

---

## Full CLI reference

| Flag | Description |
|---|---|
| `--watchlists NAME ...` | Personal Avanza watchlists to analyze (default: `Test`) |
| `--preset NAME ...` | Built-in presets: `omxs30`, `omxs-mid` |
| `--save` | Save today's metric snapshot to `data/` |
| `--correlate` | Run baseline correlation report against historical snapshots |
| `--optimize` | Re-optimize metric weights from historical data |
| `--optimize-combo` | Optimize with grid sweep + cross-validation |
| `--optimize-stepwise` | Optimize with scipy Nelder-Mead |
| `--no-opt` | Ignore saved weights, use hardcoded defaults |
| `--use-individual` | Use individual-correlation weights |
| `--use-combo` | Use combo-optimized weights |
| `--use-stepwise` | Use stepwise-optimized weights |
| `--watchlist` | Push top-scoring reliable stocks to "Bör köpa" |
| `--watchlist-top N` | How many stocks to push (default: 10) |
