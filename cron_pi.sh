#!/bin/bash
# Weekly stock screener run for Raspberry Pi.
# Add to crontab with: crontab -e
#   0 8 * * 1 /home/zacharias/tech_analyze/cron_pi.sh >> /home/zacharias/tech_analyze/logs/cron.log 2>&1

set -e

# Cron runs with a minimal PATH; uv lives in ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"

# Load project credentials from .env (keeps this isolated from other cron jobs)
# shellcheck source=/dev/null
[ -f "$(dirname "$0")/.env" ] && set -a && source "$(dirname "$0")/.env" && set +a

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

LOG_FILE="$REPO_DIR/logs/cron.log"

echo "=== $(date) ==="

# Pull latest code from git (weights/reliability are SCP'd separately from Mac)
git pull --ff-only

# Score, push to Avanza, check sells, send email
uv run python3 main.py \
  --preset omxs30 omxs-mid \
  --watchlists Test Utdelare Äger Berkshire \
  --push \
  --sell-from Äger \
  --email

echo "=== Done ==="

# Trim log after all output is flushed (~3 runs of ~200 lines each)
[ -f "$LOG_FILE" ] && tail -n 600 "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
