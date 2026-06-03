#!/bin/bash
# Monthly stock screener run for Raspberry Pi.
# Add to crontab with: crontab -e
#   0 8 1 * * /home/pi/tech_analyze/cron_pi.sh >> /home/pi/tech_analyze/cron.log 2>&1

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

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
