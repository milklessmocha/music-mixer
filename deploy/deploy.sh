#!/usr/bin/env bash
# Pull main and rebuild the bot when GitHub has new commits; a no-op otherwise.
# Run every 5 minutes by music-mixer-deploy.timer. --force rebuilds even without new commits.
set -euo pipefail
cd "$(dirname "$0")/.."

git fetch --quiet origin main
if [ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] && [ "${1:-}" != "--force" ]; then
    exit 0
fi

# --ff-only refuses to deploy if this checkout was edited by hand, instead of merging over it
git merge --ff-only --quiet origin/main
docker compose up -d --build --remove-orphans
docker image prune -f >/dev/null  # old images would slowly fill the SD card
echo "deployed $(git log -1 --format='%h %s')"
