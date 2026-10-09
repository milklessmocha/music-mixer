#!/usr/bin/env bash
# Pull main and rebuild the bot when GitHub has new commits, and once a day so yt-dlp stays current;
# a no-op otherwise. Run every 5 minutes by music-mixer-deploy.timer. --force rebuilds right away.
set -euo pipefail
cd "$(dirname "$0")/.."

git fetch --quiet origin main
today="$(date +%F)"
if [ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] && [ "$(cat .last-build 2>/dev/null)" = "$today" ] \
        && [ "${1:-}" != "--force" ]; then
    exit 0
fi

# --ff-only refuses to deploy if this checkout was edited by hand, instead of merging over it
git merge --ff-only --quiet origin/main
mkdir -p cookies  # created by us, not by docker as root, so cookies/youtube.txt can be added without sudo
YTDLP_REFRESH="$today" docker compose up -d --build --remove-orphans
echo "$today" > .last-build
docker image prune -f >/dev/null  # old images would slowly fill the SD card
echo "deployed $(git log -1 --format='%h %s')"
