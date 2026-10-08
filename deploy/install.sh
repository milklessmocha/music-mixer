#!/usr/bin/env bash
# One command on the Pi, from inside the cloned repo, after creating .env:
#   ./deploy/install.sh
# Installs Docker if missing, starts the bot, and sets up a timer that pulls main and rebuilds within
# 5 minutes of every push. Safe to run again.
set -euo pipefail
dir="$(cd "$(dirname "$0")/.." && pwd)"
user="$(id -un)"

grep -qE '^BOT_TOKEN=.+' "$dir/.env" 2>/dev/null && grep -qE '^ALLOWED_USERS=[0-9]' "$dir/.env" || {
    echo "Create $dir/.env first (see .env.example) with BOT_TOKEN and ALLOWED_USERS filled in."
    exit 1
}

echo "== Docker"
command -v docker >/dev/null || curl -fsSL https://get.docker.com | sudo sh
docker compose version >/dev/null 2>&1 || sudo apt-get install -y -qq docker-compose-plugin
id -nG "$user" | grep -qw docker || sudo usermod -aG docker "$user"

echo "== auto-update timer"
for unit in music-mixer-deploy.service music-mixer-deploy.timer; do
    sed -e "s|@REPO@|$dir|g" -e "s|@USER@|$user|g" "$dir/deploy/$unit" | sudo tee "/etc/systemd/system/$unit" >/dev/null
done
sudo systemctl daemon-reload
sudo systemctl enable --now music-mixer-deploy.timer

echo "== starting the bot (the first build takes a few minutes)"
# a docker group added just now only applies to new logins, so go through sg
sg docker -c "$dir/deploy/deploy.sh --force"
echo
echo "Done. The bot is running in Docker and updates itself within 5 minutes of every push to main."
echo "  bot logs:     docker compose logs -f"
echo "  update log:   journalctl -u music-mixer-deploy"
