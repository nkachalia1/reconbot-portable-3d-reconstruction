#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/portable-recon-robot}"
RECONBOT_USER="${SUDO_USER:-$USER}"
SERVICE_SOURCE="$PROJECT_DIR/deploy/reconbot-coordinator.service"
ENV_SOURCE="$PROJECT_DIR/deploy/coordinator.env.example"
SERVICE_TEMP="$(mktemp)"
ENV_TEMP="$(mktemp)"
trap 'rm -f "$SERVICE_TEMP" "$ENV_TEMP"' EXIT

sudo install -d -m 0755 /etc/reconbot
if [[ ! -f /etc/reconbot/coordinator.env ]]; then
  sed "s|@PROJECT_DIR@|$PROJECT_DIR|g" "$ENV_SOURCE" > "$ENV_TEMP"
  sudo install -m 0600 "$ENV_TEMP" /etc/reconbot/coordinator.env
fi
sed \
  -e "s|@RECONBOT_USER@|$RECONBOT_USER|g" \
  -e "s|@PROJECT_DIR@|$PROJECT_DIR|g" \
  "$SERVICE_SOURCE" > "$SERVICE_TEMP"
sudo install -m 0644 "$SERVICE_TEMP" /etc/systemd/system/reconbot-coordinator.service
sudo systemctl daemon-reload
sudo systemctl enable reconbot-coordinator.service

echo "Edit /etc/reconbot/coordinator.env, then run:"
echo "  sudo systemctl restart reconbot-coordinator"
echo "  sudo systemctl status reconbot-coordinator"
