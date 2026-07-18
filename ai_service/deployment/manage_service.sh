#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="nutrisnap-ai.service"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

usage() {
  echo "Usage: $0 {start|stop|restart|status|logs|health}"
}

systemctl_cmd() {
  if systemctl list-unit-files "$SERVICE_NAME" >/dev/null 2>&1; then
    systemctl "$@"
  elif systemctl --user list-unit-files "$SERVICE_NAME" >/dev/null 2>&1; then
    systemctl --user "$@"
  else
    echo "Systemd service $SERVICE_NAME is not installed." >&2
    exit 2
  fi
}

case "${1:-}" in
  start)
    systemctl_cmd start "$SERVICE_NAME"
    ;;
  stop)
    systemctl_cmd stop "$SERVICE_NAME"
    ;;
  restart)
    systemctl_cmd restart "$SERVICE_NAME"
    ;;
  status)
    systemctl_cmd status "$SERVICE_NAME" --no-pager
    ;;
  logs)
    if systemctl list-unit-files "$SERVICE_NAME" >/dev/null 2>&1; then
      journalctl -u "$SERVICE_NAME" -f
    else
      journalctl --user -u "$SERVICE_NAME" -f
    fi
    ;;
  health)
    curl --fail --silent --show-error "$BASE_URL/health"
    echo
    ;;
  *)
    usage
    exit 2
    ;;
esac
