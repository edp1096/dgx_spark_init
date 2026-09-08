#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
case "${1:-help}" in
  setup) docker compose build ;;
  start) docker compose up -d --build ;;
  stop) docker compose down ;;
  status) docker compose ps ;;
  logs) docker compose logs -f --tail 100 ;;
  *) echo 'Usage: ./manage.sh {setup|start|stop|status|logs}' ;;
esac
