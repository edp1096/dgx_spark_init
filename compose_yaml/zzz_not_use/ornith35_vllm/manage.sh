#!/usr/bin/env bash
set -euo pipefail
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-status}" in
  setup) docker compose -f "$recipe_dir/compose.yaml" build ;;
  start) docker compose -f "$recipe_dir/compose.yaml" up -d --no-build --pull never ;;
  stop) docker compose -f "$recipe_dir/compose.yaml" down ;;
  status) docker compose -f "$recipe_dir/compose.yaml" ps ;;
  logs) docker compose -f "$recipe_dir/compose.yaml" logs -f --tail 100 ;;
  *) echo 'Usage: manage.sh setup|start|stop|status|logs' >&2; exit 2 ;;
esac
