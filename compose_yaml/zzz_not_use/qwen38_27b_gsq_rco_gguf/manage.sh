#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
mkdir -p data
case "${1:-help}" in
  image)
    docker compose build tools
    docker compose build server
    ;;
  tools-image) docker compose build tools ;;
  import-source)
    test -n "${2:-}" || { echo 'Usage: ./manage.sh import-source /absolute/BF16/snapshot'; exit 1; }
    source_path=$(realpath "$2")
    if [[ "$(basename "$(dirname "$source_path")")" == snapshots ]]; then
      source_repo=$(dirname "$(dirname "$source_path")")
      docker compose run --rm --no-deps -v "$source_repo:/import-repo:ro" tools python3 pipeline.py import-source "/import-repo/snapshots/$(basename "$source_path")"
    else
      docker compose run --rm --no-deps -v "$source_path:/import:ro" tools python3 pipeline.py import-source /import
    fi
    ;;
  model|prepare) docker compose run --rm --no-deps tools python3 pipeline.py prepare ;;
  convert|quantize|verify) docker compose run --rm --no-deps tools python3 pipeline.py "$1" ;;
  setup)
    "$0" image
    "$0" quantize
    ;;
  start)
    available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    if (( available_kib < 24 * 1024 * 1024 )); then
      echo 'Need at least 24 GiB available before starting the 32K GGUF profile. Stop the other LLM first.' >&2
      exit 1
    fi
    docker compose up -d --no-build server
    ;;
  stop) docker compose stop server ;;
  validate)
    docker run --rm --network host --user "$LOCAL_UID:$LOCAL_GID" -e DATA_ROOT=/data \
      -v "$PWD:/workspace:ro" -v "$PWD/data:/data" -w /workspace \
      qwen38-27b-gsq-rco-gguf-tools:e71b805 python3 validate.py "${@:2}"
    ;;
  status) docker compose ps ;;
  logs) docker compose logs -f --tail 100 server ;;
  *) echo 'Usage: ./manage.sh {setup|image|tools-image|import-source PATH|model|convert|quantize|verify|start|stop|validate|status|logs}' ;;
esac
