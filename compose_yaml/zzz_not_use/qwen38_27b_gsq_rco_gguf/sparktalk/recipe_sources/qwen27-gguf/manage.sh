#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
set -a
[[ ! -f .env ]] || source .env
set +a
export GGUF_DATA_DIR="${GGUF_DATA_DIR:-${HF_CACHE:-${HOME}/.cache/huggingface}/qwen38_27b_gsq_rco_gguf}"
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
mkdir -p "$GGUF_DATA_DIR"
case "${1:-}" in
  model) bash models.sh ;;
  setup) docker compose build tools runtime; bash models.sh ;;
  *) echo 'Use SparkTalk to start/stop the runtime; preparation accepts model or setup.' >&2; exit 2 ;;
esac
