#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="$script_dir/.env"

if [[ -f "$env_file" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$env_file"
  set +a
if [[ ${RUNTIME_HF_TOKEN+x} ]]; then export HF_TOKEN="$RUNTIME_HF_TOKEN"; fi
fi

model_dir="${MODEL_HOST_PATH:-/home/edp1096/.cache/huggingface/exl3-qwen38-27b-uncensored-4bpw}"
cache_dir="${EXL3_CACHE_PATH:-/home/edp1096/.cache/exl3-qwen38-27b}"

usage() {
  echo "usage: $0 download | prepare | status" >&2
  exit 2
}

safe_path() {
  [[ "$1" == /* && "$1" =~ ^/[A-Za-z0-9._/-]+$ ]] || {
    echo "unsafe or non-absolute path: $1" >&2
    exit 2
  }
}

model_ok() {
  local count bytes
  [[ -f "$model_dir/config.json" && -f "$model_dir/quantization_config.json" ]] || return 1
  count=$(find "$model_dir" -maxdepth 1 -type f -name '*.safetensors' | wc -l)
  bytes=$(find "$model_dir" -maxdepth 1 -type f -name '*.safetensors' -printf '%s\n' | awk '{s += $1} END {print s + 0}')
  (( count >= 3 && bytes >= 14000000000 ))
}

safe_path "$model_dir"
safe_path "$cache_dir"

download() {
  mkdir -p "$model_dir" "$cache_dir/torch_extensions" "$cache_dir/huggingface"
  docker compose \
    --project-directory "$script_dir" \
    --profile download \
    -f "$script_dir/compose.yaml" \
    run --rm download
  model_ok || {
    echo "Model download incomplete: expected config files and three safetensor shards (>=14 GB)." >&2
    exit 1
  }
  echo "Model ready: $model_dir"
}

status() {
  if model_ok; then
    echo "model: complete ($model_dir)"
  else
    echo "model: missing/incomplete ($model_dir)"
    return 1
  fi
}

case "${1:-}" in
  download|prepare) download ;;
  status) status ;;
  *) usage ;;
esac
