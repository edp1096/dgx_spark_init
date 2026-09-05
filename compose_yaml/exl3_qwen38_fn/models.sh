#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="$script_dir/.env"

if [[ -f "$env_file" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$env_file"
  set +a
fi

model_dir="${MODEL_HOST_PATH:-/home/edp1096/.cache/huggingface/exl3-qwen38-fn-4.05bpw}"
cache_dir="${EXL3_CACHE_PATH:-/home/edp1096/.cache/exl3-qwen38-fn}"
ablit_dir="${ABLIT_OUTPUT_PATH:-$cache_dir/ablit}"
compose=(docker compose --project-directory "$script_dir" --profile prepare -f "$script_dir/compose.yaml")

usage() {
  echo "usage: $0 download | direction | prepare | status" >&2
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
  [[ -f "$model_dir/config.json" && -f "$model_dir/quantization_config.json" \
     && -f "$model_dir/ngram_embedding.safetensors" ]] || return 1
  count=$(find "$model_dir" -maxdepth 1 -type f -name '*.safetensors' | wc -l)
  bytes=$(find "$model_dir" -maxdepth 1 -type f -name '*.safetensors' -printf '%s\n' | awk '{s += $1} END {print s + 0}')
  (( count >= 11 && bytes >= 100000000000 ))
}

direction_ok() {
  [[ -s "$ablit_dir/direction.safetensors" && -s "$ablit_dir/direction.json" ]]
}

safe_path "$model_dir"
safe_path "$cache_dir"
safe_path "$ablit_dir"

download() {
  mkdir -p "$model_dir" "$cache_dir/torch_extensions" "$cache_dir/huggingface"
  "${compose[@]}" run --rm download
  model_ok || {
    echo "Model download incomplete: expected Flash-Next EXL3 shards (>=100 GB)." >&2
    exit 1
  }
  echo "Model ready: $model_dir"
}

direction() {
  mkdir -p "$ablit_dir"
  "${compose[@]}" run --rm direction
  direction_ok || {
    echo "Abliteration direction preparation failed." >&2
    exit 1
  }
}

status() {
  local rc=0
  if model_ok; then echo "model: complete ($model_dir)"; else echo "model: missing/incomplete ($model_dir)"; rc=1; fi
  if direction_ok; then echo "direction: complete ($ablit_dir)"; else echo "direction: missing ($ablit_dir)"; rc=1; fi
  return "$rc"
}

case "${1:-}" in
  download) download ;;
  direction) direction ;;
  prepare) direction; download ;;
  status) status ;;
  *) usage ;;
esac
