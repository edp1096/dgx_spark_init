#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$script_dir/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$script_dir/.env"
  set +a
if [[ ${RUNTIME_HF_TOKEN+x} ]]; then export HF_TOKEN="$RUNTIME_HF_TOKEN"; fi
fi

model_dir="${MODEL_HOST_PATH:-/home/edp1096/.cache/huggingface/glm53-exl3}"
draft_dir="${DFLASH_HOST_PATH:-/home/edp1096/.cache/huggingface/glm53-dflash2-mxfp8}"
cache_dir="${GLM53_CACHE_PATH:-/home/edp1096/.cache/glm53-vllm}"
ablit_dir="${ABLIT_HOST_PATH:-/home/edp1096/.cache/huggingface/glm53-lovesenko-oproj}"
worker_ip="${WORKER_LAN_IP:-}"
worker_user="${WORKER_USER:-$(id -un)}"
sync_host="${MODEL_SYNC_HOST:-$worker_ip}"
worker_target="${worker_user}@${sync_host}"
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10)

usage() {
  echo "usage: $0 download | sync | prepare | status" >&2
  exit 2
}

safe_path() {
  [[ "$1" == /* && "$1" =~ ^/[A-Za-z0-9._/-]+$ ]] || {
    echo "unsafe or non-absolute model path: $1" >&2
    exit 2
  }
}
safe_path "$model_dir"
safe_path "$draft_dir"
safe_path "$cache_dir"
safe_path "$ablit_dir"

ablit_enabled() {
  case "${ABLIT:-0}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

model_ok() {
  local dir=$1 count
  [[ -f "$dir/config.json" ]] || return 1
  count=$(find "$dir" -maxdepth 1 -name '*.safetensors' -type f | wc -l)
  (( count >= 120 ))
}

draft_ok() {
  local dir=$1 size
  [[ -f "$dir/config.json" && -f "$dir/model.safetensors" ]] || return 1
  size=$(stat -c %s "$dir/model.safetensors" 2>/dev/null || echo 0)
  (( size >= 1000000000 ))
}

ablit_ok() {
  python3 "$script_dir/ablit/fetch_transplant.py" "$1" --check
}

download() {
  mkdir -p "$model_dir" "$draft_dir" "$ablit_dir" \
    "$cache_dir/jit/triton" "$cache_dir/jit/flashinfer" \
    "$cache_dir/jit/b12x" "$cache_dir/jit/vllm"

  docker compose \
    --project-directory "$script_dir" \
    --profile download \
    -f "$script_dir/compose.yaml" \
    run --rm download

  model_ok "$model_dir" || {
    echo "EXL3 download incomplete: expected config.json and at least 120 safetensor shards" >&2
    exit 1
  }
  draft_ok "$draft_dir" || {
    echo "DFlash2 download incomplete: expected config.json and a >=1 GB model.safetensors" >&2
    exit 1
  }
  if ablit_enabled; then
    python3 "$script_dir/ablit/fetch_transplant.py" "$ablit_dir"
    ablit_ok "$ablit_dir" || {
      echo "o_proj donor download incomplete: expected Lovesenko manifest and 45 verified tensors" >&2
      exit 1
    }
  fi
  echo "Local checkpoints are complete."
}

require_worker() {
  [[ -n "$worker_ip" && -n "$sync_host" ]] || {
    echo "WORKER_LAN_IP is not set in .env" >&2
    exit 2
  }
  ssh "${ssh_opts[@]}" "$worker_target" true || {
    echo "SSH failed for $worker_target." >&2
    echo "Verify the host key and install this head's public key first:" >&2
    echo "  ssh ${worker_user}@${worker_ip}" >&2
    echo "  ssh-copy-id ${worker_user}@${worker_ip}" >&2
    exit 1
  }
}

sync_models() {
  model_ok "$model_dir" || {
    echo "Local EXL3 checkpoint is incomplete; run '$0 download' first." >&2
    exit 1
  }
  draft_ok "$draft_dir" || {
    echo "Local DFlash2 checkpoint is incomplete; run '$0 download' first." >&2
    exit 1
  }
  if ablit_enabled && ! ablit_ok "$ablit_dir"; then
    echo "Local o_proj donor is incomplete; run '$0 download' first." >&2
    exit 1
  fi
  require_worker

  ssh "${ssh_opts[@]}" "$worker_target" \
    "mkdir -p '$model_dir' '$draft_dir' '$cache_dir/jit/triton' '$cache_dir/jit/flashinfer' '$cache_dir/jit/b12x' '$cache_dir/jit/vllm'"
  if ablit_enabled; then
    ssh "${ssh_opts[@]}" "$worker_target" "mkdir -p '$ablit_dir'"
  fi

  # No --delete: interrupted transfers resume and unrelated worker files remain.
  rsync -a --partial --human-readable --info=progress2 \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=10" \
    "$model_dir/" "$worker_target:$model_dir/"
  rsync -a --partial --human-readable --info=progress2 \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=10" \
    "$draft_dir/" "$worker_target:$draft_dir/"
  if ablit_enabled; then
    rsync -a --partial --human-readable --info=progress2 \
      -e "ssh -o BatchMode=yes -o ConnectTimeout=10" \
      "$ablit_dir/" "$worker_target:$ablit_dir/"
  fi

  ssh "${ssh_opts[@]}" "$worker_target" \
    "test -f '$model_dir/config.json' && [ \$(find '$model_dir' -maxdepth 1 -name '*.safetensors' -type f | wc -l) -ge 120 ] && test -f '$draft_dir/config.json' && test \$(stat -c %s '$draft_dir/model.safetensors') -ge 1000000000"
  if ablit_enabled; then
    ssh "${ssh_opts[@]}" "$worker_target" \
      "python3 - '$ablit_dir' --check" < "$script_dir/ablit/fetch_transplant.py"
  fi
  echo "Worker checkpoints are complete at $sync_host."
}

status() {
  if model_ok "$model_dir"; then echo "local EXL3: complete"; else echo "local EXL3: missing/incomplete"; fi
  if draft_ok "$draft_dir"; then echo "local DFlash2: complete"; else echo "local DFlash2: missing/incomplete"; fi
  if ablit_enabled; then
    if ablit_ok "$ablit_dir"; then echo "local o_proj donor: complete"; else echo "local o_proj donor: missing/incomplete"; fi
  else
    echo "o_proj transplant: disabled"
  fi
  [[ -n "$worker_ip" ]] || { echo "worker: WORKER_LAN_IP is not set"; return; }
  if ssh "${ssh_opts[@]}" "$worker_target" \
    "test -f '$model_dir/config.json' && [ \$(find '$model_dir' -maxdepth 1 -name '*.safetensors' -type f | wc -l) -ge 120 ]"; then
    echo "worker EXL3: complete"
  else
    echo "worker EXL3: unreachable or incomplete"
  fi
  if ssh "${ssh_opts[@]}" "$worker_target" \
    "test -f '$draft_dir/config.json' && test \$(stat -c %s '$draft_dir/model.safetensors' 2>/dev/null || echo 0) -ge 1000000000"; then
    echo "worker DFlash2: complete"
  else
    echo "worker DFlash2: unreachable or incomplete"
  fi
  if ablit_enabled; then
    if ssh "${ssh_opts[@]}" "$worker_target" \
      "python3 - '$ablit_dir' --check" < "$script_dir/ablit/fetch_transplant.py"; then
      echo "worker o_proj donor: complete"
    else
      echo "worker o_proj donor: unreachable or incomplete"
    fi
  fi
}

case "${1:-}" in
  download) download ;;
  sync) sync_models ;;
  prepare) download; sync_models ;;
  status) status ;;
  *) usage ;;
esac
