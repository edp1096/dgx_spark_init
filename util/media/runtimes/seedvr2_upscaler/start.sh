#!/usr/bin/env bash
set -euo pipefail

python /opt/ComfyUI/main.py \
  --listen 127.0.0.1 \
  --port 8189 \
  --disable-auto-launch \
  --preview-method none &
comfy_pid=$!

cleanup() {
  kill "${comfy_pid}" 2>/dev/null || true
  wait "${comfy_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

exec uvicorn api:app --host 0.0.0.0 --port "${API_PORT:-8698}"
