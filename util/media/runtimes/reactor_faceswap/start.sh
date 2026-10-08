#!/usr/bin/env bash
set -euo pipefail

python /opt/reactor-api/prepare_models.py

python /opt/ComfyUI/main.py \
  --listen 127.0.0.1 \
  --port 8190 \
  --disable-auto-launch \
  --preview-method none &
comfy_pid=$!

uvicorn api:app --host 0.0.0.0 --port "${API_PORT:-8706}" &
api_pid=$!

cleanup() {
  kill "${comfy_pid}" 2>/dev/null || true
  kill "${api_pid}" 2>/dev/null || true
  wait "${comfy_pid}" 2>/dev/null || true
  wait "${api_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait -n "${comfy_pid}" "${api_pid}"
