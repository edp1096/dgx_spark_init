#!/usr/bin/env bash
set -euo pipefail

python /opt/minimax-h3/prepare_models.py

python /opt/ComfyUI/main.py \
  --listen 0.0.0.0 \
  --port "${COMFY_PORT:-8190}" \
  --disable-auto-launch \
  --preview-method none \
  --gpu-only &
comfy_pid=$!

trap 'kill "$comfy_pid" 2>/dev/null || true' EXIT INT TERM
exec python -m uvicorn api:app --host 0.0.0.0 --port "${H3_API_PORT:-8191}"
