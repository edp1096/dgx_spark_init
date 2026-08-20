#!/usr/bin/env bash

source "$(dirname "$0")/common.sh"

export VK_DRIVER_FILES="${VK_DRIVER_FILES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
if [[ ! -x "${RUNTIME_VENV_DIR}/bin/litert-lm" ]]; then
  echo "LiteRT-LM runtime is missing. Run make runtime first." >&2
  exit 1
fi

exec "${RUNTIME_VENV_DIR}/bin/litert-lm" \
  --config "${PROJECT_DIR}/config.json" \
  serve \
  --host "${LITERT_HOST:-0.0.0.0}" \
  --port "${LITERT_PORT:-8696}"
