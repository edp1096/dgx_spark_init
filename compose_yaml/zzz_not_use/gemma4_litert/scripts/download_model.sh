#!/usr/bin/env bash

source "$(dirname "$0")/common.sh"

mkdir -p "${SOURCE_MODEL_DIR}"
"${VENV_DIR}/bin/hf" download \
  huihui-ai/Huihui-gemma-4-E2B-it-abliterated \
  --revision "${MODEL_REF}" \
  --local-dir "${SOURCE_MODEL_DIR}"
