#!/usr/bin/env bash

source "$(dirname "$0")/common.sh"

if [[ ! -f "${SOURCE_MODEL_DIR}/model.safetensors" ]]; then
  echo "Source model is missing. Run make download first." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
export SOURCE_MODEL_DIR OUTPUT_DIR
"${VENV_DIR}/bin/python" - <<'PY'
import os
from litert_torch.generative.export_hf import export

export.export(
    model=os.environ["SOURCE_MODEL_DIR"],
    output_dir=os.environ["OUTPUT_DIR"],
    task="image_text_to_text",
    keep_temporary_files=True,
    prefill_lengths=[128, 256, 512],
    cache_length=4096,
    quantization_recipe="dynamic_wi8_afp32",
    vision_encoder_quantization_recipe="dynamic_wi8_afp32",
    bundle_litert_lm=True,
    gemma4_vision_max_soft_tokens=280,
)
PY
