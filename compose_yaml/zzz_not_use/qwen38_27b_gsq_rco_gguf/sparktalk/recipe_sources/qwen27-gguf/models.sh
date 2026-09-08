#!/usr/bin/env bash
set -euo pipefail
if [[ "${MODEL_VARIANT:-abliterated}" != abliterated ]]; then echo 'Only the verified Huihui GGUF checkpoint is configured.' >&2; exit 2; fi
model="$GGUF_DATA_DIR/models/Huihui-Qwen3.8-27B-ISTA-IQ3_S-Allocation-MTP.gguf"
projector="$GGUF_DATA_DIR/models/mmproj-Huihui-Qwen3.8-27B-BF16.gguf"
if [[ -f "$model" && -f "$projector" ]]; then
  printf '%s  %s\n' '235610e2f238b270b908c4a93311f24dc8d1f3decfa2ad2fd58fd95958d05ca1' "$model" '3efdf9dc94b1ebb71f34a43226c0b1f27c31b3144a4db1f2659fce2e07e57534' "$projector" | sha256sum --check
  exit 0
fi
if ! docker image inspect qwen38-27b-gsq-rco-gguf-tools:e71b805 >/dev/null 2>&1; then docker compose build tools; fi
docker compose run --rm --no-deps tools python3 pipeline.py quantize
