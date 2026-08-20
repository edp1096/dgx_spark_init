#!/usr/bin/env bash
set -euo pipefail

image="${VLLM_OMNI_IMAGE:-dgx-vllm-omni:v0.26.0}"
cache_volume="${HF_CACHE_VOLUME:-media-hf-cache}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

docker volume inspect "${cache_volume}" >/dev/null
docker run --rm \
  --entrypoint python \
  -e HF_TOKEN \
  -e OFFICIAL_MODEL_REPO \
  -e TEXT_ENCODER_REPO \
  -e ASSEMBLED_MODEL_DIR \
  -v "${cache_volume}:/root/.cache/huggingface" \
  -v "${script_dir}/prepare_model.py:/opt/prepare_model.py:ro" \
  "${image}" /opt/prepare_model.py
