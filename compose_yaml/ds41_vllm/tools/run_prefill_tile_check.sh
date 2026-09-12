#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
rank=${1:?rank required}
case "$rank" in 0|1) ;;*) exit 2;;esac
docker run --rm --gpus all --name "ds41-prefill-test-$rank" \
  --network host --ipc host --memory 12g --memory-swap 12g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  -v "$PWD:/opt/ds41:ro" -v "$HOME/.cache/ds41-stream:/cache" \
  -v "$HOME/.cache/ds41-packed/rank$rank:/packed:ro" \
  -v "$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash:/repo:ro" \
  -e PYTHONPATH=/opt/ds41 -e XDG_CACHE_HOME=/cache -e DSV41_PACKED_DIR=/packed \
  -e DSV41_SLOT_IO=direct -e DSV41_EXPERT_IO=batch -e DSV41_SLOTS_PER_LAYER=24 \
  -e DSV41_SCRATCH_MIB=256 -e DSV41_RELEASE_CHECKPOINT_CACHE=0 \
  -e B12X_COMPILE_CACHE_DIR=/cache/b12x -e CUTE_DSL_CACHE_DIR=/cache/cute \
  -e OMP_NUM_THREADS=1 -e VLLM_PLUGINS= \
  --entrypoint timeout dgx-ds41-stream:b12x8 300s \
  python3 /opt/ds41/tools/check_prefill_tiles.py /repo/snapshots/dba1be0a40aa45a94ad051997016db3960a90277 \
  --rank "$rank" --output "/cache/prefill-tiles-rank$rank.json"
