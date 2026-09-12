#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
rank=${1:?rank required}
mode=${2:?shared|routed|upgrade-base|upgrade-new}
case "$rank" in 0|1) ;; *) exit 2;; esac
image=dgx-ds41-stream:b12x8
script=check_shared_prefill.py
slots=24
extra=()
args=(--reference /opt/ds41/results/pp-round4-reference.py)
case "$mode" in
  shared) ;;
  routed) script=check_routed_pipeline.py; slots=224; extra=(-e DSV41_ROUTED_PIPELINE=1);;
  upgrade-base|upgrade-new)
    script=check_b12x_upgrade.py; slots=224
    args=(--reference "/cache/round4-upgrade-reference-rank$rank.pt")
    if [[ "$mode" == upgrade-base ]]; then args+=(--write-reference); else image=dgx-ds41-stream:b12x-081b2359; fi;;
  *) exit 2;;
esac
docker run --rm --gpus all --name "ds41-architecture-test-$rank" \
  --network host --ipc host --memory 16g --memory-swap 16g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  -v "$PWD:/opt/ds41:ro" -v "$HOME/.cache/ds41-stream:/cache" \
  -v "$HOME/.cache/ds41-packed/rank$rank:/packed:ro" \
  -v "$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash:/repo:ro" \
  -e PYTHONPATH=/opt/ds41 -e XDG_CACHE_HOME=/cache -e DSV41_PACKED_DIR=/packed \
  -e DSV41_SLOT_IO=direct -e DSV41_EXPERT_IO=batch_overlap -e DSV41_SLOTS_PER_LAYER="$slots" \
  -e DSV41_SHARED_BUFFERS=1 -e DSV41_KERNEL_TOKENS=2048 -e DSV41_SCRATCH_MIB=384 \
  -e DSV41_RELEASE_CHECKPOINT_CACHE=0 -e B12X_COMPILE_CACHE_DIR=/cache/b12x \
  -e CUTE_DSL_CACHE_DIR=/cache/cute -e OMP_NUM_THREADS=1 -e VLLM_PLUGINS= "${extra[@]}" \
  --entrypoint timeout "$image" 600s python3 "/opt/ds41/tools/$script" \
  /repo/snapshots/dba1be0a40aa45a94ad051997016db3960a90277 \
  --rank "$rank" "${args[@]}" --output "/cache/round4-$mode-rank$rank.json"
