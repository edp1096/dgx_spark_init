#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
rank=${1:?rank required}
mode=${2:?base or upstream-default required}
case "$rank" in 0|1) ;; *) exit 2;; esac
image=dgx-ds41-stream:b12x8
script=check_b12x_upgrade.py
args=()
case "$mode" in
  base) args+=(--write-reference);;
  upstream-default)
    image=dgx-ds41-stream:compact-323107ff
    script=probe_b12x_default_recipe.py;;
  *) exit 2;;
esac
# Small isolated kernel test. Never overlap it with full-model measurements.
if docker inspect -f '{{.State.Running}}' "ds41-stream-$rank" 2>/dev/null | rg -q '^true$'; then
  echo 'Stop the full-model benchmark before kernel qualification' >&2
  exit 1
fi
docker run --rm --gpus all --name "ds41-compact-test-$rank" \
  --network none --ipc host --memory 16g --memory-swap 16g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  -v "$PWD:/opt/ds41:ro" -v "$HOME/.cache/ds41-stream:/cache" \
  -v "$HOME/.cache/ds41-packed/rank$rank:/packed:ro" \
  -v "$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash:/repo:ro" \
  -e PYTHONPATH=/opt/ds41 -e XDG_CACHE_HOME=/cache -e DSV41_PACKED_DIR=/packed \
  -e DSV41_SLOT_IO=direct -e DSV41_EXPERT_IO=batch_overlap -e DSV41_SLOTS_PER_LAYER=224 \
  -e DSV41_SHARED_BUFFERS=1 -e DSV41_KERNEL_TOKENS=2048 -e DSV41_SCRATCH_MIB=384 \
  -e DSV41_RELEASE_CHECKPOINT_CACHE=0 -e B12X_COMPILE_CACHE_DIR=/cache/b12x \
  -e CUTE_DSL_CACHE_DIR=/cache/cute -e OMP_NUM_THREADS=1 -e VLLM_PLUGINS= \
  --entrypoint timeout "$image" 600s python3 "/opt/ds41/tools/$script" \
  /repo/snapshots/dba1be0a40aa45a94ad051997016db3960a90277 \
  --rank "$rank" --tokens 1,5,6,8,16,512,2048 "${args[@]}" \
  --reference "/cache/compact-reference-rank$rank.pt" \
  --output "/cache/compact-$mode-rank$rank.json"
