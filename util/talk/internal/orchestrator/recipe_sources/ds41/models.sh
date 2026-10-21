#!/usr/bin/env bash
set -euo pipefail
# Fast startup checks; full checkpoint hashes and AOT coverage were verified at preparation.
rank=${1:?rank required}
[[ $rank == 0 || $rank == 1 ]]
[[ ${MODEL_VARIANT:-official} == official || $MODEL_VARIANT == abliterated ]]
[[ $MAX_MODEL_LEN == 1048576 ]] || { echo 'TensorFold requires the qualified 1M profile and matching AOT assets' >&2; exit 2; }
[[ $(docker image inspect -f '{{.Id}}' "$DSV41_IMAGE") == "$DSV41_IMAGE_ID" ]] || { echo 'TensorFold image differs from the tested CUDA Zig engine' >&2; exit 2; }
root="$HF_CACHE/tensorfold/deepseek-v41-flash-1017"
model="$root/models/$MODEL_VARIANT"
for file in "$model/config.json" "$model/model.safetensors.index.json" "$root/engram/rank$rank/engram-l1-r${rank}of2.bin" "$root/engram/rank$rank/engram-l14-r${rank}of2.bin" "$root/assets/rank$rank/rope.bin" "$root/assets/rank$rank/engram-host.bin" "$root/assets/rank$rank/aot/aot.json" "$root/verification/pack.verified.json"; do
 [[ -s "$file" ]] || { echo "TensorFold prerequisite missing: $file. See the embedded README.md." >&2; exit 2; }
done
if [[ $MODEL_VARIANT == abliterated ]]; then
 [[ -s "$model/ABLIT_META.json" ]] || { echo 'Missing verified 2.8bpw L10-35 overlay' >&2; exit 2; }
fi
