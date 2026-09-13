#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
rank=${1:?rank required}
[[ "$rank" == 0 || "$rank" == 1 ]]
out="$HOME/.local/state/ds41-probes/20260913-speed"
mkdir -p "$out"
docker run --rm --name "ds41-engram-reader-check-$rank" --network none --memory 6g --memory-swap 6g \
 -v "$PWD:/opt/ds41:ro" \
 -v "$PWD/patches/engram.py:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/engram.py:ro" \
 -v "${HF_CACHE:-$HOME/.cache/huggingface}/hub/models--deepseek-ai--DeepSeek-V4.1-Flash:/repo:ro" \
 -v "$out:/out" -e PYTHONPATH=/opt/ds41 -e XDG_CACHE_HOME=/out/cache \
 -e OMP_NUM_THREADS=1 -e DSV41_ENGRAM_DISK_THREADS=8 -e VLLM_PLUGINS= \
 --entrypoint python3 dgx-ds41-stream:b12x8 /opt/ds41/tools/bench_engram_reader.py \
 /repo/snapshots/dba1be0a40aa45a94ad051997016db3960a90277 --rank "$rank" --output "/out/reader-rank$rank.json"
