#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
rank=${1:?rank required}
input=${2:?cache path relative to the runtime cache required}
case "$rank" in 0|1) ;; *) exit 2;; esac
if docker inspect -f '{{.State.Running}}' "ds41-stream-$rank" 2>/dev/null | rg -q '^true$'; then
  echo 'Stop the model before isolated dense tuning' >&2; exit 1
fi
docker run --rm --gpus all --name "ds41-dense-tune-$rank" \
  --network host --ipc host --memory 32g --memory-swap 32g \
  -v "$PWD:/opt/ds41:ro" -v "$HOME/.cache/ds41-stream:/cache" \
  -e OMP_NUM_THREADS=1 -e GLOO_SOCKET_IFNAME=enp1s0f1np1 \
  -e XDG_CACHE_HOME=/cache -e B12X_COMPILE_CACHE_DIR=/cache/b12x \
  -e CUTE_DSL_CACHE_DIR=/cache/cute \
  -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer -e VLLM_CACHE_ROOT=/cache/vllm \
  -e VLLM_HAS_FLASHINFER_CUBIN=1 -e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 \
  --entrypoint timeout dgx-ds41-stream:b12x8 900s \
  python3 /opt/ds41/tools/tune_dense_prefill.py --rank "$rank" \
  --input-cache "/cache/$input" --output-cache /cache/dense-prefill-8192.json \
  --output "/cache/dense-prefill-rank$rank.json"
