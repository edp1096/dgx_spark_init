#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
rank=${1:?usage: launch.sh 0|1}
case "$rank" in
  0) rail=10.200.0.1; headless=();;
  1) rail=10.200.0.2; headless=(--headless);;
  *) exit 2;;
esac
model="$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277"
image=${DSV41_IMAGE:-dgx-ds41-stream:b12x8}
cache=${DSV41_EXPERT_CACHE_GIB:-8}
backend=${DSV41_MOE_BACKEND:-b12x_slots}
model_len=${DSV41_MAX_MODEL_LEN:-65536}
kv_cache_bytes=${DSV41_KV_CACHE_BYTES:-2147483648}
prefix_interval=${DSV41_PREFIX_CACHE_INTERVAL:-128}
batch_default=512
if [[ "$backend" == b12x_slots ]]; then batch_default=4096; fi
batch_tokens=${DSV41_MAX_BATCHED_TOKENS:-$batch_default}
kernel_default=512; shared_default=0; scratch_default=256
if [[ "$backend" == b12x_slots && ${DSV41_MODEL_GRAPHS:-0} == 0 ]]; then
  kernel_default=2048; shared_default=1; scratch_default=384
fi
kernel_tokens=${DSV41_KERNEL_TOKENS:-$kernel_default}
shared_buffers=${DSV41_SHARED_BUFFERS:-$shared_default}
scratch_mib=${DSV41_SCRATCH_MIB:-$scratch_default}
case "$kernel_tokens" in 512|1024|2048) ;; *) echo 'Invalid expert kernel capacity' >&2; exit 1;; esac
case ${DSV41_FINAL_DECODER_ROWS:-0} in 0|1) ;; *) echo 'Invalid final decoder row flag' >&2; exit 1;; esac
if [[ ${DSV41_FINAL_DECODER_ROWS:-0} == 1 && ${DSV41_MODEL_GRAPHS:-0} != 0 ]]; then
  echo 'Final decoder row selection requires eager execution' >&2; exit 1
fi
case $shared_buffers in 0|1) ;; *) echo 'Invalid shared buffer flag' >&2; exit 1;; esac
case ${DSV41_ROUTED_PIPELINE:-0} in 0|1) ;; *) echo 'Invalid routed pipeline flag' >&2; exit 1;; esac
[[ "$scratch_mib" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid scratch size' >&2; exit 1; }
(( scratch_mib >= 256 )) || { echo 'Expert scratch requires at least 256 MiB' >&2; exit 1; }
if (( kernel_tokens == 2048 && scratch_mib < 290 )); then
  echo '2048-token expert kernels require at least 290 MiB scratch' >&2; exit 1
fi
if [[ ${DSV41_ROUTED_PIPELINE:-0} == 1 ]]; then
  [[ ${DSV41_EXPERT_IO:-batch_overlap} == batch_overlap && $shared_buffers == 1 && ${DSV41_CACHE_LAYOUT:-} == '' ]] || { echo 'Routed staging requires shared buffers, native batch overlap and a uniform cache' >&2; exit 1; }
fi
if [[ ${DSV41_MODEL_GRAPHS:-0} == 1 && ( $shared_buffers == 1 || ${DSV41_ROUTED_PIPELINE:-0} == 1 ) ]]; then
  echo 'Shared execution and routed staging require eager model execution' >&2; exit 1
fi
[[ "$model_len" =~ ^[1-9][0-9]*$ && "$kv_cache_bytes" =~ ^[1-9][0-9]*$ ]] || { echo 'Context length and KV cache bytes must be positive integers' >&2; exit 1; }
[[ "$prefix_interval" =~ ^[0-9]+$ ]] || { echo 'Invalid prefix cache retention interval' >&2; exit 1; }
[[ "$batch_tokens" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid prefill batch size' >&2; exit 1; }
if [[ "$backend" == b12x_slots ]]; then
  (( batch_tokens <= 4096 )) || { echo 'Streaming prefill is qualified up to 4096 tokens' >&2; exit 1; }
  if (( batch_tokens > 2048 )) && [[ $shared_buffers != 1 ]]; then
    echo 'Prefill above 2048 requires shared expert I/O buffers' >&2; exit 1
  fi
  if (( batch_tokens > 2048 )) && [[ -n ${DSV41_PROBE_TOKEN:-} ]]; then
    python3 - "$rank" "$DSV41_PROBE_TOKEN" <<'GUARD'
import json,os,sys
from pathlib import Path
rank,token=sys.argv[1:]
p=Path.home()/'.local/state/ds41-probes'/token/f'ready-rank{rank}.json'
x=json.loads(p.read_text())
assert x['token']==token and x['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert b'probe_watchdog.py' in Path(f"/proc/{x['pid']}/cmdline").read_bytes()
os.kill(x['pid'],0)
GUARD
  fi
  if (( batch_tokens > 512 )) && [[ ${DSV41_MODEL_GRAPHS:-0} == 1 ]]; then
    echo 'Larger streaming prefills require eager whole-model execution' >&2
    exit 1
  fi
fi
DSV41_SPEC_TOKENS=${DSV41_SPEC_TOKENS:-5}
case ${DSV41_ADAPTIVE_VERIFY:-0} in 0) adaptive=false;;1) adaptive=true;;*) echo 'Invalid adaptive verification flag' >&2; exit 1;;esac
if [[ "$adaptive" == true && ${DSV41_MODEL_GRAPHS:-0} != 1 ]]; then
  echo 'The pinned adaptive verifier requires model graph samples; eager adaptive verification is not qualified' >&2
  exit 1
fi
expert_io=${DSV41_EXPERT_IO:-batch_overlap}
case "$expert_io" in serial|overlap|batch|batch_overlap) ;; *) echo 'Invalid expert I/O mode' >&2; exit 1;; esac
if [[ "$expert_io" != serial ]]; then
  [[ "$backend" == b12x_slots && ${DSV41_SLOT_IO:-direct} == direct && ${DSV41_MODEL_GRAPHS:-0} == 0 && ${DSV41_PREFETCH_TEST:-0} == 0 ]] || { echo 'Scheduled expert I/O requires eager b12x direct I/O without forecast prefetch' >&2; exit 1; }
fi
avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
if (( avail < 40 )); then
  echo "Need >=40 GiB available for reference boot, have ${avail} GiB" >&2
  exit 1
fi
gpu_cache=${DSV41_GPU_CACHE_GIB:-$(( avail > 72 ? 48 : avail - 24 ))}
[ -f "$model/config.json" ]
# Pick the RoCE-v2 IPv4 GID for this exact rail, not an assumed index.
gid=''
expected_gid=$(python3 -c 'import ipaddress,sys; print(ipaddress.IPv6Address("::ffff:"+sys.argv[1]).exploded)' "$rail")
for f in /sys/class/infiniband/rocep1s0f1/ports/1/gids/*; do
  i=${f##*/}
  [[ $(cat "$f") == "$expected_gid" ]] || continue
  [[ $(cat "/sys/class/infiniband/rocep1s0f1/ports/1/gid_attrs/types/$i" 2>/dev/null) == 'RoCE v2' ]] || continue
  gid=$i
  break
done
[ -n "$gid" ] || { echo 'No RoCE IPv4 GID found' >&2; exit 1; }
mounts=()
while read -r src dst; do
  [ -n "$src" ] || continue
  mounts+=(-v "$PWD/patches/$src:/usr/local/lib/python3.12/dist-packages/vllm/$dst:ro")
done < patches/mounts.txt
extra=()
spec=()
execution=(--enforce-eager)
if [[ ${DSV41_MODEL_GRAPHS:-0} == 1 ]]; then
  [[ "$backend" == b12x_slots && ${DSV41_PREFETCH_TEST:-0} == 0 ]] || { echo 'Model graphs require b12x_slots without forecast prefetch' >&2; exit 1; }
  [[ ${DSV41_ENGRAM_PRESTAGE:-1} == 1 ]] || { echo 'Model graphs require Engram prestaging' >&2; exit 1; }
  execution=(--compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,3,4,5,6]}')
  if [[ ${DSV41_ATTENTION_GRAPHS:-0} == 1 ]]; then
    execution=(--compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[1,2,3,4,5,6]}')
  fi
fi
if [[ ${DSV41_SHORT_CONTEXT_GRAPHS:-0} == 1 ]]; then
  [[ ${DSV41_MODEL_GRAPHS:-0} == 1 && ${DSV41_ATTENTION_GRAPHS:-0} == 1 ]] || { echo 'Short-context graphs require model and attention graphs' >&2; exit 1; }
fi
if [[ "$backend" == b12x_slots ]]; then
  packed="$HOME/.cache/ds41-packed/rank$rank"
  expected_layers=40
  if (( ${DSV41_SPEC_TOKENS:-0} > 0 )); then expected_layers=43; fi
  for ((layer=0; layer<expected_layers; layer++)); do
    printf -v packed_file '%s/layer-%02d.bin' "$packed" "$layer"
    [ -f "$packed_file" ] || { echo "Missing packed layer: $packed_file" >&2; exit 1; }
  done
  slots=${DSV41_SLOTS_PER_LAYER:-$(( ((avail-30)*1073741824/376012800)/32*32 ))}
  (( slots > 224 )) && slots=224
  (( slots >= 32 )) || { echo 'Insufficient memory for streaming slots' >&2; exit 1; }
  draft_slots=0
  if (( ${DSV41_SPEC_TOKENS:-0} > 0 )); then draft_slots=$((slots<128?slots:128)); fi
  target_slots=$((40*slots))
  if [[ -n ${DSV41_CACHE_LAYOUT:-} ]]; then
    target_slots=$(python3 cache_layout.py "$DSV41_CACHE_LAYOUT" "$slots")
  fi
  required=$(( ((target_slots+3*draft_slots)*9400320+1073741823)/1073741824+26 ))
  # Sum additional byte allocations before rounding once to GiB. Rounding
  # KV, scratch and staging separately overstates the required memory by 1 GiB.
  extra_bytes=0
  if (( kv_cache_bytes > 536870912 )); then extra_bytes=$((extra_bytes+kv_cache_bytes-536870912)); fi
  if (( scratch_mib > 256 )); then extra_bytes=$((extra_bytes+(scratch_mib-256)*1048576)); fi
  if [[ ${DSV41_ROUTED_PIPELINE:-0} == 1 ]]; then
    (( slots == 224 )) || { echo 'Routed staging is qualified with 224 target slots' >&2; exit 1; }
    extra_bytes=$((extra_bytes+1504051200))
  fi
  required=$((required+(extra_bytes+1073741823)/1073741824))
  # Retain the 2048 allowance, plus 2 GiB for the qualified 4096 scheduler.
  # Shared expert I/O stays bounded; dense/attention activations still grow.
  if (( batch_tokens > 2048 )); then required=$((required+5));
  elif (( batch_tokens > 512 )); then required=$((required+(batch_tokens-512+511)/512)); fi
  if [[ ${DSV41_PREFETCH_TEST:-0} == 1 ]]; then required=$((required+1)); fi
  if [[ ${DSV41_MODEL_GRAPHS:-0} == 1 ]]; then required=$((required+3)); fi
  (( avail >= required )) || { echo "Need $required GiB, available $avail GiB" >&2; exit 1; }
  echo "Streaming target_slots=$target_slots layout=${DSV41_CACHE_LAYOUT:-uniform-$slots}; available=${avail}GiB estimated_required=${required}GiB"
  extra+=(-v "$packed:/packed:ro" -e DSV41_PACKED_DIR=/packed
          -e DSV41_SLOTS_PER_LAYER="$slots"
          -e DSV41_CACHE_LAYOUT="${DSV41_CACHE_LAYOUT:-}"
          -e DSV41_ROUTED_PIPELINE="${DSV41_ROUTED_PIPELINE:-0}"
          -e DSV41_KERNEL_TOKENS="$kernel_tokens"
          -e DSV41_SHARED_BUFFERS="$shared_buffers"
          -e DSV41_SCRATCH_MIB="$scratch_mib" -e DSV41_RELEASE_CHECKPOINT_CACHE=1
          -e DSV41_READ_THREADS="${DSV41_READ_THREADS:-4}"
          -e DSV41_SLOT_IO="${DSV41_SLOT_IO:-direct}"
          -e DSV41_EXPERT_IO="$expert_io"
          -e DSV41_PREFETCH_TEST="${DSV41_PREFETCH_TEST:-0}"
          -e DSV41_MODEL_GRAPHS="${DSV41_MODEL_GRAPHS:-0}"
          -e DSV41_ATTENTION_GRAPHS="${DSV41_ATTENTION_GRAPHS:-0}"
          -e DSV41_SHORT_CONTEXT_GRAPHS="${DSV41_SHORT_CONTEXT_GRAPHS:-0}"
          -e DSV41_PROBE_TOKEN="${DSV41_PROBE_TOKEN:-}" -e PYTHONFAULTHANDLER=1
          -e DSV41_FINAL_DECODER_ROWS="${DSV41_FINAL_DECODER_ROWS:-0}" \
          -e DSV41_BENCH_CONTROL="${DSV41_BENCH_CONTROL:-0}"
          -e DSV41_ENGRAM_PRESTAGE="${DSV41_ENGRAM_PRESTAGE:-${DSV41_MODEL_GRAPHS:-0}}"
          -e VLLM_USE_BREAKABLE_CUDAGRAPH=1
          -e XDG_CACHE_HOME=/cache
          -e DSV41_EXPERT_GRAPHS="${DSV41_EXPERT_GRAPHS:-1}"
          -e B12X_COMPILE_CACHE_DIR=/cache/b12x -e CUTE_DSL_CACHE_DIR=/cache/cute
          -e VLLM_PLUGINS=)
fi
if (( ${DSV41_SPEC_TOKENS:-0} > 0 )); then
  spec=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$DSV41_SPEC_TOKENS,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":$adaptive}")
fi
mkdir -p "$HOME/.cache/ds41-stream"
docker run -d --gpus all --name "ds41-stream-$rank" \
  --label "ds41.probe=${DSV41_PROBE_TOKEN:-}" \
  --network host --ipc host --memory 100g --memory-swap 100g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband:/dev/infiniband \
  -v "${model%/snapshots/*}:/repo:ro" -v "$HOME/.cache/ds41-stream:/cache" \
  -v "$PWD:/opt/ds41:ro" "${mounts[@]}" \
  "${extra[@]}" \
  -e PYTHONPATH=/opt/ds41 -e DSV41_MODEL=/repo/snapshots/dba1be0a40aa45a94ad051997016db3960a90277 \
  -e DSV41_EXPERT_STREAMING=1 -e DSV41_EXPERT_CACHE_GIB="$cache" \
  -e DSV41_MOE_BACKEND="$backend" -e DSV41_GPU_CACHE_GIB="$gpu_cache" \
  -e DSV41_ENGRAM_DISK=1 -e DSV41_ENGRAM_DISK_THREADS=8 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_USE_RUST_FRONTEND=0 -e VLLM_HOST_IP="$rail" \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 -e VLLM_HAS_FLASHINFER_CUBIN=1 \
  -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer -e VLLM_CACHE_ROOT=/cache/vllm -e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton \
  -e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e OMP_NUM_THREADS=1 \
  -e NCCL_NET=IB -e NCCL_IB_HCA=rocep1s0f1 -e NCCL_IB_GID_INDEX="$gid" \
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e GLOO_SOCKET_IFNAME=enp1s0f1np1 \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_ROCE_VERSION_NUM=2 \
  -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ADDR_RANGE=10.200.0.0/24 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=WARN \
  "$image" /repo/snapshots/dba1be0a40aa45a94ad051997016db3960a90277 --served-model-name deepseek-v4.1-flash \
  --host 0.0.0.0 --port 8010 --tokenizer-mode deepseek_v41 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 \
  --enable-per-request-metrics --enable-prompt-tokens-details \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.65 --distributed-executor-backend mp \
  --nnodes 2 --node-rank "$rank" --master-addr 10.200.0.1 --master-port 29641 \
  "${execution[@]}" --language-model-only --engram-config '{"cpu_offload":false}' \
  --max-model-len "$model_len" --max-num-seqs 1 --max-num-batched-tokens "$batch_tokens" \
  --kv-cache-memory-bytes "$kv_cache_bytes" --block-size 128 \
  --prefix-cache-retention-interval "$prefix_interval" \
  --load-format safetensors --safetensors-load-strategy lazy \
  --default-chat-template-kwargs '{"thinking":false}' "${spec[@]}" "${headless[@]}"
