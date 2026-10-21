#!/usr/bin/env bash
set -euo pipefail

: "${NODE_RANK:?NODE_RANK must be 0 (head) or 1 (worker)}"
: "${VLLM_HOST_IP:?VLLM_HOST_IP is required}"
: "${HEAD_RAIL_IP:?HEAD_RAIL_IP is required}"
: "${NCCL_IB_HCA:?NCCL_IB_HCA is required}"

case "$NODE_RANK" in
  0) headless=() ;;
  1) headless=(--headless) ;;
  *) echo "NODE_RANK must be 0 or 1" >&2; exit 2 ;;
esac

if [[ -z "${NCCL_IB_GID_INDEX:-}" ]]; then
  IFS=. read -r a b c d <<<"$VLLM_HOST_IP"
  printf -v wanted '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x' "$a" "$b" "$c" "$d"
  for gid_file in "/sys/class/infiniband/$NCCL_IB_HCA/ports/1/gids/"*; do
    index="${gid_file##*/}"
    [[ "$(<"$gid_file")" == "$wanted" ]] || continue
    [[ "$(<"/sys/class/infiniband/$NCCL_IB_HCA/ports/1/gid_attrs/types/$index")" == *v2* ]] || continue
    export NCCL_IB_GID_INDEX="$index"
    break
  done
fi

if [[ -z "${NCCL_IB_GID_INDEX:-}" ]]; then
  echo "No RoCE v2 GID for $VLLM_HOST_IP on $NCCL_IB_HCA; set NCCL_IB_GID_INDEX explicitly." >&2
  exit 2
fi

draft_path=""
case "${DFLASH_TOKENS:-5}" in
 0) ;;
 [1-7]) draft_path="${DRAFT_MODEL_HOST_PATH:?DFlash checkpoint required}" ;;
 *) echo 'DFLASH_TOKENS must be an integer from 0 to 7' >&2; exit 2 ;;
esac
if [[ "${NCCL_NCHANNELS:-auto}" != auto ]]; then
  [[ "$NCCL_NCHANNELS" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid NCCL_NCHANNELS' >&2; exit 2; }
  export NCCL_MIN_NCHANNELS="$NCCL_NCHANNELS" NCCL_MAX_NCHANNELS="$NCCL_NCHANNELS"
fi
python3 /opt/glm53/check_models.py "${MODEL_HOST_PATH:?}" "rank $NODE_RANK" "$draft_path"
python3 /opt/glm53/prepare_chat_template.py "$MODEL_HOST_PATH" /cache/chat_template.jinja
args=(vllm serve "$MODEL_HOST_PATH"
  --chat-template /cache/chat_template.jinja
  --served-model-name "${SERVED_MODEL_NAME:-glm-5.3-flash}"
  --host "${VLLM_BIND:-127.0.0.1}" --port "${API_PORT:-8000}"
  --moe-backend b12x --linear-backend b12x
  --skip-mm-profiling --mm-processor-cache-gb 0.5
  --dtype bfloat16 --block-size 256 --no-enable-flashinfer-autotune
  --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill
  --tensor-parallel-size 2 --distributed-executor-backend mp
  --data-parallel-backend mp --nnodes 2 --node-rank "$NODE_RANK"
  --master-addr "$HEAD_RAIL_IP" --master-port "${MASTER_PORT:-29521}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.88}"
  --max-model-len "${MAX_MODEL_LEN:-1048576}"
  --max-num-seqs "${MAX_NUM_SEQS:-4}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-1024}"
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}"
  --async-scheduling --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE:-16}"
  --prefix-cache-retention-interval "${PREFIX_CACHE_RETENTION_INTERVAL:-4608}"
  --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45
)
: "${KV_CACHE_MEMORY=10240000000}"
[[ -z "$KV_CACHE_MEMORY" ]] || args+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY")
[[ "${ENFORCE_EAGER:-0}" == 0 ]] || args+=(--enforce-eager)
if [[ -n "$draft_path" ]]; then
  spec=$(python3 - "$draft_path" "${DFLASH_TOKENS:-5}" <<'PY'
import json,sys
print(json.dumps({'method':'dflash','model':sys.argv[1],'num_speculative_tokens':int(sys.argv[2]),
 'attention_backend':'TRITON_ATTN','kv_cache_dtype':'auto','draft_sample_method':'probabilistic',
 'rejection_sample_method':'standard','enable_adaptive_verification':False,'disable_eagle_block_drop':False}))
PY
  )
  args+=(--speculative-config "$spec" --per-request-spec-decode-metrics summary)
fi
if [[ "${MTP_TOKENS:-0}" != 0 ]]; then
  echo "MTP is not qualified for this TP2/1M recipe; MTP_TOKENS must be 0." >&2
  exit 2
fi

args+=("${headless[@]}")
echo "Starting GLM NVFP4 rank=$NODE_RANK host=$VLLM_HOST_IP gid=$NCCL_IB_GID_INDEX"
exec "${args[@]}"
