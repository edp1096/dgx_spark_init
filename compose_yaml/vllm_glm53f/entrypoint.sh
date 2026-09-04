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

[[ -f /models/glm53-exl3/config.json ]] || {
  echo "EXL3 weights are missing; run the download profile on this node first." >&2
  exit 2
}

args=(
  vllm serve /models/glm53-exl3
  --served-model-name glm-5.3-flash
  --host 0.0.0.0 --port "${API_PORT:-8000}"
  --trust-remote-code --quantization exl3
  --tensor-parallel-size 2
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}"
  --max-model-len "${MAX_MODEL_LEN:-524288}"
  --max-num-seqs "${MAX_NUM_SEQS:-4}"
  --block-size "${BLOCK_SIZE:-2304}"
  --mm-processor-cache-gb "${MM_PROCESSOR_CACHE_GB:-0.5}"
  --tool-call-parser glm47 --enable-auto-tool-choice
  --reasoning-parser glm45
  --default-chat-template-kwargs '{"enable_thinking": false}'
  --distributed-executor-backend mp
  --nnodes 2 --node-rank "$NODE_RANK"
  --master-addr "$HEAD_RAIL_IP" --master-port "${MASTER_PORT:-29521}"
)

[[ -n "${LOAD_FORMAT:-}" ]] && args+=(--load-format "$LOAD_FORMAT")
[[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]] && args+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
[[ -n "${KV_CACHE_DTYPE:-}" ]] && args+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
[[ -n "${KV_CACHE_DTYPE_SKIP_LAYERS:-}" ]] && args+=(--kv-cache-dtype-skip-layers "$KV_CACHE_DTYPE_SKIP_LAYERS")
[[ -n "${ATTENTION_BACKEND:-}" ]] && args+=(--attention-backend "$ATTENTION_BACKEND")
[[ -n "${KV_CACHE_MEMORY:-}" ]] && args+=(--kv-cache-memory "$KV_CACHE_MEMORY")
[[ -n "${PREFIX_MATCH_UNIT:-}" ]] && args+=(--prefix-match-unit "$PREFIX_MATCH_UNIT")
[[ -n "${KDA_PREFILL_BACKEND:-}" ]] && args+=(--kda-prefill-backend "$KDA_PREFILL_BACKEND")
[[ -n "${MIXED_PREFILL_DECODE_WEIGHT:-}" ]] && args+=(--mixed-prefill-decode-weight "$MIXED_PREFILL_DECODE_WEIGHT")
[[ -n "${MIXED_PREFILL_CAP:-}" ]] && args+=(--mixed-prefill-token-cap "$MIXED_PREFILL_CAP")
[[ "${SKIP_MM_PROFILING:-1}" != 0 ]] && args+=(--skip-mm-profiling)
[[ "${ENFORCE_EAGER:-0}" != 0 ]] && args+=(--enforce-eager)

if [[ "${MTP_TOKENS:-0}" != 0 ]]; then
  args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}")
elif [[ "${SPEC_METHOD:-dflash}" == dflash ]]; then
  [[ -f /models/glm53-dflash2/config.json && -f /models/glm53-dflash2/model.safetensors ]] || {
    echo "DFlash2 weights are incomplete; set SPEC_METHOD=none or run the download profile." >&2
    exit 2
  }
  args+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"/models/glm53-dflash2\",\"num_speculative_tokens\":${DFLASH_TOKENS:-7}}")
fi

args+=("${headless[@]}")
echo "Starting GLM-5.3 rank=$NODE_RANK host=$VLLM_HOST_IP gid=$NCCL_IB_GID_INDEX"
exec "${args[@]}"
