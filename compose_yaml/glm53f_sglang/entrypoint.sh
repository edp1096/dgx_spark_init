#!/usr/bin/env bash
set -euo pipefail
: "${NODE_RANK:?}" "${MODEL_HOST_PATH:?}" "${HEAD_RAIL_IP:?}" "${SGLANG_HOST_IP:?}" "${NCCL_IB_HCA:?}"
if [[ -z "${NCCL_IB_GID_INDEX:-}" ]]; then
 IFS=. read -r a b c d <<<"$SGLANG_HOST_IP"
 printf -v wanted '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x' "$a" "$b" "$c" "$d"
 for f in "/sys/class/infiniband/$NCCL_IB_HCA/ports/1/gids/"*; do
  i="${f##*/}"
  [[ "$(<"$f")" == "$wanted" ]] || continue
  [[ "$(<"/sys/class/infiniband/$NCCL_IB_HCA/ports/1/gid_attrs/types/$i")" == *v2* ]] || continue
  export NCCL_IB_GID_INDEX="$i"; break
 done
fi
: "${NCCL_IB_GID_INDEX:?No matching RoCE v2 GID}"
[[ "${MAX_NUM_SEQS:-1}" == 1 ]] || {
 echo 'This SGLang GLM profile is qualified for MAX_NUM_SEQS=1 only' >&2; exit 2;
}
case "${CHUNKED_PREFILL_SIZE:-4096}" in
 1024|2048|4096) ;;
 *) echo 'CHUNKED_PREFILL_SIZE must be 1024, 2048, or 4096' >&2; exit 2 ;;
esac
case "${DECODE_CUDA_GRAPH:-1}" in
 1) graph_config='{"decode":{"backend":"full","max_bs":1,"bs":[1]},"prefill":{"backend":"disabled"}}' ;;
 0) graph_config='{"decode":{"backend":"disabled"},"prefill":{"backend":"disabled"}}' ;;
 *) echo 'DECODE_CUDA_GRAPH must be 0 or 1' >&2; exit 2 ;;
esac
[[ "${MTP_TOKENS:-0}" == 0 ]] || {
 echo 'MTP is not qualified for this GLM profile; use DFlash2' >&2; exit 2;
}
spec_args=()
preflight_draft=""
algorithm="${SPECULATIVE_ALGORITHM:-DFLASH}"
[[ "${DFLASH_TOKENS:-5}" != 0 ]] || algorithm=none
case "$algorithm" in
 none) ;;
 DFLASH)
  : "${DRAFT_MODEL_HOST_PATH:?DFLASH requires a local draft checkpoint}"
  case "${DFLASH_TOKENS:-5}" in
   [1-7]) ;;
   *) echo 'DFLASH_TOKENS must be between 1 and 7' >&2; exit 2 ;;
  esac
  [[ -f "$DRAFT_MODEL_HOST_PATH/config.json" ]] || {
   echo "Missing draft config: $DRAFT_MODEL_HOST_PATH/config.json" >&2; exit 2;
  }
  spec_args=(--speculative-algorithm DFLASH
   --speculative-draft-model-path "$DRAFT_MODEL_HOST_PATH"
   --speculative-num-draft-tokens "$(( ${DFLASH_TOKENS:-5} + 1 ))"
   --speculative-draft-model-quantization unquant
   --speculative-draft-attention-backend triton
   --speculative-draft-kv-cache-dtype "${DRAFT_KV_CACHE_DTYPE:-fp8_e4m3}"
   --speculative-draft-window-size 2048)
  preflight_draft="$DRAFT_MODEL_HOST_PATH"
  ;;
 *) echo 'SPECULATIVE_ALGORITHM must be none or DFLASH' >&2; exit 2 ;;
esac
python3 /opt/glm53/check_models.py "$MODEL_HOST_PATH" "rank $NODE_RANK" "$preflight_draft"
python3 /opt/glm53/validate_processor.py "$MODEL_HOST_PATH"
python3 /opt/glm53/prepare_chat_template.py "$MODEL_HOST_PATH" /root/.cache/glm53-chat.jinja
exec python3 -m sglang.launch_server \
 --model-path "$MODEL_HOST_PATH" --served-model-name "${SERVED_MODEL_NAME:-glm-5.3-flash}" --dtype bfloat16 \
 --chat-template /root/.cache/glm53-chat.jinja \
 --host "${API_BIND:-127.0.0.1}" --port "${API_PORT:-8000}" \
 --tp-size 2 --nnodes 2 --node-rank "$NODE_RANK" \
 --dist-init-addr "$HEAD_RAIL_IP:${MASTER_PORT:-29531}" \
 --context-length "${MAX_MODEL_LEN:-1048576}" --max-total-tokens "${MAX_TOTAL_TOKENS:-${MAX_MODEL_LEN:-1048576}}" \
 --max-running-requests 1 --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-4096}" \
 --mem-fraction-static "${MEM_FRACTION_STATIC:-0.94}" --max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE:-8}" \
 --model-loader-extra-config '{"enable_multithread_load":false}' --weight-loader-drop-cache-after-load \
 --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 --page-size 64 \
 --attention-backend dsa --dsa-prefill-backend b12x_glm --dsa-decode-backend b12x_glm \
 --linear-attn-decode-backend triton --linear-attn-prefill-backend triton \
 --moe-runner-backend flashinfer_cutlass --fp4-gemm-backend flashinfer_cutlass \
 --disable-shared-experts-fusion --cuda-graph-config "$graph_config" \
 --enable-cache-report --enable-metrics --disable-flashinfer-autotune --reasoning-parser glm45 --tool-call-parser glm47 \
 "${spec_args[@]}"
