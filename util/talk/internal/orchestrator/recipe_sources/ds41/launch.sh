#!/usr/bin/env bash
# Adapted from jayleaton/deepseek-v41-tensorfold-spark scripts/run-zig.sh at 853310d.
# Engine settings are unchanged; Talk owns names, paths and lifecycle.
set -euo pipefail
rank=${1:?rank required}
set -a
source profile.env
# Preserve the serving geometry/AOT set; select only the existing draft policy.
case ${MTP_TOKENS:-auto} in
 auto) TF_DSV41_DEPTH=cost; TF_DSV41_DRAFT_DEPTH=5;;
 [1-5]) TF_DSV41_DEPTH=static; TF_DSV41_DRAFT_DEPTH=$MTP_TOKENS;;
 *) echo 'MTP_TOKENS must be auto or a fixed depth 1..5' >&2; exit 2;;
esac
root="$HF_CACHE/tensorfold/deepseek-v41-flash-1017"
MODEL_DIR="$root/models/$MODEL_VARIANT"
ENGRAM_DIR="$root/engram/rank$rank"
ASSETS_DIR="$root/assets/rank$rank"
PREPARED_DIR="$root/prepared/$MODEL_VARIANT/rank$rank"
STATE_DIR="$root/state/$MODEL_VARIANT/rank$rank"
CACHE_DIR="$root/cache/rank$rank"
TF_TP_MASTER="$HEAD_RAIL_IP"
TF_TP_PORT="$MASTER_PORT"
if [[ $rank == 0 ]]; then
 container="$HEAD_CONTAINER"; NCCL_SOCKET_IFNAME="$HEAD_NCCL_IF"; NCCL_IB_HCA="$HEAD_NCCL_HCA"
else
 container="$WORKER_CONTAINER"; NCCL_SOCKET_IFNAME="$WORKER_NCCL_IF"; NCCL_IB_HCA="$WORKER_NCCL_HCA"
fi
set +a
rope_rows=$(od -An -t u4 -j 8 -N 4 "$ASSETS_DIR/rope.bin" | tr -d ' ')
(( rope_rows >= TF_DSV41_CONTEXT + 2048 )) || { echo 'rope.bin is too short for the qualified context' >&2; exit 2; }
mkdir -p "$PREPARED_DIR" "$STATE_DIR/sessions" "$CACHE_DIR"
[[ -e "$CACHE_DIR/pfdense-table.json" ]] || cp pfdense-table.json "$CACHE_DIR/pfdense-table.json"
envs=()
while IFS='=' read -r key value; do
 case "$key" in TF_DSV41_*|GLM53_TF_*|NCCL_*) envs+=(-e "$key");; esac
done < <(env)
args=(docker run -d --init --name "$container" --gpus all --ipc=host --network host
 --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK
 --label sparktalk.recipe=ds41-tensorfold --label "sparktalk.variant=$MODEL_VARIANT"
 -v "$MODEL_DIR:/model:ro" -v "$ENGRAM_DIR:/engram:ro" -v "$ASSETS_DIR:/assets:ro"
 -v "$PREPARED_DIR:/prepared" -v "$STATE_DIR:/state" -v "$STATE_DIR/sessions:/sessions" -v "$CACHE_DIR:/cache"
 "${envs[@]}" -e TF_TP_WORLD=2 -e "TF_TP_RANK=$rank" -e TF_TP_DEVICE=0
 -e "TF_TP_MASTER=$TF_TP_MASTER" -e "TF_TP_PORT=$TF_TP_PORT" -e TF_COMM_BACKEND=roce
 -e TF_TP_KERNEL_IMAGE=/opt/tensorfold/share/tp_mailbox.fatbin "$DSV41_IMAGE")
if [[ $rank == 0 ]]; then
 args+=(tensorfold-dsv41 serve /model --host "${VLLM_HOST:-0.0.0.0}" --port "$API_PORT"
  --parallel "$TF_DSV41_SLOTS" --context "$TF_DSV41_CONTEXT" --name "$SERVED_MODEL_NAME" --max-tokens 32768)
else
 args+=(tf-dsv41-m1 follow /model /assets)
fi
exec "${args[@]}"
