#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export QWEN_TP2_WORKER="$WORKER_HOST" QWEN_TP2_WORKER_ROOT="$REMOTE_COMPOSE_DIR"
export QWEN_TP2_HEAD="$HEAD_RAIL_IP" QWEN_TP2_WORKER_RAIL="$WORKER_RAIL_IP"
export QWEN_TP2_API_PORT="$API_PORT" QWEN_TP2_DIST_PORT="$MASTER_PORT"
export QWEN_TP2_HEAD_CONTAINER="$HEAD_CONTAINER" QWEN_TP2_WORKER_CONTAINER="$WORKER_CONTAINER"
export QWEN_TP2_MODEL="$SERVED_MODEL_NAME"
case ${1:-status} in
 start) bash models.sh; exec python3 manage_tp2.py start --context "$MAX_MODEL_LEN";;
 stop|status) exec python3 manage_tp2.py "$1";;
 validate) bash models.sh;;
 *) echo 'Use start|stop|status|validate; prepare the pinned checkpoint and image on both hosts first.' >&2; exit 2;;
esac
