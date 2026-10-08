#!/usr/bin/env bash
set -euo pipefail
python3 /opt/qwen-image21/prepare_models.py --release-file-cache
mkdir -p /tmp/qwen-image21/input /tmp/qwen-image21/output
case "${SPARKTALK_IMAGE_RESIDENCY:-legacy}" in
  dit)
    [[ "${SPARKTALK_KEEP_MODELS_LOADED:-0}" == 1 ]] || { echo "DiT residency requires KEEP_MODELS_LOADED=1" >&2; exit 1; }
    export JOB_DIR="${JOB_DIR:-/job}"
    mkdir -p "$JOB_DIR"/{requests,results,output,cancel,state}
    rm -f "$JOB_DIR"/{ready.json,worker-state.json,stop,worker.pid} "$JOB_DIR"/requests/* "$JOB_DIR"/results/* "$JOB_DIR"/output/* "$JOB_DIR"/cancel/*
    cp /opt/qwen-image21/resident_settings.json "$JOB_DIR/settings.json"
    python3 /opt/qwen-image21/resident_worker.py &
    comfy_pid=$!
    echo "$comfy_pid" > "$JOB_DIR/worker.pid"
    ;;
  legacy)
memory_args=(--disable-smart-memory)
case "${SPARKTALK_KEEP_MODELS_LOADED:-0}" in
  0) ;;
  1) memory_args=() ;;
  *) echo "SPARKTALK_KEEP_MODELS_LOADED must be 0 or 1" >&2; exit 1 ;;
esac
python3 /opt/qwen-image21/run_comfy.py --listen 127.0.0.1 --port "${COMFY_PORT:-8188}" \
  --disable-auto-launch --disable-pinned-memory "${memory_args[@]}" --cache-classic \
  --disable-all-custom-nodes --disable-api-nodes --preview-method none --reserve-vram 4 \
  --input-directory /tmp/qwen-image21/input --output-directory /tmp/qwen-image21/output &
comfy_pid=$!
    ;;
  *) echo "SPARKTALK_IMAGE_RESIDENCY must be legacy or dit" >&2; exit 1 ;;
esac
uvicorn api:app --host "${API_HOST:-127.0.0.1}" --port "${API_PORT:-8691}" --workers 1 &
api_pid=$!
shutdown() {
  kill -TERM "$api_pid" "$comfy_pid" 2>/dev/null || true
  sleep 2
  kill -KILL "$api_pid" "$comfy_pid" 2>/dev/null || true
  wait "$api_pid" "$comfy_pid" 2>/dev/null || true
}
trap shutdown EXIT
trap 'exit 0' TERM INT
wait -n "$api_pid" "$comfy_pid"
