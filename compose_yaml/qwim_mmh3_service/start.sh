#!/usr/bin/env bash
set -euo pipefail
python3 /opt/qwim-mmh3/prepare_models.py --release-file-cache
mkdir -p /job
cp /opt/qwim-mmh3/settings.json /job/settings.json
rm -f /job/ready.json /job/stop
python3 /opt/qwim-mmh3/worker.py &
worker_pid=$!
echo "$worker_pid" > /job/worker.pid
uvicorn api:app --host "${API_HOST:-127.0.0.1}" --port "${API_PORT:-8730}" --workers 1 &
api_pid=$!
cleanup(){ kill -TERM "$worker_pid" "$api_pid" 2>/dev/null || true; wait "$worker_pid" "$api_pid" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 0' TERM INT
wait -n "$worker_pid" "$api_pid"
