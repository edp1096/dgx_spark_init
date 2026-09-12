#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10)
remote() { ssh "${ssh_opts[@]}" "$WORKER_HOST" "$@"; }
quote() { printf '%q' "$1"; }
case ${1:-status} in
 rank)
  rank=$2
  if [[ $rank == 1 ]]; then export HF_CACHE="$WORKER_HF_CACHE"; export DSV41_CONTAINER="$WORKER_CONTAINER"; else export DSV41_CONTAINER="$HEAD_CONTAINER"; fi
  bash models.sh "$rank"
  export DSV41_PACKED_ROOT="$(dirname "$HF_CACHE")/ds41-packed"
  export DSV41_RUNTIME_CACHE="$(dirname "$HF_CACHE")/ds41-stream"
  export DSV41_SLOTS_PER_LAYER=224 DSV41_MAX_MODEL_LEN="$MAX_MODEL_LEN"
  export DSV41_FINAL_DECODER_ROWS=0 DSV41_BENCH_CONTROL=0
  exec bash launch.sh "$rank";;
 start)
  [[ ${MODEL_VARIANT:-official} == official ]] || { echo 'Only original DS41 weights supported' >&2; exit 1; }
  [[ $(docker inspect -f '{{.State.Running}}' "$HEAD_CONTAINER" 2>/dev/null || true) != true ]]
  [[ $(remote "docker inspect -f '{{.State.Running}}' $(quote "$WORKER_CONTAINER") 2>/dev/null || true") != true ]]
  remote "mkdir -p $(quote "$REMOTE_COMPOSE_DIR") && chmod 700 $(quote "$REMOTE_COMPOSE_DIR")"
  tar -czf - --exclude=.git . | remote "tar -xzf - -C $(quote "$REMOTE_COMPOSE_DIR")"
  docker rm "$HEAD_CONTAINER" >/dev/null 2>&1 || true
  remote "docker rm $(quote "$WORKER_CONTAINER") >/dev/null 2>&1 || true"
  remote "bash $(quote "$REMOTE_COMPOSE_DIR/manage.sh") rank 1"
  if ! bash manage.sh rank 0; then
   remote "docker stop $(quote "$WORKER_CONTAINER")" || true
   exit 1
  fi;;
 stop)
  docker stop "$HEAD_CONTAINER" || true
  remote "docker stop $(quote "$WORKER_CONTAINER")";;
 status)
  docker ps -a --filter "name=$HEAD_CONTAINER"
  remote "docker ps -a --filter name=$(quote "$WORKER_CONTAINER")";;
 setup|model|prepare|image)
  echo 'Automatic DS41 image/model preparation is not available; prepare the pinned checkpoint, packed experts and b12x8 image first.' >&2; exit 2;;
 validate)
  echo 'DS41 requires the prepared b12x8 image, pinned HF checkpoint and rank-specific packed experts on both hosts.'
  bash models.sh 0
  echo 'Head prerequisites verified. Worker prerequisites are checked before its rank starts.';;
 logs) docker logs --tail 100 "$HEAD_CONTAINER";;
 *) echo 'Usage: manage.sh start|stop|status|validate|logs' >&2; exit 2;;
esac
