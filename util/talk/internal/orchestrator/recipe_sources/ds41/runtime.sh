#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10)
remote() { ssh "${ssh_opts[@]}" "$WORKER_HOST" "$@"; }
quote() { printf '%q' "$1"; }
case ${1:-status} in
 rank)
  rank=${2:?rank required}
  if [[ $rank == 1 ]]; then export HF_CACHE="$WORKER_HF_CACHE"; fi
  bash models.sh "$rank"
  exec bash launch.sh "$rank";;
 start|validate|setup|model|prepare|image)
  action=$1
  bash models.sh 0
  remote "mkdir -p $(quote "$REMOTE_COMPOSE_DIR") && chmod 700 $(quote "$REMOTE_COMPOSE_DIR")"
  tar -czf - --exclude=.git . | remote "tar -xzf - -C $(quote "$REMOTE_COMPOSE_DIR")"
  remote "bash $(quote "$REMOTE_COMPOSE_DIR/manage.sh") check-rank 1"
  [[ $action == start ]] || { echo 'Verified TensorFold image, 1017 checkpoint, Engram and assets on both hosts'; exit 0; }
  [[ $(docker inspect -f '{{.State.Running}}' "$HEAD_CONTAINER" 2>/dev/null || true) != true ]] || { echo 'Head rank is already running' >&2; exit 1; }
  [[ $(remote "docker inspect -f '{{.State.Running}}' $(quote "$WORKER_CONTAINER") 2>/dev/null || true") != true ]] || { echo 'Worker rank is already running' >&2; exit 1; }
  docker rm "$HEAD_CONTAINER" >/dev/null 2>&1 || true
  remote "docker rm $(quote "$WORKER_CONTAINER") >/dev/null 2>&1 || true"
  rollback() { docker stop -t 30 "$HEAD_CONTAINER" >/dev/null 2>&1 || true; remote "docker stop -t 30 $(quote "$WORKER_CONTAINER") >/dev/null 2>&1 || true" || true; }
  trap rollback ERR
  remote "bash $(quote "$REMOTE_COMPOSE_DIR/manage.sh") rank 1"
  bash manage.sh rank 0
  trap - ERR;;
 check-rank)
  rank=${2:?rank required}
  if [[ $rank == 1 ]]; then export HF_CACHE="$WORKER_HF_CACHE"; fi
  exec bash models.sh "$rank";;
 stop)
  failed=0
  docker stop -t 30 "$HEAD_CONTAINER" || failed=1
  remote "docker stop -t 30 $(quote "$WORKER_CONTAINER")" || failed=1
  exit "$failed";;
 status)
  docker ps -a --filter "name=$HEAD_CONTAINER"
  remote "docker ps -a --filter name=$(quote "$WORKER_CONTAINER")";;
 logs) docker logs --tail 100 "$HEAD_CONTAINER";;
 *) echo 'Usage: manage.sh start|stop|status|validate|prepare|logs' >&2; exit 2;;
esac
