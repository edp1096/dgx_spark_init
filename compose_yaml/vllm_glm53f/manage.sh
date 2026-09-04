#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="$script_dir/.env"

if [[ ! -f "$env_file" ]]; then
  cp "$script_dir/env.sample" "$env_file"
  echo "Created $env_file from the defaults for head .61 / worker .60."
fi

set -a
# shellcheck disable=SC1091
. "$env_file"
set +a

: "${WORKER_LAN_IP:?WORKER_LAN_IP is required}"
: "${WORKER_USER:?WORKER_USER is required}"
: "${HEAD_RAIL_IP:?HEAD_RAIL_IP is required}"
: "${WORKER_RAIL_IP:?WORKER_RAIL_IP is required}"
: "${HEAD_NCCL_IF:?HEAD_NCCL_IF is required}"
: "${WORKER_NCCL_IF:?WORKER_NCCL_IF is required}"

worker="${WORKER_USER}@${WORKER_LAN_IP}"
remote_dir="${REMOTE_COMPOSE_DIR:-/home/edp1096/workspace/dgx_spark_init/compose_yaml/vllm_glm53f}"
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10)
head_compose=(docker compose --project-directory "$script_dir" -f "$script_dir/compose.yaml" -f "$script_dir/compose.head.yaml")
prefix="${NCCL_SUBNET##*/}"
ssh_ready=0

usage() {
  cat <<'EOF'
usage: ./manage.sh setup | start | stop | restart | status | logs [head|worker]

  setup    configure the rail, download once, sync models, and pull images
  start    configure the rail and start worker -> head
  stop     stop head -> worker
  restart  stop and start in the safe order
  status   show both containers and API health
  logs     follow head logs (or pass worker)
EOF
  exit 2
}

require_ssh() {
  [[ "$ssh_ready" == 1 ]] && return
  if [[ -n "${HEAD_LAN_IP:-}" ]] && \
     ! ip -o -4 addr show | grep -Fq " $HEAD_LAN_IP/"; then
    echo "This host does not own HEAD_LAN_IP=$HEAD_LAN_IP." >&2
    echo "Update HEAD_LAN_IP and WORKER_LAN_IP in $env_file after an OS reinstall or DHCP change." >&2
    exit 1
  fi
  ssh "${ssh_opts[@]}" "$worker" true || {
    echo "Passwordless SSH is not ready. Run this once, then retry:" >&2
    echo "  ssh-copy-id $worker" >&2
    exit 1
  }
  remote_name=$(ssh "${ssh_opts[@]}" "$worker" hostname)
  echo "Worker connected: $remote_name ($WORKER_LAN_IP)"
  ssh_ready=1
}

sync_compose() {
  require_ssh
  ssh "${ssh_opts[@]}" "$worker" "mkdir -p '$remote_dir'"
  rsync -a --partial --exclude '.git/' \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=10" \
    "$script_dir/" "$worker:$remote_dir/"
}

setup_rail() {
  require_ssh
  ip link show dev "$HEAD_NCCL_IF" | grep -q LOWER_UP || {
    echo "Head QSFP interface $HEAD_NCCL_IF has no carrier." >&2
    exit 1
  }
  ssh "${ssh_opts[@]}" "$worker" "ip link show dev '$WORKER_NCCL_IF' | grep -q LOWER_UP" || {
    echo "Worker QSFP interface $WORKER_NCCL_IF has no carrier." >&2
    exit 1
  }

  if ! ip -o -4 addr show dev "$HEAD_NCCL_IF" | grep -q " $HEAD_RAIL_IP/"; then
    echo "Assigning $HEAD_RAIL_IP/$prefix to head $HEAD_NCCL_IF."
    docker run --rm --privileged --network host alpine:3.22 sh -c \
      "ip address add '$HEAD_RAIL_IP/$prefix' dev '$HEAD_NCCL_IF' && ip link set '$HEAD_NCCL_IF' up"
  fi
  if ! ssh "${ssh_opts[@]}" "$worker" \
    "ip -o -4 addr show dev '$WORKER_NCCL_IF' | grep -q ' $WORKER_RAIL_IP/'"; then
    echo "Assigning $WORKER_RAIL_IP/$prefix to worker $WORKER_NCCL_IF."
    ssh "${ssh_opts[@]}" "$worker" \
      "docker run --rm --privileged --network host alpine:3.22 sh -c \"ip address add '$WORKER_RAIL_IP/$prefix' dev '$WORKER_NCCL_IF' && ip link set '$WORKER_NCCL_IF' up\""
  fi

  rail_ready=0
  for _ in $(seq 1 10); do
    if ping -c 1 -W 1 "$WORKER_RAIL_IP" >/dev/null; then
      rail_ready=1
      break
    fi
    sleep 1
  done
  [[ "$rail_ready" == 1 ]] || {
    echo "QSFP rail cannot reach worker at $WORKER_RAIL_IP." >&2
    exit 1
  }
  echo "QSFP rail ready: $HEAD_RAIL_IP -> $WORKER_RAIL_IP via $HEAD_NCCL_IF"
}

remote_compose() {
  ssh "${ssh_opts[@]}" "$worker" \
    "cd '$remote_dir' && docker compose -f compose.yaml -f compose.worker.yaml $*"
}

drop_caches() {
  echo "Clearing cold filesystem cache on both nodes."
  docker run --rm --privileged alpine:3.22 sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
  ssh "${ssh_opts[@]}" "$worker" \
    "docker run --rm --privileged alpine:3.22 sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"
}

setup() {
  setup_rail
  "$script_dir/models.sh" prepare
  sync_compose
  remote_compose pull glm53
  "${head_compose[@]}" pull glm53
}

start() {
  setup_rail
  sync_compose
  drop_caches
  remote_compose up -d glm53
  "${head_compose[@]}" up -d glm53
  echo "Cluster started. Follow readiness with: ./manage.sh logs"
}

stop() {
  require_ssh
  "${head_compose[@]}" down
  remote_compose down
}

status() {
  require_ssh
  echo "HEAD"
  "${head_compose[@]}" ps
  echo "WORKER"
  remote_compose ps
  if curl -fsS --max-time 3 "http://127.0.0.1:${API_PORT:-8000}/health" >/dev/null; then
    echo "API healthy: http://${HEAD_LAN_IP:-127.0.0.1}:${API_PORT:-8000}/v1"
  else
    echo "API not ready"
  fi
}

logs() {
  case "${1:-head}" in
    head) "${head_compose[@]}" logs -f glm53 ;;
    worker) require_ssh; remote_compose logs -f glm53 ;;
    *) usage ;;
  esac
}

case "${1:-}" in
  setup)
    setup
    echo "Setup complete. Start the cluster with: ./manage.sh start"
    ;;
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  logs) logs "${2:-head}" ;;
  *) usage ;;
esac
