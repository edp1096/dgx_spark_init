#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="$script_dir/.env"
compose=(docker compose --project-directory "$script_dir" -f "$script_dir/compose.yaml")

ensure_env() {
  if [[ ! -f "$env_file" ]]; then
    cp "$script_dir/env.sample" "$env_file"
    chmod 600 "$env_file"
    echo "Created $env_file from env.sample."
  fi
}

load_env() {
  ensure_env
  set -a
  # shellcheck disable=SC1091
  . "$env_file"
  set +a
if [[ ${RUNTIME_HF_TOKEN+x} ]]; then export HF_TOKEN="$RUNTIME_HF_TOKEN"; fi
}

usage() {
  cat <<'EOF'
usage: ./manage.sh setup | build | model | start | stop | restart | status | logs

  setup    build the GB10 image and download the model
  build    rebuild the image, including the CUDA extension
  model    download or resume the model
  start    start the API container
  stop     remove this Compose stack
  restart  recreate the API container
  status   show the container and API health
  logs     follow server logs
EOF
  exit 2
}

build() {
  load_env
  "${compose[@]}" build exl3
}

model() {
  load_env
  "$script_dir/models.sh" download
}

start() {
  load_env
  "$script_dir/models.sh" status >/dev/null || {
    echo "Model is missing. Run: ./manage.sh model" >&2
    exit 1
  }
  mkdir -p "$EXL3_CACHE_PATH/torch_extensions"
  "${compose[@]}" up -d exl3
  echo "API starting at http://127.0.0.1:${API_PORT:-8000}/v1"
}

stop() {
  load_env
  "${compose[@]}" down
}

status() {
  load_env
  "${compose[@]}" ps
  if curl -fsS --max-time 3 "http://127.0.0.1:${API_PORT:-8000}/health"; then
    echo
    echo "API healthy: http://127.0.0.1:${API_PORT:-8000}/v1"
  else
    echo "API not ready"
  fi
}

logs() {
  load_env
  "${compose[@]}" logs -f exl3
}

case "${1:-}" in
  setup) build; model ;;
  build|image) build ;;
  model) model ;;
  validate) load_env; "${compose[@]}" config -q ;;
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  logs) logs ;;
  *) usage ;;
esac
