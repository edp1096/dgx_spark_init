#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/models.sh"
select_model
ensure_image() {
 if ! docker image inspect "$RUNTIME_IMAGE" >/dev/null 2>&1; then
  docker build -t "$RUNTIME_IMAGE" -f "$script_dir/Dockerfile.dflash2" "$script_dir"
 fi
}
compose() { docker compose -p qwen27-managed -f "$script_dir/compose.managed.yaml" "$@"; }
case "$1" in
 image) ensure_image;;
 model|setup) prepare_model;;
 start|restart)
  [[ -f "$MODEL_HOST_PATH/config.json" ]] || { echo 'Run ./manage.sh setup first' >&2; exit 1; }
  ensure_image
  if [[ "$1" == restart ]]; then docker stop "$RUNTIME_CONTAINER" >/dev/null 2>&1 || true; fi
  if docker inspect "$RUNTIME_CONTAINER" >/dev/null 2>&1; then
   [[ "$(docker inspect -f '{{.State.Running}}' "$RUNTIME_CONTAINER")" != true ]] || { echo 'Container is already running; use restart.' >&2; exit 1; }
   docker rm "$RUNTIME_CONTAINER"
  fi
  compose up -d;;
 stop) docker stop "$RUNTIME_CONTAINER";;
 status) docker inspect -f '{{.State.Status}}' "$RUNTIME_CONTAINER";;
 logs) docker logs -f --tail 100 "$RUNTIME_CONTAINER";;
 validate) compose config -q; printf 'variant=%s\nrepository=%s\nrevision=%s\nmodel_path=%s\n' "$MODEL_VARIANT" "$MODEL_REPO" "$MODEL_REVISION" "$MODEL_HOST_PATH";;
 *) exit 2;;
esac
