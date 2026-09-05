#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
set -a
source "$script_dir/.env"
set +a
if [[ ${RUNTIME_HF_TOKEN+x} ]]; then export HF_TOKEN="$RUNTIME_HF_TOKEN"; fi
export ENV_FILE="$script_dir/.env"
case "${1:-status}" in
 setup) "$script_dir/runtime.sh" image; "$script_dir/models.sh" prepare ;;
 image) docker pull "$DSPARK_VLLM_IMAGE"; ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_HOST" docker pull "$DSPARK_VLLM_IMAGE" ;;
 model|prepare) exec bash "$script_dir/models.sh" prepare ;;
 start) exec bash "$script_dir/upstream/start-deepseek-v4-flash-dspark.sh" ;;
 stop) exec bash "$script_dir/upstream/stop-deepseek-v4-flash-dspark.sh" ;;
 restart) "$script_dir/runtime.sh" stop; exec "$script_dir/runtime.sh" start ;;
 status) exec bash "$script_dir/upstream/status-deepseek-v4-flash-dspark.sh" ;;
 logs) if [[ "${2:-head}" == worker ]]; then exec ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_HOST" docker logs -f --tail 160 deepseek-v4-flash-vllm-dspark-1; else exec docker logs -f --tail 160 deepseek-v4-flash-vllm-dspark-1; fi ;;
 validate) exec bash "$script_dir/upstream/validate-dspark-config.sh" ;;
 *) exit 2 ;;
esac
