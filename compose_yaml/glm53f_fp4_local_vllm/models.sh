#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$script_dir/.env" ]]; then
 set -a; source "$script_dir/.env"; set +a
fi
source "$script_dir/select_model.sh"
if [[ ${RUNTIME_HF_TOKEN+x} ]]; then export HF_TOKEN="$RUNTIME_HF_TOKEN"; fi
worker="${WORKER_USER:-$(id -un)}@${MODEL_SYNC_HOST:-${WORKER_LAN_IP:-}}"
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10)
[[ "$MODEL_HOST_PATH" =~ ^/[A-Za-z0-9._/-]+$ ]] || { echo 'Unsupported model path' >&2; exit 2; }
check() { python3 "$script_dir/check_models.py" "$MODEL_HOST_PATH"; }
check_target() { check; }
download_snapshot() {
 mkdir -p "$HF_CACHE" "$HF_CACHE/.glm53-xet-$(id -u)"
 docker run --rm --network host --memory 2g --user "$(id -u):$(id -g)" \
  -e HF_TOKEN -e HF_HOME="$HF_CACHE" -e HOME=/tmp \
  -e HF_XET_CACHE="$HF_CACHE/.glm53-xet-$(id -u)" \
  -e HF_HUB_DISABLE_XET=0 -e HF_XET_HIGH_PERFORMANCE=0 \
  -e HF_XET_NUM_CONCURRENT_RANGE_GETS=1 \
  -e HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS=1 \
  -e HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY=1 \
  -e HF_XET_CLIENT_AC_MIN_DOWNLOAD_CONCURRENCY=1 \
  -e HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY=1 \
  -v "$HF_CACHE:$HF_CACHE" --entrypoint python3 \
  "${GLM53_IMAGE:-eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595}" -c \
  'import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], max_workers=1)' \
  "$1" "$2" "$3"
}
download() {
 if ! check_target >/dev/null 2>&1; then download_snapshot "$MODEL_REPO" "$MODEL_REVISION" "$MODEL_HOST_PATH"; fi
 check
}
sync_models() {
 check
 [[ -n "${WORKER_LAN_IP:-}" ]] || { echo 'WORKER_LAN_IP required' >&2; exit 2; }
 for path in "$MODEL_HOST_PATH"; do
  [[ -n "$path" ]] || continue
  ssh "${ssh_opts[@]}" "$worker" "mkdir -p '$path'"
  rsync -aL --checksum --partial --human-readable --info=progress2 \
   -e 'ssh -o BatchMode=yes -o ConnectTimeout=10' \
   "$path/" "$worker:$path/"
 done
 ssh "${ssh_opts[@]}" "$worker" "python3 - '$MODEL_HOST_PATH' worker" < "$script_dir/check_models.py"
}
case "${1:-}" in
 download) download ;;
 sync) sync_models ;;
 prepare) download; sync_models ;;
 status) check ;;
 *) echo 'usage: models.sh download | sync | prepare | status' >&2; exit 2 ;;
esac
