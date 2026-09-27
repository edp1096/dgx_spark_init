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
for path in "$MODEL_HOST_PATH" "$DRAFT_MODEL_HOST_PATH"; do
 [[ "$path" =~ ^/[A-Za-z0-9._/-]+$ ]] || { echo 'Unsupported model path' >&2; exit 2; }
done
draft_path=""
[[ "${DFLASH_TOKENS:-5}" == 0 ]] || draft_path="$DRAFT_MODEL_HOST_PATH"
check() { python3 "$script_dir/check_models.py" "$MODEL_HOST_PATH" "${1:-헤드}" "$draft_path"; }
check_target() { python3 "$script_dir/check_models.py" "$MODEL_HOST_PATH" 헤드; }
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
  "${GLM53_IMAGE:-pilcothink/vllm_spark_glm53@sha256:09da6eb394216d174ab8692758d90f9f458398d9c8fbc11ba6f04e93d5cf6392}" -c \
  'import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], max_workers=1)' \
  "$1" "$2" "$3"
}
download() {
 local repo revision
 if [[ "${MODEL_VARIANT:-official}" == official ]]; then
  repo=nvidia/GLM-5.3-Flash-NVFP4
  revision=09b04e5e74bca08ca8549fc736d4cdd8624bfde3
  # Reuse the original HF cache without another copy of the 190 GiB checkpoint.
  local snapshot="$HF_CACHE/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/$revision"
  if [[ ! -e "$MODEL_HOST_PATH" && ! -L "$MODEL_HOST_PATH" && -f "$snapshot/model.safetensors.index.json" ]]; then
   mkdir -p "$(dirname "$MODEL_HOST_PATH")"
   ln -s "../hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/$revision" "$MODEL_HOST_PATH"
  fi
 else
  repo=edp1096/Huihui-GLM-5.3-Flash-abliterated-NVFP4
  revision=48437d914cbde5c750093b9606a98508cb5430c0
 fi
 if ! check_target >/dev/null 2>&1; then
  download_snapshot "$repo" "$revision" "$MODEL_HOST_PATH"
 fi
 if [[ -n "$draft_path" ]] && ! python3 - "$script_dir" "$draft_path" <<'PYDRAFT'
import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from check_models import check_draft
try:check_draft(Path(sys.argv[2]))
except (OSError,ValueError,KeyError,TypeError):sys.exit(1)
PYDRAFT
 then
  download_snapshot incoai/GLM-5.3-Flash-DFlash2 bf582e4eacc1810f76656d1811693ff6c6737d2a "$draft_path"
 fi
 check
}
sync_models() {
 check
 [[ -n "${WORKER_LAN_IP:-}" ]] || { echo 'WORKER_LAN_IP required' >&2; exit 2; }
 for path in "$MODEL_HOST_PATH" "$draft_path"; do
  [[ -n "$path" ]] || continue
  ssh "${ssh_opts[@]}" "$worker" "mkdir -p '$path'"
  rsync -aL --checksum --partial --human-readable --info=progress2 \
   -e 'ssh -o BatchMode=yes -o ConnectTimeout=10' \
   "$path/" "$worker:$path/"
 done
 ssh "${ssh_opts[@]}" "$worker" "python3 - '$MODEL_HOST_PATH' 워커 '$draft_path'" < "$script_dir/check_models.py"
}
case "${1:-}" in
 download) download ;;
 sync) sync_models ;;
 prepare) download; sync_models ;;
 status) check ;;
 *) echo 'usage: models.sh download | sync | prepare | status' >&2; exit 2 ;;
esac
