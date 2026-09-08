#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
caller_token="${HF_TOKEN-${HUGGING_FACE_HUB_TOKEN-}}"
set -a
source "$script_dir/.env"
set +a
if [[ ${RUNTIME_HF_TOKEN+x} ]]; then export HF_TOKEN="$RUNTIME_HF_TOKEN"; fi
[[ -z "$caller_token" ]] || HF_TOKEN="$caller_token"
unset caller_token
MODEL_VARIANT="${MODEL_VARIANT:-official}"
repo="$DSPARK_MODEL_OFFICIAL"
[[ "$MODEL_VARIANT" != abliterated ]] || repo="$DSPARK_MODEL_ABLITERATED"
model_cache="$HF_CACHE/hub/models--${repo//\//--}"
case "$HF_CACHE" in /*) ;; *) echo 'HF_CACHE must be absolute' >&2; exit 2;; esac
case "${1:-prepare}" in
 status) test -d "$model_cache/snapshots"; exit ;;
 prepare|download) ;;
 *) echo 'Usage: models.sh prepare|download|status' >&2; exit 2;;
esac
secret=$(mktemp)
trap 'rm -f "$secret"' EXIT
chmod 600 "$secret"
if [[ -n "${HF_TOKEN:-}" ]]; then printf '%s' "$HF_TOKEN" > "$secret";
elif [[ -f "$HF_CACHE/token" ]]; then cat "$HF_CACHE/token" > "$secret"; fi
mkdir -p "$HF_CACHE"
docker run --rm -i --user "$(id -u):$(id -g)" \
 -v "$HF_CACHE:/cache/huggingface" -v "$secret:/run/secrets/hf_token:ro" \
 -v "$script_dir/download.py:/opt/download.py:ro" -v "$script_dir/tensor_patch.py:/opt/tensor_patch.py:ro" \
 -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=0 -e HF_HUB_DISABLE_XET=1 \
 -e HF_HUB_DOWNLOAD_TIMEOUT=120 -e MODEL_VARIANT \
 -e DSPARK_MODEL_OFFICIAL -e DSPARK_MODEL_ABLITERATED -e DSPARK_REVISION -e DSPARK_REVISION_ABLITERATED \
 --entrypoint /usr/bin/python3 "$DSPARK_VLLM_IMAGE" /opt/download.py
if [[ "${1:-prepare}" == prepare ]]; then
 worker_cache="${WORKER_HF_CACHE:-$HF_CACHE}/hub/models--${repo//\//--}"
 [[ "$worker_cache" =~ ^/[A-Za-z0-9._/-]+$ ]] || { echo 'Unsupported worker cache path' >&2; exit 2; }
 # Existing model caches may have been created by a root-owned download container.
 worker_uid=$(ssh -o BatchMode=yes "$WORKER_HOST" id -u)
 worker_gid=$(ssh -o BatchMode=yes "$WORKER_HOST" id -g)
 ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_HOST" "mkdir -p '$worker_cache' 2>/dev/null || docker run --rm --user 0:0 -v '${WORKER_HF_CACHE:-$HF_CACHE}:/cache' --entrypoint /bin/sh '$DSPARK_VLLM_IMAGE' -c 'mkdir -p \"/cache/hub/models--${repo//\//--}\" && chown $worker_uid:$worker_gid \"/cache/hub/models--${repo//\//--}\"'"
 if [[ "$MODEL_VARIANT" == abliterated ]]; then
  echo "워커 선택 텐서 전송 및 로컬 원본으로 구성"
  rsync --info=progress2 -a "$model_cache/tensor-patches/" "$WORKER_HOST:$worker_cache/tensor-patches/"
  scp -q "$script_dir/tensor_patch.py" "$WORKER_HOST:$worker_cache/tensor_patch.py"
  ssh -o BatchMode=yes "$WORKER_HOST" "docker run --rm --user 0:0 -v '${WORKER_HF_CACHE:-$HF_CACHE}:/cache' --entrypoint /usr/bin/python3 '$DSPARK_VLLM_IMAGE' '/cache/hub/models--${repo//\//--}/tensor_patch.py' '/cache/hub/models--${repo//\//--}/tensor-patches/$DSPARK_REVISION_ABLITERATED' /cache"
  exit
 fi
 echo "워커 모델 동기화 시작"
 rsync --info=progress2 -a --no-owner --no-group --exclude='*.incomplete' -e 'ssh -o BatchMode=yes -o ConnectTimeout=10' "$model_cache/" "$WORKER_HOST:$worker_cache/"
fi
