#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
image=${DSV41_IMAGE:?}
worker=${WORKER_HOST:?}
for p in "$HF_CACHE" "$WORKER_HF_CACHE" "$REMOTE_COMPOSE_DIR"; do
 [[ "$p" =~ ^/[A-Za-z0-9._/-]+$ ]] || { echo 'Unsupported preparation path' >&2; exit 2; }
done
if ! docker image inspect "$image" >/dev/null 2>&1; then docker build -t "$image" .; fi
head_id=$(docker image inspect -f '{{.Id}}' "$image")
worker_id=$(ssh -o BatchMode=yes "$worker" "docker image inspect -f '{{.Id}}' '$image'" 2>/dev/null || true)
if [[ "$head_id" != "$worker_id" ]]; then docker save "$image" | ssh -o BatchMode=yes "$worker" docker load; fi
[[ ${1:-setup} != image ]] || exit 0
repo=deepseek-ai/DeepSeek-V4.1-Flash
revision=dba1be0a40aa45a94ad051997016db3960a90277
mkdir -p "$HF_CACHE"
printf '%s\n' "${HF_TOKEN:-}" | docker run --rm -i --memory 2g --memory-swap 2g -e HF_HUB_OFFLINE=0 -e HF_HOME=/hf -v "$HF_CACHE:/hf" --entrypoint python3 "$image" -c 'import sys; from huggingface_hub import snapshot_download; token=sys.stdin.readline().strip(); snapshot_download(sys.argv[1], revision=sys.argv[2], cache_dir="/hf/hub", token=token or False, max_workers=1)' "$repo" "$revision"
cache=models--deepseek-ai--DeepSeek-V4.1-Flash
ssh -o BatchMode=yes "$worker" "mkdir -p '$WORKER_HF_CACHE/hub/$cache' '$REMOTE_COMPOSE_DIR'"
rsync -a --partial --exclude='*.incomplete' "$HF_CACHE/hub/$cache/" "$worker:$WORKER_HF_CACHE/hub/$cache/"
rsync -a --exclude='.env' ./ "$worker:$REMOTE_COMPOSE_DIR/"
model="/hf/hub/$cache/snapshots/$revision"
packed="$(dirname "$HF_CACHE")/ds41-packed"
worker_packed="$(dirname "$WORKER_HF_CACHE")/ds41-packed"
mkdir -p "$packed"
docker run --rm --gpus all --memory 8g --memory-swap 8g -v "$HF_CACHE:/hf:ro" -v "$packed:/packed" --entrypoint python3 "$image" /opt/ds41/pack_experts.py "$model" /packed/rank0 --rank 0
ssh -o BatchMode=yes "$worker" "mkdir -p '$worker_packed'; docker run --rm --gpus all --memory 8g --memory-swap 8g -v '$WORKER_HF_CACHE:/hf:ro' -v '$worker_packed:/packed' --entrypoint python3 '$image' /opt/ds41/pack_experts.py '$model' /packed/rank1 --rank 1"
echo 'DS41 checkpoint, images and rank-specific packed experts prepared'
