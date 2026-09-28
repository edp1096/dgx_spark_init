#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
image=${QWEN_TP2_IMAGE:?}
worker=${WORKER_HOST:?}
for p in "$HF_CACHE" "$WORKER_HF_CACHE" "$REMOTE_COMPOSE_DIR"; do
 [[ "$p" =~ ^/[A-Za-z0-9._/-]+$ ]] || { echo 'Unsupported preparation path' >&2; exit 2; }
done
if ! docker image inspect "$image" >/dev/null 2>&1; then
 docker build --target tp2 -t "$image" build
fi
head_id=$(docker image inspect -f '{{.Id}}' "$image")
worker_id=$(ssh -o BatchMode=yes "$worker" "docker image inspect -f '{{.Id}}' '$image'" 2>/dev/null || true)
if [[ "$head_id" != "$worker_id" ]]; then docker save "$image" | ssh -o BatchMode=yes "$worker" docker load; fi
if [[ ${1:-setup} == image ]]; then exit; fi
repo=edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4
mkdir -p "$HF_CACHE/$repo"
printf '%s\n' "${HF_TOKEN:-}" | docker run --rm -i --memory 2g --memory-swap 2g -e HF_HUB_OFFLINE=0 -e HF_HOME=/hf -v "$HF_CACHE:/hf" --entrypoint python3 "$image" -c 'import sys; from huggingface_hub import snapshot_download; token=sys.stdin.readline().strip(); snapshot_download(sys.argv[1], revision="016905fbd6ac5c584799007a524827e2486a0711", local_dir="/hf/"+sys.argv[1], token=token or False, max_workers=1)' "$repo"
ssh -o BatchMode=yes "$worker" "mkdir -p '$WORKER_HF_CACHE/$repo'"
rsync -aL --partial --exclude='.cache/' "$HF_CACHE/$repo/" "$worker:$WORKER_HF_CACHE/$repo/"
echo 'Qwen TP2 images and checkpoint prepared on both hosts'
