#!/usr/bin/env bash
set -euo pipefail
docker image inspect "$QWEN_TP2_IMAGE" >/dev/null
file="$HF_CACHE/edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4/config.json"
[[ -s "$file" ]] || { echo "Missing main checkpoint: $file" >&2; exit 1; }
