#!/usr/bin/env bash
set -euo pipefail
# Registration reuses the prepared, pinned checkpoint; never downloads or requantizes it.
[[ ${MODEL_VARIANT:-official} == official ]] || { echo 'DS41 streaming supports original weights only' >&2; exit 1; }
docker image inspect "$DSV41_IMAGE" >/dev/null
[[ -f "$HF_CACHE/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277/config.json" ]]
for ((layer=0; layer<43; layer++)); do
 printf -v file '%s/rank%s/layer-%02d.bin' "$(dirname "$HF_CACHE")/ds41-packed" "$1" "$layer"
 [[ -s "$file" ]] || { echo "Missing prepared expert file: $file" >&2; exit 1; }
done
