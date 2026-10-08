#!/usr/bin/env bash
# Sourced by runtime.sh; preparation and runtime always select the same path.
select_model() {
 if [[ "$MODEL_VARIANT" == official ]]; then
  export MODEL_REPO="$OFFICIAL_REPO" MODEL_REVISION="$OFFICIAL_REVISION" MODEL_HOST_PATH="$MODEL_OFFICIAL_PATH"
 else
  export MODEL_REPO="$ABLITERATED_REPO" MODEL_REVISION="$ABLITERATED_REVISION" MODEL_HOST_PATH="$MODEL_ABLITERATED_PATH"
 fi
 # Retain a previous local checkpoint mount. Only recognize this exact model.
 if [[ "$MODEL_VARIANT" == abliterated && ! -e "$MODEL_HOST_PATH/config.json" ]]; then
  previous="$(docker inspect "$RUNTIME_CONTAINER" 2>/dev/null | python3 -c 'import json,sys,os
try:
 for m in json.load(sys.stdin)[0].get("Mounts",[]):
  if m.get("Type")=="bind" and m.get("Destination") in ["/models/qwen27", "/models/Huihui-RadixArk-Qwen3.8-27B-abliterated-NVFP4"] and os.path.basename(m["Source"])=="Huihui-RadixArk-Qwen3.8-27B-abliterated-NVFP4":
   print(m["Source"]); break
except (ValueError,IndexError,KeyError): pass' || true)"
  if [[ -n "$previous" && -f "$previous/config.json" ]]; then export MODEL_HOST_PATH="$previous"; fi
 fi
 export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Huihui-RadixArk-Qwen3.8-27B-abliterated-NVFP4}"
}
prepare_model() {
 if [[ "$(docker inspect -f '{{.State.Running}}' "$RUNTIME_CONTAINER" 2>/dev/null || true)" == true ]]; then
  echo 'Stop this model before preparing weights.' >&2; exit 1
 fi
 ensure_image
 mkdir -p "$HF_CACHE" "$MODEL_HOST_PATH"
 printf '%s\n' "${HF_TOKEN-}" | docker run --rm -i --network host \
  -v "$HF_CACHE:/root/.cache/huggingface" -v "$MODEL_HOST_PATH:/model" \
  -v "$script_dir/download_model.py:/download_model.py:ro" \
  -e MODEL_REPO -e MODEL_REVISION -e DRAFT_REPO -e DRAFT_REVISION \
  --entrypoint python3 "$RUNTIME_IMAGE" /download_model.py
}
