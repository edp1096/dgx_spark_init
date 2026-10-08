#!/usr/bin/env bash

source "$(dirname "$0")/common.sh"

repo_id="${HF_REPO_ID:-edp1096/Huihui-gemma-4-E2B-it-abliterated-litert-lm}"
bundle="${OUTPUT_DIR}/model.litertlm"
model_filename="Huihui-gemma-4-E2B-it-abliterated.litertlm"
export HF_HOME="${HF_HOME:-${ARTIFACT_DIR}/huggingface}"
mkdir -p "${HF_HOME}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN with write permission is required." >&2
  exit 1
fi
if [[ ! -f "${bundle}" ]]; then
  echo "Converted bundle is missing: ${bundle}" >&2
  exit 1
fi

"${VENV_DIR}/bin/hf" repo create "${repo_id}" --repo-type model --exist-ok --token "${HF_TOKEN}"
"${VENV_DIR}/bin/hf" upload "${repo_id}" "${bundle}" "${model_filename}" --token "${HF_TOKEN}"
"${VENV_DIR}/bin/hf" upload "${repo_id}" "${PROJECT_DIR}/huggingface/README.md" README.md --token "${HF_TOKEN}"
