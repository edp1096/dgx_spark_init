#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
runtime_dir="${LLAMA_RUNTIME_DIR:-${project_dir}/artifacts/current}"
model_dir="${LLAMA_MODEL_DIR:-/home/edp1096/.cache/gguf/huihui-gemma4-e2b-qat}"
model_file="${LLAMA_MODEL_FILE:-Huihui-gemma-4-E2B-it-qat-q4_0-unquantized-abliterated-Q4_K.gguf}"
mtp_file="${LLAMA_MTP_FILE:-mtp-ggml-model-bf16.gguf}"
mmproj_file="${LLAMA_MMPROJ_FILE:-mmproj-model-bf16.gguf}"
server_host="${LLAMA_HOST:-0.0.0.0}"
server_port="${LLAMA_PORT:-8696}"
context_size="${LLAMA_CONTEXT_SIZE:-65536}"

server_bin="${runtime_dir}/llama-server"
model_path="${model_dir}/${model_file}"
mtp_path="${model_dir}/${mtp_file}"
mmproj_path="${model_dir}/${mmproj_file}"

test -x "${server_bin}" || { echo "Missing llama-server: ${server_bin}" >&2; exit 1; }
test -f "${model_path}" || { echo "Missing model: ${model_path}" >&2; exit 1; }

export LD_LIBRARY_PATH="${runtime_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

args=(
  -m "${model_path}"
  --alias "${LLAMA_MODEL_ALIAS:-huihui-gemma4-e2b}"
  --jinja
  --reasoning off
  --no-mmap
  --ctx-size "${context_size}"
  --parallel "${LLAMA_PARALLEL:-1}"
  --host "${server_host}"
  --port "${server_port}"
  --n-gpu-layers all
  --flash-attn on
)

if [[ "${LLAMA_MMPROJ_ENABLED:-true}" == "true" && -f "${mmproj_path}" ]]; then
  args+=(--mmproj "${mmproj_path}")
else
  args+=(--no-mmproj)
fi

if [[ "${LLAMA_MTP_ENABLED:-true}" == "true" && -f "${mtp_path}" ]]; then
  args+=(
    --spec-type draft-mtp
    --spec-draft-n-max "${LLAMA_MTP_TOKENS:-4}"
    --spec-draft-model "${mtp_path}"
    --spec-draft-ngl all
  )
fi

exec "${server_bin}" "${args[@]}" "$@"
