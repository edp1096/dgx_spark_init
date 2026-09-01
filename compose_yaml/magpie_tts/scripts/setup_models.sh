#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "${script_dir}/.." && pwd)
model_dir=${MAGPIE_MODEL_DIR:-/home/edp1096/.cache/nemo-speech/magpie-v2607}
magpie_archive=${model_dir}/magpie_tts_multilingual_357m.nemo
magpie_gguf=${model_dir}/magpie-v2607-pr17-speaker-order-v2.f16.gguf
codec_gguf=${model_dir}/nano-codec.decoder.f16.gguf

mkdir -p "${model_dir}" "${model_dir}/extracted"

download() {
  local url=$1
  local target=$2
  local -a curl_args=(--fail --location --retry 5 --continue-at -)
  if [[ -s "${target}" ]]; then
    echo "using existing ${target}"
    return
  fi
  if [[ -n "${HF_TOKEN:-}" ]]; then
    curl_args+=(--header "Authorization: Bearer ${HF_TOKEN}")
  fi
  curl "${curl_args[@]}" --output "${target}.part" "${url}"
  mv "${target}.part" "${target}"
}

download \
  "https://huggingface.co/nvidia/magpie_tts_multilingual_357m/resolve/v2607/magpie_tts_multilingual_357m.nemo" \
  "${magpie_archive}"
download \
  "https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps/resolve/main/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf" \
  "${codec_gguf}"

tar -xf "${magpie_archive}" -C "${model_dir}/extracted"

if [[ ! -s "${magpie_gguf}" ]]; then
  docker compose --project-directory "${project_dir}" build converter
  docker compose --project-directory "${project_dir}" --profile tools run --rm converter \
    /models/magpie_tts_multilingual_357m.nemo \
    --outfile /models/magpie-v2607-pr17-speaker-order-v2.f16.gguf
fi

echo "Magpie v2607 models are ready in ${model_dir}"
