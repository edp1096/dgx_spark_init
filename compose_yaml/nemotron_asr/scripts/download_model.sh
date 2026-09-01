#!/usr/bin/env bash
set -euo pipefail

model_dir="${NEMO_MODEL_DIR:-/home/edp1096/.cache/nemo-speech}"
model_file="${NEMO_MODEL_FILE:-nemotron-3.5-asr-streaming-0.6b.q8_0.gguf}"
model_url="https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/resolve/main/${model_file}"

mkdir -p "$model_dir"
if [[ -s "$model_dir/$model_file" ]]; then
  echo "already present: $model_dir/$model_file"
  exit 0
fi

partial="$model_dir/$model_file.partial"
curl --fail --location --retry 5 --continue-at - --output "$partial" "$model_url"
mv "$partial" "$model_dir/$model_file"
echo "downloaded: $model_dir/$model_file"
