#!/usr/bin/env bash
set -euo pipefail

model_dir="${NEMO_MODEL_DIR:-${HOME}/.cache/nemo-speech}"
model_file="${NEMO_MODEL_FILE:-nemotron-3.5-asr-streaming-0.6b.q8_0.gguf}"
model_url="https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/resolve/main/${model_file}"

mkdir -p "$model_dir"
if [[ -s "$model_dir/$model_file" ]]; then
  echo "already present: $model_dir/$model_file"
else

partial="$model_dir/$model_file.partial"
curl --fail --location --retry 5 --continue-at - --output "$partial" "$model_url"
mv "$partial" "$model_dir/$model_file"
echo "downloaded: $model_dir/$model_file"
fi

# Official Q8 artifact, pinned to the upstream model manifest and verified.
diar_file="Nemotron-3-Diarization.q8_0.gguf"
diar_sha="08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1"
if ! printf '%s  %s\n' "$diar_sha" "$model_dir/$diar_file" | sha256sum --check --status 2>/dev/null; then
  curl --fail --location --retry 5 --continue-at - --output "$model_dir/$diar_file.partial"     "https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/f667ed73aee57d40cc39428eb768b4fd87a0a29e/$diar_file"
  printf '%s  %s\n' "$diar_sha" "$model_dir/$diar_file.partial" | sha256sum --check
  mv "$model_dir/$diar_file.partial" "$model_dir/$diar_file"
fi
