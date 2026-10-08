#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${NEMO_MODEL_DIR:-${HOME}/.cache/nemo-speech}"
model_file="${NEMO_MODEL_FILE:-nemotron-3.5-asr-streaming-0.6b.q5_k.gguf}"
revision="ea30d66debe3740a08b573244286791d423d6b3e"
ref="${NEMO_SPEECH_REF:-6a3ca369370782acee0dd155e34394ba60b920c4}"
converter_image="sparktalk-nemotron-converter:$ref"
q5_sha="f0dab30ca22a606c2ab9a65efae88421851c6ca4ab01160708ab733b4467ce58"
q8_file="nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
q8_sha="3fc991d3badad7277c11030a7519832cddaf2057aafed6d4b25147e953a070b1"
mkdir -p "$model_dir"

verified() {
  printf '%s  %s\n' "$2" "$model_dir/$1" | sha256sum --check --status 2>/dev/null
}

fetch() {
  local filename="$1" digest="$2" url="$3"
  if verified "$filename" "$digest"; then
    echo "verified: $model_dir/$filename"
    return
  fi
  curl --fail --location --retry 5 --continue-at - --output "$model_dir/$filename.partial" "$url"
  printf '%s  %s\n' "$digest" "$model_dir/$filename.partial" | sha256sum --check
  mv "$model_dir/$filename.partial" "$model_dir/$filename"
}

if [[ "$model_file" = nemotron-3.5-asr-streaming-0.6b.q5_k.gguf ]]; then
  if ! verified "$model_file" "$q5_sha"; then
    fetch "$q8_file" "$q8_sha" "https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/resolve/$revision/$q8_file"
    f16_file="nemotron-3.5-asr-streaming-0.6b.f16.gguf"
    if ! verified "$f16_file" cc5fc1b6e05fdb905c4d9bc01842e1d297650cc39e117334bf16d64eb2b55fc0; then
      source_file="nemotron-3.5-asr-streaming-0.6b.nemo"
      fetch "$source_file" 210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74 \
        "https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/resolve/$revision/$source_file"
    fi
    docker build --target converter --build-arg "NEMO_SPEECH_REF=$ref" -t "$converter_image" "$root"
    docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
      --memory 12g --memory-swap 12g -v "$model_dir:/models" --entrypoint python \
      "$converter_image" /opt/nemo-quant/prepare_q5.py
  fi
  verified "$model_file" "$q5_sha"
elif [[ "$model_file" = "$q8_file" ]]; then
  fetch "$q8_file" "$q8_sha" "https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/resolve/$revision/$q8_file"
elif [[ ! -s "$model_dir/$model_file" ]]; then
  echo "custom model missing: $model_dir/$model_file; use convert_model.sh" >&2
  exit 1
fi

diar_file="Nemotron-3-Diarization.q8_0.gguf"
fetch "$diar_file" 08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1 \
  "https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/f667ed73aee57d40cc39428eb768b4fd87a0a29e/$diar_file"
