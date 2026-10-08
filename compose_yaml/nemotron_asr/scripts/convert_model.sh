#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${NEMO_MODEL_DIR:-${HOME}/.cache/nemo-speech}"
cache_dir="${HF_HOME:-${HOME}/.cache/huggingface}"
outtype="${1:-f16}"
outfile="nemotron-3.5-asr-streaming-0.6b.${outtype}.gguf"
ref="${NEMO_SPEECH_REF:-6a3ca369370782acee0dd155e34394ba60b920c4}"

# The qualified Q5 release preserves the pinned Q8 protected tensors.
if [[ "$outtype" = q5_k ]]; then
  NEMO_MODEL_FILE="$outfile" bash "$root/scripts/download_model.sh"
  exit 0
fi

mkdir -p "$model_dir" "$cache_dir"
if [[ -e "$model_dir/$outfile" ]]; then
  echo "refusing to overwrite existing model: $model_dir/$outfile" >&2
  exit 1
fi

docker build --target converter \
  --build-arg "NEMO_SPEECH_REF=$ref" \
  -t "sparktalk-nemotron-converter:$ref" "$root"
docker run --rm \
  -v "$model_dir:/models" \
  -v "$cache_dir:/cache" \
  "sparktalk-nemotron-converter:$ref" \
  nvidia/nemotron-3.5-asr-streaming-0.6b \
  --outfile "/models/$outfile" \
  --outtype "$outtype" \
  --cache-dir /cache
echo "converted: $model_dir/$outfile"
