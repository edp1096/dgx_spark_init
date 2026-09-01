#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${NEMO_MODEL_DIR:-/home/edp1096/.cache/nemo-speech}"
cache_dir="${HF_HOME:-/home/edp1096/.cache/huggingface}"
outtype="${1:-f16}"
outfile="nemotron-3.5-asr-streaming-0.6b.${outtype}.gguf"
ref="${NEMO_SPEECH_REF:-4f9676226f667d14608487df744f375db87127f8}"

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
