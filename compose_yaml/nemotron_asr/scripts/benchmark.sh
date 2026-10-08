#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
audio="${1:?usage: scripts/benchmark.sh AUDIO.wav [LANGUAGE]}"
language="${2:-ko-KR}"
endpoint="${NEMO_ENDPOINT:-http://127.0.0.1:8693}"
result_dir="$root/results/$(date +%Y%m%d-%H%M%S)-${language}"
mkdir -p "$result_dir"

curl -fsS "$endpoint/ready" | tee "$result_dir/ready.json"
free -b > "$result_dir/memory-before.txt"
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader \
  > "$result_dir/gpu-before.csv" || true

start_ns="$(date +%s%N)"
curl -fsS "$endpoint/v1/audio/transcriptions" \
  -F "file=@$audio" \
  -F "model=nemotron-3.5-asr-streaming-0.6b" \
  -F "language=$language" \
  -F "response_format=verbose_json" \
  | tee "$result_dir/transcription.json"
end_ns="$(date +%s%N)"

free -b > "$result_dir/memory-after.txt"
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader \
  > "$result_dir/gpu-after.csv" || true
awk -v start="$start_ns" -v end="$end_ns" 'BEGIN { printf "%.3f\n", (end-start)/1000000000 }' \
  | tee "$result_dir/elapsed-seconds.txt"
echo "results: $result_dir"
