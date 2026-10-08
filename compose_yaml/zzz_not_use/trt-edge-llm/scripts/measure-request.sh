#!/usr/bin/env bash
set -euo pipefail

label=${1:?usage: measure-request.sh LABEL REQUEST_JSON}
request_file=${2:?usage: measure-request.sh LABEL REQUEST_JSON}
endpoint=${TRT_EDGE_ENDPOINT:-http://127.0.0.1:8696}
sample_interval=${TRT_EDGE_SAMPLE_INTERVAL:-0.20}
result_dir=${TRT_EDGE_RESULT_DIR:-./results}

mkdir -p "${result_dir}"
stamp=$(date +%Y%m%d-%H%M%S)
metrics_file="${result_dir}/${stamp}-${label}-memory.csv"
response_file="${result_dir}/${stamp}-${label}-response.json"
summary_file="${result_dir}/${stamp}-${label}-summary.json"

unified_used_mib() {
  awk '
    /MemTotal:/ { total = $2 }
    /MemAvailable:/ { available = $2 }
    END { printf "%.1f", (total - available) / 1024 }
  ' /proc/meminfo
}

gpu_process_mib() {
  nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk '{ total += $1 } END { printf "%.1f", total + 0 }'
}

baseline_unified=$(unified_used_mib)
baseline_gpu=$(gpu_process_mib)
printf 'epoch_ms,unified_used_mib,gpu_process_mib\n' >"${metrics_file}"

monitor() {
  while true; do
    now_ms=$(date +%s%3N)
    printf '%s,%s,%s\n' "${now_ms}" "$(unified_used_mib)" "$(gpu_process_mib)" \
      >>"${metrics_file}"
    sleep "${sample_interval}"
  done
}

monitor &
monitor_pid=$!
cleanup() {
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
}
trap cleanup EXIT

start_ns=$(date +%s%N)
http_code=$(curl -sS -o "${response_file}" -w '%{http_code}' \
  "${endpoint}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary "@${request_file}")
end_ns=$(date +%s%N)
cleanup
trap - EXIT

duration_s=$(awk -v start="${start_ns}" -v end="${end_ns}" \
  'BEGIN { printf "%.3f", (end - start) / 1000000000 }')
peak_unified=$(awk -F, 'NR > 1 && $2 > max { max = $2 } END { printf "%.1f", max }' "${metrics_file}")
peak_gpu=$(awk -F, 'NR > 1 && $3 > max { max = $3 } END { printf "%.1f", max }' "${metrics_file}")
completion_tokens=$(jq -r '.usage.completion_tokens // 0' "${response_file}")
prompt_tokens=$(jq -r '.usage.prompt_tokens // 0' "${response_file}")
tokens_per_second=$(awk -v tokens="${completion_tokens}" -v seconds="${duration_s}" \
  'BEGIN { if (seconds > 0) printf "%.2f", tokens / seconds; else print "0.00" }')

jq -n \
  --arg label "${label}" \
  --argjson http_code "${http_code}" \
  --argjson duration_s "${duration_s}" \
  --argjson prompt_tokens "${prompt_tokens}" \
  --argjson completion_tokens "${completion_tokens}" \
  --argjson tokens_per_second "${tokens_per_second}" \
  --argjson baseline_unified_mib "${baseline_unified}" \
  --argjson peak_unified_mib "${peak_unified}" \
  --argjson baseline_gpu_process_mib "${baseline_gpu}" \
  --argjson peak_gpu_process_mib "${peak_gpu}" \
  '{
    label: $label,
    http_code: $http_code,
    duration_s: $duration_s,
    prompt_tokens: $prompt_tokens,
    completion_tokens: $completion_tokens,
    completion_tokens_per_second_including_prefill: $tokens_per_second,
    unified_memory: {
      baseline_mib: $baseline_unified_mib,
      peak_mib: $peak_unified_mib,
      request_delta_mib: ($peak_unified_mib - $baseline_unified_mib)
    },
    nvidia_process_memory: {
      baseline_mib: $baseline_gpu_process_mib,
      peak_mib: $peak_gpu_process_mib,
      request_delta_mib: ($peak_gpu_process_mib - $baseline_gpu_process_mib)
    }
  }' | tee "${summary_file}"
