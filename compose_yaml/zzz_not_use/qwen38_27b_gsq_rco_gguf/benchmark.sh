#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
pause_container=""
resume_url=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pause-container) pause_container=${2:?Missing container}; shift 2 ;;
    --resume-url) resume_url=${2:?Missing API URL}; shift 2 ;;
    *) echo 'Usage: ./benchmark.sh [--pause-container NAME] [--resume-url http://host:port/v1/models]' >&2; exit 2 ;;
  esac
done
[[ "$pause_container" != qwen38-27b-gsq-rco-gguf ]] || exit 2
if [[ -n "$pause_container" ]]; then
  docker inspect --format '{{.State.Status}}' "$pause_container" >/dev/null
  if [[ -z "$resume_url" ]]; then
    binding=$(docker port "$pause_container" | awk '$1 ~ /^(30000|8000|8080|8888)\/tcp$/ {print $NF; exit}')
    if [[ -z "$binding" ]]; then
      echo 'Cannot determine the old LLM API port. Provide --resume-url before pausing it.' >&2
      exit 2
    fi
    resume_url="http://127.0.0.1:${binding##*:}/v1/models"
  fi
fi
paused=0
cleanup() {
  status=$?
  trap - EXIT INT TERM
  ./manage.sh stop || true
  if [[ $paused == 1 ]]; then
    docker start "$pause_container" >/dev/null || status=1
    docker run --rm --network host -v "$PWD:/workspace:ro" -w /workspace \
      qwen38-27b-gsq-rco-gguf-tools:e71b805 python3 wait_api.py "$resume_url" || status=1
    if [[ $status == 0 ]]; then echo "Restored API for $pause_container"; fi
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ -n "$pause_container" && $(docker inspect --format '{{.State.Running}}' "$pause_container") == true ]]; then
  paused=1
  docker stop --timeout 45 "$pause_container"
fi
for n in 0 1 2 3; do
  if [[ $n == 0 ]]; then label=serial; kind=none; draft=2; else label="mtp$n"; kind=draft-mtp; draft=$n; fi
  echo "Starting benchmark: $label"
  SPEC_TYPE="$kind" MTP_TOKENS="$draft" ./manage.sh start
  docker exec qwen38-27b-gsq-rco-gguf python3 -c '
import time,urllib.request
for i in range(240):
 try:
  with urllib.request.urlopen("http://127.0.0.1:8080/health",timeout=3) as response:
   if response.status==200:break
 except Exception:time.sleep(2)
else:raise SystemExit("Model readiness timed out")
'
  binding=$(docker compose port server 8080)
  endpoint="http://127.0.0.1:${binding##*:}"
  if [[ $n == 2 ]]; then
    ./manage.sh validate --url "$endpoint" --label "$label" --quality
  else
    ./manage.sh validate --url "$endpoint" --label "$label"
  fi
  docker compose logs --no-color server > "data/reports/$label/server.log"
  ./manage.sh stop
 done
