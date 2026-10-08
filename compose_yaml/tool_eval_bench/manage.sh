#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
set -a
source ./upstream.env
set +a
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
# Detached `compose run` jobs intentionally coexist with short-lived history/CLI jobs.
export COMPOSE_IGNORE_ORPHANS=true
action="${1:-help}"
if (( $# )); then shift; fi
usage() {
 cat <<'EOF'
Usage: ./manage.sh COMMAND [tool-eval-bench options]
  setup / image  Clone the pinned source and build the image (no model download).
  probe          Check the target model API.
  run [OPTIONS]  Foreground evaluation; default: 88 scenarios, seed 42, 3 trials.
  start [OPTIONS] Same evaluation in background (one managed job at a time).
  logs           Follow the background job's progress (Ctrl-C only leaves logs).
  status         Show background job state and exit code.
  stop           Interrupt background job; completed scenarios remain in SQLite.
  history        List saved runs.
  reports        List model/score/time and create readable filenames in runs/named/.
  compare        Select two reports from a numbered list and generate HTML.
  compare A B    Compare two saved run IDs.
  compare A.md B.md  Generate HTML from two Markdown reports.
  cli ARGS...    Pass any upstream command, e.g. resume RUN_ID or run --dry-run.
  validate       Validate Compose without printing configuration/secrets.
Configuration: .env (created from env.sample). Output: runs/, data/, cache/.
EOF
}
case "$action" in help|-h|--help) usage; exit 0;;
 setup|image|probe|run|start|logs|status|stop|history|reports|compare|cli|validate) ;;
 *) usage >&2; exit 2;; esac
if [[ ! -f .env ]]; then
 (umask 077; cp env.sample .env)
fi
mkdir -p runs data cache
compose() { docker compose "$@"; }
prepare() {
 if [[ ! -d upstream ]]; then
  git clone https://github.com/SeraphimSerapis/tool-eval-bench.git upstream
 fi
 [[ -d upstream/.git ]] || { echo 'upstream is not a standalone Git clone' >&2; exit 1; }
 [[ -z "$(git -C upstream status --porcelain)" ]] || {
  echo 'upstream has local changes; preserve them before rebuilding.' >&2; exit 1;
 }
 if ! git -C upstream cat-file -e "$BENCH_REVISION^{commit}" 2>/dev/null; then
  git -C upstream fetch origin "$BENCH_REVISION"
 fi
 git -C upstream checkout --detach "$BENCH_REVISION"
 compose build bench
}
job_name=tool-eval-bench-job
case "$action" in
 setup|image) prepare;;
 validate) compose config --quiet;;
 probe|history) compose run --rm bench "$action" "$@";;
 reports) python3 reports.py;;
 compare)
  if (( $# == 0 )); then python3 reports.py compare
  elif [[ "$1" == *.md || "$1" == */* || "${2:-}" == *.md || "${2:-}" == */* ]]; then
   python3 reports.py compare-files "$@"
  else compose run --rm bench compare "$@"; fi;;
 cli) compose run --rm bench "$@";;
 run|start)
  if (( $# == 0 )); then set -- --hardmode --seed 42 --trials 3; fi
  if [[ "$action" == run ]]; then
   compose run --rm bench run "$@"
  else
   if docker container inspect "$job_name" >/dev/null 2>&1; then
    if [[ "$(docker inspect -f '{{.State.Running}}' "$job_name")" == true ]]; then
     echo 'A benchmark is already running. Use logs/status/stop.' >&2; exit 1
    fi
    docker rm "$job_name" >/dev/null
   fi
   compose run -d --name "$job_name" bench run "$@"
   sleep 2
   if [[ "$(docker inspect -f '{{.State.Running}}' "$job_name")" != true ]]; then
    code="$(docker inspect -f '{{.State.ExitCode}}' "$job_name")"
    docker logs --tail 8 "$job_name"
    echo "Benchmark exited (code=$code). See ./manage.sh logs." >&2
    exit "$code"
   fi
   echo 'Started. Use ./manage.sh logs or ./manage.sh status.'
  fi;;
 logs) docker logs -f "$job_name";;
 status)
  python3 reports.py
  if docker container inspect "$job_name" >/dev/null 2>&1; then
   docker inspect -f 'status={{.State.Status}} exit={{.State.ExitCode}} started={{.State.StartedAt}} finished={{.State.FinishedAt}}' "$job_name"
   if [[ "$(docker inspect -f '{{.State.Status}}:{{.State.ExitCode}}' "$job_name")" == exited:[1-9]* ]]; then
    docker logs --tail 4 "$job_name"
   fi
  else echo 'No background benchmark job.'; fi;;
 stop) docker stop --signal SIGINT --timeout 30 "$job_name";;
esac
