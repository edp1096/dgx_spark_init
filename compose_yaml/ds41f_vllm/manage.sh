#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
worker=${DSV41_WORKER:-edp1096@192.168.100.60}
remote_dir=/home/edp1096/workspace/dgx_spark_init/compose_yaml/ds41f_vllm
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10)
case "${1:-status}" in
  status)
    docker ps -a --filter name=ds41-stream-0 --format '{{.Names}} {{.Image}} {{.Status}}'
    ssh "${ssh_opts[@]}" "$worker" "docker ps -a --filter name=ds41-stream-1 --format '{{.Names}} {{.Image}} {{.Status}}'"
    curl -s -o /dev/null -w 'API health: %{http_code}\n' --max-time 3 http://127.0.0.1:8010/health
    ;;
  stop)
    if docker inspect ds41-stream-0 >/dev/null 2>&1; then docker stop ds41-stream-0; fi
    ssh "${ssh_opts[@]}" "$worker" 'if docker inspect ds41-stream-1 >/dev/null 2>&1; then docker stop ds41-stream-1; fi'
    ;;
  start)
    ssh "${ssh_opts[@]}" "$worker" true
    # Reapply only this recipe's dedicated rail addresses after host reboot.
    ip link show dev enp1s0f1np1 | grep -q LOWER_UP
    if ! ip -o -4 addr show dev enp1s0f1np1 | grep -Fq '10.200.0.1/24'; then
      docker run --rm --cap-add NET_ADMIN --network host alpine:3.22 ip address add 10.200.0.1/24 dev enp1s0f1np1
    fi
    ssh "${ssh_opts[@]}" "$worker" 'ip link show dev enp1s0f1np1 | grep -q LOWER_UP && { ip -o -4 addr show dev enp1s0f1np1 | grep -Fq "10.200.0.2/24" || docker run --rm --cap-add NET_ADMIN --network host alpine:3.22 ip address add 10.200.0.2/24 dev enp1s0f1np1; }'
    ping -c 1 -W 2 10.200.0.2 >/dev/null
    # Never replace an active rank. Both peers must start from a stopped state.
    [[ $(docker inspect -f '{{.State.Running}}' ds41-stream-0 2>/dev/null || true) != true ]]
    [[ $(ssh "${ssh_opts[@]}" "$worker" "docker inspect -f '{{.State.Running}}' ds41-stream-1 2>/dev/null || true") != true ]]
    if docker inspect ds41-stream-0 >/dev/null 2>&1; then docker rm ds41-stream-0; fi
    ssh "${ssh_opts[@]}" "$worker" 'if docker inspect ds41-stream-1 >/dev/null 2>&1; then docker rm ds41-stream-1; fi'
    remote_env=''
    for key in DSV41_PRELOAD_COUNT VLLM_HOST DSV41_FINAL_DECODER_ROWS DSV41_PROBE_TOKEN DSV41_IMAGE DSV41_MOE_BACKEND DSV41_MAX_MODEL_LEN DSV41_KV_CACHE_BYTES DSV41_PREFIX_CACHE_INTERVAL DSV41_SPEC_TOKENS DSV41_ADAPTIVE_VERIFY DSV41_ROUTED_PIPELINE DSV41_KERNEL_TOKENS DSV41_SHARED_BUFFERS DSV41_SCRATCH_MIB DSV41_MAX_BATCHED_TOKENS DSV41_SLOTS_PER_LAYER DSV41_CACHE_LAYOUT DSV41_READ_THREADS DSV41_EXPERT_GRAPHS DSV41_SLOT_IO DSV41_EXPERT_IO DSV41_PREFETCH_TEST DSV41_MODEL_GRAPHS DSV41_ATTENTION_GRAPHS DSV41_SHORT_CONTEXT_GRAPHS DSV41_ENGRAM_PRESTAGE DSV41_BENCH_CONTROL; do
      if [[ -v "$key" ]]; then
        printf -v assignment '%q=%q ' "$key" "${!key}"
        remote_env+="$assignment"
      fi
    done
    ssh "${ssh_opts[@]}" "$worker" "cd '$remote_dir' && $remote_env ./launch.sh 1"
    if ! ./launch.sh 0; then
      ssh "${ssh_opts[@]}" "$worker" docker stop ds41-stream-1
      exit 1
    fi
    ;;
  logs) docker logs -f --tail 30 ds41-stream-0;;
  *) echo 'Usage: manage.sh start|stop|status|logs' >&2; exit 2;;
esac
