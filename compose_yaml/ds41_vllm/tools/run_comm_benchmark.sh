#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
rank=${1:?rank required}
case "$rank" in 0) rail=10.200.0.1;;1) rail=10.200.0.2;;*) exit 2;;esac
docker run --rm --gpus all --name "ds41-comm-test-$rank" \
  --network host --ipc host --memory 12g --memory-swap 12g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband:/dev/infiniband \
  -v "$PWD:/opt/ds41:ro" -v "$HOME/.cache/ds41-stream:/cache" \
  -e PYTHONPATH=/opt/ds41 -e XDG_CACHE_HOME=/cache \
  -e VLLM_HOST_IP="$rail" -e VLLM_PLUGINS= -e OMP_NUM_THREADS=1 \
  -e NCCL_NET=IB -e NCCL_IB_HCA=rocep1s0f1 -e NCCL_IB_GID_INDEX=3 \
  -e B12X_ROCE_HCA=rocep1s0f1 -e B12X_ROCE_GID_INDEX=3 \
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e GLOO_SOCKET_IFNAME=enp1s0f1np1 \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ADDR_RANGE=10.200.0.0/24 -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 \
  --entrypoint timeout dgx-ds41-stream:b12x8-dev 240s \
  torchrun --nnodes=2 --nproc-per-node=1 --node-rank="$rank" \
  --master-addr=10.200.0.1 --master-port=29653 /opt/ds41/tools/benchmark_tp_collectives.py \
  --output /cache/tp-collectives-comparison.json
