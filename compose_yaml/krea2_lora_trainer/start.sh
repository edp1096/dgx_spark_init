#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${TRAINER_DATA_ROOT:-/data}"/{datasets,jobs,output}
mkdir -p "${REGISTERED_LORA_ROOT:-/registered-loras}"

exec uvicorn trainer_api:app --host 0.0.0.0 --port "${API_PORT:-8704}"
