#!/usr/bin/env bash
set -euo pipefail

# An explicitly empty HF_TOKEN takes precedence over the token persisted in
# the shared Hugging Face cache and makes huggingface_hub emit an invalid
# `Authorization: Bearer ` header.  Leave authentication discovery to the
# cache when Compose did not receive a token.
if [[ -z "${HF_TOKEN:-}" ]]; then
    unset HF_TOKEN
fi

mkdir -p "${TRAINER_DATA_ROOT:-/data}"/{datasets,jobs,output}
mkdir -p "${REGISTERED_LORA_ROOT:-/registered-loras}"

exec uvicorn trainer_api:app --host 0.0.0.0 --port "${API_PORT:-8704}"
