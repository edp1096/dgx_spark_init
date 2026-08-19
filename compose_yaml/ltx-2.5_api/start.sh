#!/usr/bin/env bash
set -euo pipefail

python /opt/ltx-api/prepare_models.py
exec uvicorn api:app --host 0.0.0.0 --port "${API_PORT:-8695}"
