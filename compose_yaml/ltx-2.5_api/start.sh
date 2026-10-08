#!/usr/bin/env bash
set -euo pipefail

exec uvicorn api:app --host 0.0.0.0 --port "${API_PORT:-8695}"
