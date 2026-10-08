#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
output=${1:-runs/hancom}
mkdir -p "$output"
output=$(cd "$output" && pwd)
docker build -f validation/Dockerfile.hancom -t sparktalk-hancom-validation .
docker run --rm --init --user "$(id -u):$(id -g)" -v "$PWD/validation:/validation:ro" -v "$output:/out" sparktalk-hancom-validation sh -c 'node /app/renderer.test.mjs && node /validation/hancom-service.mjs /out && /opt/verify/bin/python /validation/verify-independent.py /out'
