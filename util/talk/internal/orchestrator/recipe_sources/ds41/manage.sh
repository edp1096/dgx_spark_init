#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
set -a
source env.sample
[[ ! -f .env ]] || source .env
set +a
exec bash runtime.sh "${@:-status}"
