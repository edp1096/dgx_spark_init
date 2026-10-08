#!/usr/bin/env bash
set -euo pipefail
export MOE_RECIPE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$MOE_RECIPE_DIR/runtime/manage.sh" "$@"
