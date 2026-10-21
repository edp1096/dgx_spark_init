#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Validate prepared assets; never download the retired checkpoint.
exec bash manage.sh validate
