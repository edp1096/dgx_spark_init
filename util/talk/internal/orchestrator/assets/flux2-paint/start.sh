#!/usr/bin/env bash
set -euo pipefail
python /opt/nvfp4-api/prepare_lora.py
/opt/rembg-venv/bin/python /opt/nvfp4-api/rembg_worker.py --prepare
exec /opt/nvfp4-api/base_start.sh
