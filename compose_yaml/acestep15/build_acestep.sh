#!/bin/bash

# Clone ACE-Step v0.1.4
cd ~
git clone https://github.com/ace-step/ACE-Step-1.5.git -b v0.1.4
cd ACE-Step-1.5

# FFmpeg libraries for torchcodec
apt update
apt install -y ffmpeg libavcodec60 libavformat60 libavutil58 libavfilter9 libswscale7 libswresample4

# pyproject.toml patches
# - Remove flash-attn dependency (use system FA2 via symlink)
sed -i '/flash.attn/d' pyproject.toml
# - Remove torchcodec dependency (install pre-built aarch64 wheel separately)
sed -i '/torchcodec/d' pyproject.toml
# - Skip flash-attn resolution entirely (transitive dependency)
sed -i '/^\[tool\.uv\]/a override-dependencies = ["flash-attn ; sys_platform == '\''never'\''"]' pyproject.toml

rm -f uv.lock
uv venv --system-site-packages
uv sync

# torchcodec: install pre-built aarch64 cp312 wheel into venv
source .venv/bin/activate
uv pip install https://github.com/edp1096/dgx_spark_init/raw/refs/heads/main/compose_yaml/acestep15/torchcodec-0.10.0a0-cp312-cp312-linux_aarch64.whl
deactivate

# flash-attn: symlink system FA2 (26.02 built-in) into venv
ln -sf /usr/local/lib/python3.12/dist-packages/flash_attn .venv/lib/python3.12/site-packages/flash_attn
ln -sf /usr/local/lib/python3.12/dist-packages/flash_attn_2_cuda.cpython-312-aarch64-linux-gnu.so .venv/lib/python3.12/site-packages/
ln -sf /usr/local/lib/python3.12/dist-packages/flash_attn-*.dist-info .venv/lib/python3.12/site-packages/

# server_name: bind to 0.0.0.0 for container access
sed -i "s/server_name=args.server_name/server_name='0.0.0.0'/g" acestep/acestep_v15_pipeline.py

# enforce_eager: prevent CUDA graph capture failure on GB10
sed -i 's/enforce_eager=enforce_eager,/enforce_eager=True,/g' acestep/llm_inference.py
