#!/bin/bash

# Build ACE-Step 1.5 for NGC PyTorch 26.02 container (DGX Spark)
#
# Container pre-installed (DO NOT reinstall):
#   CUDA 13.1, Python 3.12.3
#   torch 2.11.0a0+nv26.2, torchvision 0.25.0a0+nv26.2
#   flash_attn 2.7.4+nv26.2, torchao 0.16.0
#   einops 0.8.2, numba 0.63.1, scipy 1.17.0, soundfile 0.13.1

# Clone ACE-Step v0.1.4
cd ~
git clone https://github.com/ace-step/ACE-Step-1.5.git -b v0.1.4
cd ACE-Step-1.5

# FFmpeg libraries for torchcodec
apt update
apt install -y ffmpeg libavcodec60 libavformat60 libavutil58 libavfilter9 libswscale7 libswresample4

# pyproject.toml patches - remove deps already in NGC container or installed separately
sed -i '/"torch==/d' pyproject.toml
sed -i '/"torchvision/d' pyproject.toml
sed -i '/"torchaudio/d' pyproject.toml
sed -i '/flash.attn/d' pyproject.toml
sed -i '/torchcodec/d' pyproject.toml
sed -i '/torchao/d' pyproject.toml
sed -i '/nano-vllm/d' pyproject.toml

pip install torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu130 --no-deps

# Install nano-vllm from local source
pip install ./acestep/third_parts/nano-vllm

# Install ace-step package (editable) with remaining deps
pip install -e .

# torchcodec: pre-built aarch64 cp312 wheel
pip install https://github.com/edp1096/dgx_spark_init/raw/refs/heads/main/compose_yaml/acestep15/builder/torchcodec-0.10.0a0-cp312-cp312-linux_aarch64.whl

# server_name: bind to 0.0.0.0 for container access
sed -i "s/server_name=args.server_name/server_name='0.0.0.0'/g" acestep/acestep_v15_pipeline.py

# enforce_eager: prevent CUDA graph capture failure on GB10
sed -i 's/enforce_eager=enforce_eager,/enforce_eager=True,/g' acestep/llm_inference.py

# Run command:
# acestep --port 7860 --server-name 0.0.0.0 --config_path acestep-v15-turbo --lm_model_path acestep-5Hz-lm-0.6B --init_service true  --offload_to_cpu true
