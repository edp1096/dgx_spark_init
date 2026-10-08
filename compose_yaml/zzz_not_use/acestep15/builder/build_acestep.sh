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

# ko.json: Korean i18n
curl -sL https://raw.githubusercontent.com/edp1096/dgx_spark_init/refs/heads/main/compose_yaml/acestep15/builder/ko.json -o acestep/ui/gradio/i18n/ko.json

# server_name: bind to 0.0.0.0 for container access
sed -i "s/server_name=args.server_name/server_name='0.0.0.0'/g" acestep/acestep_v15_pipeline.py

# enforce_eager: prevent CUDA graph capture failure on GB10
sed -i 's/enforce_eager=enforce_eager,/enforce_eager=True,/g' acestep/llm_inference.py

# ACESTEP_SKIP_VRAM_CHECK: use CUDA allocator stats instead of mem_get_info on unified memory (GB10)
# mem_get_info reports system RAM usage as GPU usage, giving misleadingly low free values
# Patch get_effective_free_vram_gb()
sed -i 's/device_free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)/device_free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)\n            if os.environ.get("ACESTEP_SKIP_VRAM_CHECK"):\n                total_bytes = torch.cuda.get_device_properties(device_index).total_memory\n                device_free_bytes = total_bytes - torch.cuda.memory_allocated(device_index)/' acestep/gpu_config.py
# Patch get_lm_gpu_memory_ratio() - also calls mem_get_info directly
sed -i 's/free_bytes, total_bytes = torch.cuda.mem_get_info()/free_bytes, total_bytes = torch.cuda.mem_get_info()\n            if os.environ.get("ACESTEP_SKIP_VRAM_CHECK"):\n                total_bytes = torch.cuda.get_device_properties(0).total_memory\n                free_bytes = total_bytes - torch.cuda.memory_allocated(0)/' acestep/gpu_config.py

# Run command:
# acestep --port 7860 --server-name 0.0.0.0 --config_path acestep-v15-turbo --lm_model_path acestep-5Hz-lm-0.6B --init_service true  --offload_to_cpu true
