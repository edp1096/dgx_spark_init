FROM nvcr.io/nvidia/pytorch:26.07-py3

ARG LTX_COMMIT=400fd31054597515f47125691032c04b1c3ee24e
ARG TORCHAUDIO_COMMIT=4e3e282b0e23a0b9133abc8f719e2fa39be2a6e3

ENV PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_CUDA_ARCH_LIST=12.1a \
    MAX_JOBS=2

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake ffmpeg git ninja-build && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Lightricks/LTX-2.git /opt/LTX-2 && \
    git -C /opt/LTX-2 checkout "${LTX_COMMIT}" && \
    python -c "from pathlib import Path; p=Path('/opt/LTX-2/packages/ltx-kernels/setup.py'); s=p.read_text(); s=s.replace('_NVFP4_ARCHES = [\"100a\", \"120a\"]', '_NVFP4_ARCHES = [\"100a\", \"120a\", \"121a\"]'); s=s.replace('elif tok.startswith(\"12.0\"):\n            sel.append(\"120a\")', 'elif tok.startswith(\"12.0\"):\n            sel.append(\"120a\")\n        elif tok.startswith(\"12.1\"):\n            sel.append(\"121a\")'); p.write_text(s)"

RUN python -m pip install --no-cache-dir \
      fastapi==0.116.1 \
      huggingface-hub==0.36.0 \
      python-multipart==0.0.20 \
      uvicorn==0.35.0 && \
    python -m pip install --no-cache-dir -e /opt/LTX-2/packages/ltx-core -e /opt/LTX-2/packages/ltx-pipelines && \
    python -m pip install --no-cache-dir --no-build-isolation /opt/LTX-2/packages/ltx-kernels

# PyPI TorchAudio is built against CUDA 13.0 while the NGC 26.07 PyTorch image uses
# CUDA 13.3. Build against the image's exact PyTorch/CUDA ABI instead.
RUN python -m pip uninstall -y torchaudio && \
    git clone https://github.com/pytorch/audio.git /tmp/audio && \
    git -C /tmp/audio checkout "${TORCHAUDIO_COMMIT}" && \
    BUILD_SOX=0 USE_FFMPEG=0 USE_CUDA=1 \
      python -m pip install --no-cache-dir --no-deps --no-build-isolation /tmp/audio && \
    rm -rf /tmp/audio

# Community ComfyUI Gemma packs embed all tensors and tokenizer assets but may
# omit the native LTX `gemma_config` metadata key. Keep their weights/assets and
# source only the architecture config from the official compatible encoder.
COPY patches/comfy-gemma-config-fallback.patch /tmp/comfy-gemma-config-fallback.patch
RUN git -C /opt/LTX-2 apply /tmp/comfy-gemma-config-fallback.patch && \
    rm -f /tmp/comfy-gemma-config-fallback.patch

WORKDIR /opt/ltx-api
COPY api.py prepare_models.py start.sh ./
RUN chmod +x start.sh

EXPOSE 8695
ENTRYPOINT ["/opt/ltx-api/start.sh"]
