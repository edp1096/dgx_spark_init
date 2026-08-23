FROM nvcr.io/nvidia/pytorch:26.07-py3

ARG AI_TOOLKIT_COMMIT=27a03a91f23eb1b757d5ec2e80ee3e129cfcc350

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update \
    && apt-get install --no-install-recommends -y git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ai-toolkit's DGX setup pins this mutually compatible CUDA 13 trio. The NGC
# 26.07 image otherwise carries a CUDA 13.3 alpha torch without torchaudio.
RUN pip install --no-cache-dir \
    torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu130

# requirements_base still pins a TorchCodec release that never shipped an
# ARM64 wheel. ai-toolkit's current environment manager uses 0.15 with
# torch 2.13 on DGX/Grace, so mirror that resolution in this CLI image.
RUN git clone https://github.com/ostris/ai-toolkit.git /opt/ai-toolkit \
    && cd /opt/ai-toolkit \
    && git checkout "${AI_TOOLKIT_COMMIT}" \
    && sed -i 's/^torchcodec==0\.9\.1$/torchcodec==0.15.0/' requirements_base.txt \
    && pip install --no-cache-dir -r dgx_requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn python-multipart

WORKDIR /opt/trainer-api
COPY trainer_api.py start.sh ./
RUN chmod +x start.sh

EXPOSE 8704
ENTRYPOINT ["/opt/trainer-api/start.sh"]
