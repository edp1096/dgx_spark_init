FROM nvcr.io/nvidia/pytorch:26.07-py3

ARG COMFYUI_COMMIT=cc0fc21fea7a6a82f568362b15b7fbd713b419c1
ARG COMFY_KITCHEN_COMMIT=ff83be31dd7f397975a0b8f2c0e86cd50ad78dd5

ENV PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1 \
    COMFY_CUDA_ARCHS=121

RUN apt-get update && \
    apt-get install -y --no-install-recommends cmake git ninja-build && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "nanobind>=2.0.0" "cmake>=3.26" ninja

RUN git clone --recurse-submodules https://github.com/Comfy-Org/comfy-kitchen.git /tmp/comfy-kitchen && \
    git -C /tmp/comfy-kitchen checkout "${COMFY_KITCHEN_COMMIT}" && \
    pip install --no-cache-dir --no-build-isolation /tmp/comfy-kitchen && \
    rm -rf /tmp/comfy-kitchen

RUN git clone https://github.com/Comfy-Org/ComfyUI.git /opt/ComfyUI && \
    git -C /opt/ComfyUI checkout "${COMFYUI_COMMIT}" && \
    sed -i '/^torch$/d; /^torchvision$/d; /^torchaudio$/d; /^comfy-kitchen/d' /opt/ComfyUI/requirements.txt && \
    sed -i '/^import torchaudio$/d' /opt/ComfyUI/comfy/ldm/lightricks/vae/audio_vae.py && \
    sed -i '/^import torchaudio\.functional as AF$/d' /opt/ComfyUI/comfy/text_encoders/gemma4.py && \
    pip install --no-cache-dir -r /opt/ComfyUI/requirements.txt

RUN pip install --no-cache-dir fastapi huggingface-hub httpx python-multipart uvicorn

WORKDIR /opt/nvfp4-api
COPY api.py prepare_models.py quantize_uncensored_text_encoder.py start.sh ./
RUN chmod +x start.sh

EXPOSE 8691
ENTRYPOINT ["/opt/nvfp4-api/start.sh"]
