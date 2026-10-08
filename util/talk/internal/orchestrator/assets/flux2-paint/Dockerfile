FROM nvcr.io/nvidia/pytorch:26.07-py3 AS flux-base

ARG COMFYUI_COMMIT=cc0fc21fea7a6a82f568362b15b7fbd713b419c1
ARG COMFY_KITCHEN_COMMIT=ff83be31dd7f397975a0b8f2c0e86cd50ad78dd5

ENV PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=2 \
    MAX_JOBS=2 \
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
COPY base/api.py base/prepare_models.py base/quantize_uncensored_text_encoder.py base/start.sh ./
RUN chmod +x start.sh

EXPOSE 8691
ENTRYPOINT ["/opt/nvfp4-api/start.sh"]

FROM flux-base
ARG LANPAINT_COMMIT=32cf848e93971da380d868936e007f5611218bee
COPY extract_inpaint.py /tmp/extract_inpaint.py
RUN git clone https://github.com/scraed/LanPaint.git /tmp/lanpaint-upstream && \
    git -C /tmp/lanpaint-upstream checkout "${LANPAINT_COMMIT}" && \
    python /tmp/extract_inpaint.py /tmp/lanpaint-upstream /opt/ComfyUI/custom_nodes/sparktalk_inpaint && \
    rm -rf /tmp/lanpaint-upstream /tmp/extract_inpaint.py && \
    mv /opt/nvfp4-api/api.py /opt/nvfp4-api/base_api.py && \
    mv /opt/nvfp4-api/start.sh /opt/nvfp4-api/base_start.sh
RUN python -m venv /opt/rembg-venv && \
    /opt/rembg-venv/bin/pip install --no-cache-dir "rembg[cpu]==2.0.84" "onnxruntime==1.29.0"
COPY api.py prepare_lora.py rembg_worker.py start.sh /opt/nvfp4-api/
RUN chmod +x /opt/nvfp4-api/start.sh
# On GB10 unified memory, pinned staging reduces MemAvailable by about 6 GiB.
# Keep the model/cache policy and numerical path unchanged.
RUN sed -i 's/--disable-auto-launch/--disable-auto-launch --disable-pinned-memory --cache-classic/' /opt/nvfp4-api/base_start.sh
COPY patch_vae_tiling.py /tmp/patch_vae_tiling.py
RUN python /tmp/patch_vae_tiling.py && rm /tmp/patch_vae_tiling.py

COPY phased/ /opt/ComfyUI/custom_nodes/sparktalk_phased/

ENV SPARKTALK_FLUX_PHASED=1
