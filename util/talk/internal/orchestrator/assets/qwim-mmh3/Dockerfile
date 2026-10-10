FROM nvcr.io/nvidia/pytorch:26.07-py3

ARG COMFYUI_COMMIT=5c460d8172fe30761ff67c0df3d5643bb74e0d70
ARG SOL_ENGINE_REPO=https://github.com/edp1096/Sana.git
ARG SOL_ENGINE_COMMIT=809638b437f49bdda969ebf568d12b8e91806c98
ENV PIP_ROOT_USER_ACTION=ignore PYTHONUNBUFFERED=1 COMFY_KITCHEN_BACKEND=cuda \
    CUTE_DSL_ARCH=sm_121a SOL_ATTN_STRICT=1 PYTHONPATH=/opt/sol-engine/techniques/sparse_backends
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 libx11-6 && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none https://github.com/Comfy-Org/ComfyUI.git /opt/ComfyUI \
    && git -C /opt/ComfyUI checkout --detach "${COMFYUI_COMMIT}" \
    && sed -i '/^torch$/d; /^torchvision$/d; /^torchaudio$/d' /opt/ComfyUI/requirements.txt \
    && pip install --no-cache-dir -r /opt/ComfyUI/requirements.txt gguf==0.19.0 fastapi==0.141.1 httpx==0.28.1 python-multipart==0.0.32 uvicorn==0.52.3
RUN python -m pip install --no-cache-dir nvidia-cutlass-dsl==4.7.0 && \
    git clone --filter=blob:none --no-checkout "${SOL_ENGINE_REPO}" /opt/sol-engine && \
    git -C /opt/sol-engine fetch --filter=blob:none origin "${SOL_ENGINE_COMMIT}" && \
    git -C /opt/sol-engine checkout --detach "${SOL_ENGINE_COMMIT}"

WORKDIR /opt/qwim-mmh3
COPY ClipProj /opt/ComfyUI/custom_nodes/ComfyUI-ClipProj/
COPY api.py worker.py sol_runtime.py image_inputs.py generation_progress.py prepare_models.py settings.json start.sh ./
EXPOSE 8730
ENTRYPOINT ["/bin/bash", "/opt/qwim-mmh3/start.sh"]
