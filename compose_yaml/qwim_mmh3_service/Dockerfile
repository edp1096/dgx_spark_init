FROM nvcr.io/nvidia/pytorch:26.07-py3

ARG COMFYUI_COMMIT=5c460d8172fe30761ff67c0df3d5643bb74e0d70
ENV PIP_ROOT_USER_ACTION=ignore PYTHONUNBUFFERED=1 COMFY_KITCHEN_BACKEND=cuda
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 libx11-6 && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none https://github.com/Comfy-Org/ComfyUI.git /opt/ComfyUI \
    && git -C /opt/ComfyUI checkout --detach "${COMFYUI_COMMIT}" \
    && sed -i '/^torch$/d; /^torchvision$/d; /^torchaudio$/d' /opt/ComfyUI/requirements.txt \
    && pip install --no-cache-dir -r /opt/ComfyUI/requirements.txt gguf==0.19.0 fastapi==0.141.1 httpx==0.28.1 python-multipart==0.0.32 uvicorn==0.52.3


WORKDIR /opt/qwim-mmh3
COPY ClipProj /opt/ComfyUI/custom_nodes/ComfyUI-ClipProj/
COPY api.py worker.py generation_progress.py prepare_models.py settings.json start.sh ./
EXPOSE 8730
ENTRYPOINT ["/bin/bash", "/opt/qwim-mmh3/start.sh"]
