FROM dgx-flux2-klein-nvfp4:4b

RUN git clone https://github.com/Gourieff/ComfyUI-ReActor.git \
      /opt/ComfyUI/custom_nodes/ComfyUI-ReActor \
    && cd /opt/ComfyUI/custom_nodes/ComfyUI-ReActor \
    && git checkout 6ad6b35a4df250d14cb2abf0808c9ffedf59f747 \
    && python -m pip install --no-cache-dir \
      onnxruntime-gpu==1.29.0 \
      onnx>=1.14.0 \
      opencv-python-headless>=4.7.0.72 \
      albumentations>=1.4.16 \
      segment-anything \
      ultralytics \
      aiohttp==3.12.15 \
      fastapi==0.116.1 \
      httpx==0.28.1 \
      huggingface-hub==1.24.0 \
      python-multipart==0.0.20 \
      uvicorn==0.35.0

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libxcb1 libx11-6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/reactor-api
COPY api.py prepare_models.py start.sh ./
RUN chmod +x start.sh

EXPOSE 8706
ENTRYPOINT ["/opt/reactor-api/start.sh"]
