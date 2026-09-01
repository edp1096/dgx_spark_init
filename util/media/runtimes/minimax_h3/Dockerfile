FROM dgx-ltx2.5-nvfp4-api:latest AS torchaudio_source

FROM dgx-krea2-turbo-nvfp4:latest

ARG TOOBUSY_COMMIT=8a1f3858deaac51187b08800c5faa23ee84b63c9

# Both local images use the same NGC 26.07 PyTorch/CUDA ABI. Reuse the
# TorchAudio build already verified by the LTX runtime instead of compiling it
# again for every H3 image build.
COPY --from=torchaudio_source \
  /usr/local/lib/python3.12/dist-packages/torchaudio \
  /usr/local/lib/python3.12/dist-packages/torchaudio
COPY --from=torchaudio_source \
  /usr/local/lib/python3.12/dist-packages/torchaudio-2.11.0a0+4e3e282.dist-info \
  /usr/local/lib/python3.12/dist-packages/torchaudio-2.11.0a0+4e3e282.dist-info

RUN python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__)" && \
    git clone https://github.com/nicekriss/toobusy.git /opt/toobusy-source && \
    git -C /opt/toobusy-source checkout "${TOOBUSY_COMMIT}" && \
    mkdir -p /opt/ComfyUI/custom_nodes/toobusy_h3 && \
    cp /opt/toobusy-source/minimax_h3_image_latent_node/minimax_h3_image_latent.py \
      /opt/ComfyUI/custom_nodes/toobusy_h3/ && \
    cp /opt/toobusy-source/minimax_h3_semantic_reference_node/minimax_h3_semantic_reference.py \
      /opt/ComfyUI/custom_nodes/toobusy_h3/ && \
    cp /opt/toobusy-source/minimax_h3_optional_reference_node/minimax_h3_optional_reference.py \
      /opt/ComfyUI/custom_nodes/toobusy_h3/

WORKDIR /opt/minimax-h3
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir fastapi uvicorn python-multipart httpx
COPY prepare_models.py start.sh api.py ./
COPY h3_nodes_init.py /opt/ComfyUI/custom_nodes/toobusy_h3/__init__.py
RUN chmod +x start.sh

EXPOSE 8190
ENTRYPOINT ["/opt/minimax-h3/start.sh"]
