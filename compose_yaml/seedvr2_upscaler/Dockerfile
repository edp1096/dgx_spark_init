FROM dgx-flux2-klein-nvfp4:4b

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git \
      /opt/ComfyUI/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler \
    && cd /opt/ComfyUI/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler \
    && git checkout 4490bd1f482e026674543386bb2a4d176da245b9 \
    && pip install --no-cache-dir 'diffusers>=0.33.1' omegaconf peft rotary-embedding-torch gguf matplotlib opencv-python-headless

WORKDIR /opt/seedvr2-api
COPY api.py start.sh ./
RUN chmod +x start.sh

EXPOSE 8698
ENTRYPOINT ["/opt/seedvr2-api/start.sh"]
