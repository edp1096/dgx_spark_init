FROM dgx-flux2-klein-nvfp4:4b
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
