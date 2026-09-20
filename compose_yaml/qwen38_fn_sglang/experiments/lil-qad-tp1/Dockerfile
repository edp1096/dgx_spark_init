FROM dgx-sglang-qwen38-fn:sm121-vocab1
COPY qad/nvfp4_ple.py qad/ple_embedding.py /opt/qad-tp1/
COPY qad/patch_sglang.py /opt/qad-tp1/patch_sglang.py
ENV PYTHONPATH=/opt/qad-tp1
RUN python3 /opt/qad-tp1/patch_sglang.py
