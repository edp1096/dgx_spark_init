FROM dgx-sglang-qwen38-fn:sm121-b12x-head-v1
COPY tp_vocab.py /sgl-workspace/sglang/python/sglang/srt/speculative/tp_vocab.py
COPY install.py /tmp/install-tp-vocab.py
RUN python3 /tmp/install-tp-vocab.py
