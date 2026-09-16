FROM dgx-sglang-qwen38-fn:sm121-vocab1
# Pin B12X without replacing the qualified serving image's torch/FlashInfer/DSL.
RUN python3 -m pip install --no-deps https://github.com/local-inference-lab/b12x/archive/ce419b52681b7922bb0972d4b58b590a3fd005b2.tar.gz
COPY adapter.py /usr/local/lib/python3.12/dist-packages/sglang_b12x_probe.py
COPY patch.py /tmp/b12x-probe-patch.py
RUN python3 /tmp/b12x-probe-patch.py && rm /tmp/b12x-probe-patch.py
