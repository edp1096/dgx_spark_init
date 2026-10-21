ARG BASE_IMAGE=nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
FROM ${BASE_IMAGE} AS build
ARG ZIG_VERSION=0.17.0
ARG ZIG_SHA256=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8
RUN curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/${ZIG_VERSION}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz" && \
    echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - && \
    mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz && \
    /opt/zig/zig version
ARG TF_REPO=https://github.com/ashhart/TensorFold.git
ARG TF_REF=db281878ddb836fd0df510d8771ecb7e0fe47d26
RUN git init -q /tensorfold && cd /tensorfold && git remote add origin "${TF_REPO}" && \
    git fetch -q --depth 1 origin "${TF_REF}" && git checkout -q --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${TF_REF}"
COPY patches /opt/tf-patches
RUN cd /tensorfold && \
    for p in /opt/tf-patches/*.patch; do [ -e "$p" ] || continue; echo "applying $p"; git apply --check "$p" && git apply "$p" || exit 1; done
COPY draft-vocab-korean.txt /tensorfold/zig/src/families/flashnext/cuda_draft_vocab.txt
RUN cd /tensorfold && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=safe --prefix /opt/tensorfold \
      -j2 fatbins install native && \
    test -x /opt/tensorfold/native/bin/tensorfold-native
RUN mkdir -p /opt/tensorfold/share/doc/tensorfold && cd /tensorfold && \
    for f in LICENSE LICENSE.md LICENSES NOTICE THIRD_PARTY_NOTICES.md; do \
      if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/ || exit 1; fi; done
ARG KERNEL_SOURCE_MTIME=1791318675
RUN cd /tensorfold && python -B -c 'import json, os, sys; t = int(sys.argv[1]); \
      [os.utime(os.path.join("src", f), (t, t)) for f in sorted({k["source"]["file"] for k in \
       json.load(open("zig/tests/cuda/flashnext/kernels.json"))["kernels"]})]' "${KERNEL_SOURCE_MTIME}" && \
    PYTHONPATH=/tensorfold/src python -B tools/zig/flashnext_aot.py build \
      --spec zig/tests/cuda/flashnext/kernels.json --jit zig/tests/cuda/flashnext/jit.json --tp 1 \
      --out /opt/tensorfold/share/tensorfold/cuda/sm121 > /tmp/aot.log || true && \
    cat /tmp/aot.log && \
    grep -q 'kernels ->' /tmp/aot.log && \
    bad=$(grep PROBLEM /tmp/aot.log | grep -v 'cubin sha256' || true) && \
    test -z "$bad" && \
    test -f /opt/tensorfold/share/tensorfold/cuda/sm121/aot.json

FROM lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1
COPY --from=build /opt/tensorfold /opt/tensorfold
COPY --from=build /tensorfold/src/tensorfold /opt/tensorfold/python/tensorfold
RUN ln -s /opt/tensorfold/native/bin/tensorfold-native /usr/local/bin/tensorfold-native && \
    pip install --no-cache-dir "huggingface_hub==2.2.0" "safetensors==0.8.0" "jinja2==3.1.6" "Pillow==12.3.0" && \
    pip install --no-cache-dir --no-deps "transformers==5.19.0" "tokenizers==0.23.3" "av==19.0.1"
COPY tf_structure.py structure_bridge.cc /opt/tf-structure/
RUN g++ -O2 -shared -fPIC /opt/tf-structure/structure_bridge.cc $(python3-config --includes --embed --ldflags) -ldl -o /opt/tf-structure/libtf_structure.so && \
    python3 -c 'import importlib.metadata; assert importlib.metadata.version("xgrammar") == "0.2.1"'
ENV TENSORFOLD_STRUCTURE_BRIDGE=/opt/tf-structure/libtf_structure.so TENSORFOLD_STRUCTURE_MODEL=/model
ARG PATCHES_HASH
LABEL tf.patches=${PATCHES_HASH} tf.kernels=present
ENV HF_HOME=/root/.cache/huggingface TENSORFOLD_CUDA_KERNELS=/opt/tensorfold/share/tensorfold/cuda/sm121 \
    CUDA_CACHE_PATH=/cache/nv \
    TENSORFOLD_VISION_PYTHONPATH=/opt/tensorfold/python
WORKDIR /workspace
ENTRYPOINT ["/opt/tensorfold/native/bin/tensorfold-native"]
