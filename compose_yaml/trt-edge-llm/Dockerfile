ARG TRT_EDGE_RUNTIME_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
ARG TRT_EDGE_CUDA_VERSION=13.3

# Export is CPU-only for an already quantized checkpoint.  The newer PyTorch
# image matches the Python frontend versions used by Edge-LLM 0.10.
FROM nvcr.io/nvidia/pytorch:26.07-py3 AS exporter

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git /opt/TensorRT-Edge-LLM \
    && cd /opt/TensorRT-Edge-LLM \
    && git checkout 71dd1bae032e70771265917ec74d3ff4cad07a10 \
    && git submodule update --init --recursive

WORKDIR /opt/TensorRT-Edge-LLM

COPY patches/dflash-mixed-precision-lm-head.patch /tmp/dflash-mixed-precision-lm-head.patch
RUN git apply /tmp/dflash-mixed-precision-lm-head.patch
COPY patches/modelopt-embedded-mixed-precision.patch /tmp/modelopt-embedded-mixed-precision.patch
RUN git apply --check /tmp/modelopt-embedded-mixed-precision.patch \
    && git apply /tmp/modelopt-embedded-mixed-precision.patch
COPY patches/qwen38-dspark-config.patch /tmp/qwen38-dspark-config.patch
RUN git apply --check /tmp/qwen38-dspark-config.patch \
    && git apply /tmp/qwen38-dspark-config.patch
COPY patches/dspark-shared-lm-head.patch /tmp/dspark-shared-lm-head.patch
RUN git apply --check /tmp/dspark-shared-lm-head.patch \
    && git apply /tmp/dspark-shared-lm-head.patch
COPY patches/qwen38-hybrid-dspark.patch /tmp/qwen38-hybrid-dspark.patch
RUN git apply --check /tmp/qwen38-hybrid-dspark.patch \
    && git apply /tmp/qwen38-hybrid-dspark.patch

# Keep the NVIDIA-provided CUDA-enabled torch build instead of replacing it
# with a generic PyPI wheel.
RUN python -m pip install \
      transformers==5.14.1 \
      onnx==1.19.0 \
      onnxscript==0.7.1 \
      safetensors==0.8.0 \
      numpy==2.2.6 \
      onnx-graphsurgeon==0.6.1 \
    && python -m pip install --no-deps -e .

ENTRYPOINT ["tensorrt-edgellm-export"]


# Defaults to the container version NVIDIA documents for the supported
# workflow.  Build arguments also let us test a newer TensorRT/CUDA pair
# without changing anything on the host or replacing the known-good image.
FROM ${TRT_EDGE_RUNTIME_IMAGE} AS runtime-builder

ARG TRT_EDGE_CUDA_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      git \
      ninja-build \
      python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git /opt/TensorRT-Edge-LLM \
    && cd /opt/TensorRT-Edge-LLM \
    && git checkout 71dd1bae032e70771265917ec74d3ff4cad07a10 \
    && git submodule update --init --recursive

WORKDIR /opt/TensorRT-Edge-LLM

COPY patches/openai-image-url.patch /tmp/openai-image-url.patch
RUN git apply /tmp/openai-image-url.patch
COPY patches/spec-decode-multimodal.patch /tmp/spec-decode-multimodal.patch
RUN git apply /tmp/spec-decode-multimodal.patch
COPY patches/dflash-mixed-precision-lm-head.patch /tmp/dflash-mixed-precision-lm-head.patch
RUN git apply /tmp/dflash-mixed-precision-lm-head.patch

RUN python -m pip install \
      pybind11==3.0.4 \
      fastapi==0.139.2 \
      uvicorn==0.51.0 \
      av==17.1.0 \
      python-multipart==0.0.32 \
      transformers==5.14.1 \
      safetensors==0.8.0 \
    && python -m pip install --no-deps -e .

RUN cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DTRT_PACKAGE_DIR=/usr \
      -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
      -DEMBEDDED_TARGET=gb10 \
      -DCUDA_CTK_VERSION="${TRT_EDGE_CUDA_VERSION}" \
      -DENABLE_CUTE_DSL=ALL \
      -DBUILD_PYTHON_BINDINGS=ON \
      -Dpybind11_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
    && cmake --build build --parallel "$(nproc)"

# Keep the upstream native build cached, then recompile only the DSpark hybrid
# registry/config objects and the binaries that link them.
COPY patches/qwen38-hybrid-dspark-runtime.patch /tmp/qwen38-hybrid-dspark-runtime.patch
RUN git apply --check /tmp/qwen38-hybrid-dspark-runtime.patch \
    && git apply /tmp/qwen38-hybrid-dspark-runtime.patch \
    && cmake --build build --parallel "$(nproc)"

# Local compatibility layer, pinned to v0.10.0 commit 71dd1bae.  Keep this
# after the native build so Python-only streaming/API changes reuse the costly
# C++ build cache and never trigger an engine rebuild.
COPY patches/modelopt-embedded-mixed-precision.patch /tmp/modelopt-embedded-mixed-precision.patch
RUN git apply --check /tmp/modelopt-embedded-mixed-precision.patch \
    && git apply /tmp/modelopt-embedded-mixed-precision.patch
COPY patches/qwen38-dspark-config.patch /tmp/qwen38-dspark-config.patch
RUN git apply --check /tmp/qwen38-dspark-config.patch \
    && git apply /tmp/qwen38-dspark-config.patch
COPY patches/dspark-shared-lm-head.patch /tmp/dspark-shared-lm-head.patch
RUN git apply --check /tmp/dspark-shared-lm-head.patch \
    && git apply /tmp/dspark-shared-lm-head.patch
COPY patches/qwen38-hybrid-dspark.patch /tmp/qwen38-hybrid-dspark.patch
RUN git apply --check /tmp/qwen38-hybrid-dspark.patch \
    && git apply /tmp/qwen38-hybrid-dspark.patch
COPY patches/qwen-streaming-reasoning.patch /tmp/qwen-streaming-reasoning.patch
RUN git apply --check /tmp/qwen-streaming-reasoning.patch \
    && git apply /tmp/qwen-streaming-reasoning.patch
COPY patches/local-webp-image.patch /tmp/local-webp-image.patch
RUN git apply --check /tmp/local-webp-image.patch \
    && git apply /tmp/local-webp-image.patch


FROM runtime-builder AS runtime

ENV PYTHONPATH=/opt/TensorRT-Edge-LLM
WORKDIR /opt/TensorRT-Edge-LLM
ENTRYPOINT []
CMD ["bash"]


# Optional BuildKit local-output target, analogous to the llama.cpp host
# artifact target.  The files still require matching CUDA/TensorRT libraries,
# so normal execution uses the runtime container above.
FROM scratch AS host-artifacts
COPY --from=runtime-builder /opt/TensorRT-Edge-LLM/build/ /
