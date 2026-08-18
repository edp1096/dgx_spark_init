# syntax=docker/dockerfile:1.6

ARG BUILD_JOBS=16
ARG GPU_ARCH=12.1a

# =========================================================
# Stage 1: Base
# =========================================================
FROM nvcr.io/nvidia/pytorch:26.02-py3 AS base

ARG BUILD_JOBS
ARG GPU_ARCH

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV TORCH_CUDA_ARCH_LIST=${GPU_ARCH}
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# ccache
ENV PATH=/usr/lib/ccache:$PATH
ENV CCACHE_DIR=/root/.ccache
ENV CCACHE_MAXSIZE=50G
ENV CCACHE_COMPRESS=1
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CMAKE_CUDA_COMPILER_LAUNCHER=ccache

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ninja-build ccache curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv && pip uninstall -y flash-attn

ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy

# =========================================================
# Stage 2: FlashInfer Builder
# =========================================================
FROM base AS flashinfer-builder

ARG GPU_ARCH
ENV FLASHINFER_CUDA_ARCH_LIST=${GPU_ARCH}

ARG FLASHINFER_REF=main
ARG CACHEBUST_FLASHINFER=1

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install nvidia-nvshmem-cu13 "apache-tvm-ffi<0.2"

RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
    cd /repo-cache && \
    if [ ! -d "flashinfer" ]; then \
        git clone --recursive https://github.com/flashinfer-ai/flashinfer.git && \
        if [ "$FLASHINFER_REF" != "main" ]; then cd flashinfer && git checkout ${FLASHINFER_REF}; fi; \
    else \
        cd flashinfer && \
        git fetch origin && git fetch origin --tags --force && \
        (git checkout --detach origin/${FLASHINFER_REF} 2>/dev/null || git checkout ${FLASHINFER_REF}) && \
        git submodule update --init --recursive && git clean -fdx; \
    fi && \
    cp -a /repo-cache/flashinfer /workspace/flashinfer

WORKDIR /workspace/flashinfer

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    sed -i -e 's/license = "Apache-2.0"/license = { text = "Apache-2.0" }/' -e '/license-files/d' pyproject.toml && \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v && \
    cd flashinfer-cubin && uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v && \
    cd ../flashinfer-jit-cache && uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v

# =========================================================
# Stage 3: vLLM Builder
# =========================================================
FROM base AS vllm-builder

ARG VLLM_REF=main
ARG CACHEBUST_VLLM=1
ARG VLLM_PRS=""

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install nvidia-nvshmem-cu13 "apache-tvm-ffi<0.2"

RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
    cd /repo-cache && \
    if [ ! -d "vllm" ]; then \
        git clone --recursive https://github.com/vllm-project/vllm.git && \
        if [ "$VLLM_REF" != "main" ]; then cd vllm && git checkout ${VLLM_REF}; fi; \
    else \
        cd vllm && \
        git fetch origin && git fetch origin --tags --force && \
        (git checkout --detach origin/${VLLM_REF} 2>/dev/null || git checkout ${VLLM_REF}) && \
        git submodule update --init --recursive && git clean -fdx; \
    fi && \
    cp -a /repo-cache/vllm /workspace/vllm

WORKDIR /workspace/vllm

# Apply PR patches if specified
RUN if [ -n "$VLLM_PRS" ]; then \
        for pr in $VLLM_PRS; do \
            echo "Applying PR #$pr..." && \
            curl -fL "https://github.com/vllm-project/vllm/pull/${pr}.diff" | git apply -v; \
        done; \
    fi

# Hopper compatibility patches
RUN curl -L https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/34758.diff | patch -p1 -R || echo "Cannot revert PR #34758, skipping"
RUN curl -L https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/34302.diff | patch -p1 -R || echo "Cannot revert PR #34302, skipping"

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    python3 use_existing_torch.py && \
    sed -i "/flashinfer/d" requirements/cuda.txt && \
    sed -i '/^triton\b/d' requirements/test.txt && \
    sed -i '/^fastsafetensors\b/d' requirements/test.txt && \
    uv pip install -r requirements/build.txt

RUN --mount=type=cache,id=ccache,target=/root/.ccache \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v

# =========================================================
# Stage 4: Runner
# =========================================================
FROM nvcr.io/nvidia/pytorch:26.02-py3 AS runner

ARG BUILD_JOBS=16
ARG GPU_ARCH=12.1a

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV MAX_JOBS=${BUILD_JOBS}
ENV TORCH_CUDA_ARCH_LIST=${GPU_ARCH}
ENV FLASHINFER_CUDA_ARCH_LIST=${GPU_ARCH}
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl vim git libxcb1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv && pip uninstall -y flash-attn

WORKDIR /workspace/vllm

# Tiktoken encodings (offline)
RUN mkdir -p tiktoken_encodings && \
    wget -O tiktoken_encodings/o200k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" && \
    wget -O tiktoken_encodings/cl100k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
ENV TIKTOKEN_ENCODINGS_BASE=/workspace/vllm/tiktoken_encodings

# Install wheels from builder stages
RUN --mount=type=bind,from=flashinfer-builder,source=/workspace/wheels,target=/workspace/fi-wheels \
    --mount=type=bind,from=vllm-builder,source=/workspace/wheels,target=/workspace/vllm-wheels \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install /workspace/fi-wheels/*.whl /workspace/vllm-wheels/*.whl

# Runtime deps
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install ray[default] fastsafetensors nvidia-nvshmem-cu13
