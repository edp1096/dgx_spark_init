# Stage 1: Build
FROM nvcr.io/nvidia/cuda:13.1.1-devel-ubuntu24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake git build-essential libgomp1 libcurl4-openssl-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN git clone https://github.com/ggml-org/llama.cpp.git

WORKDIR /tmp/llama.cpp
RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_F16=ON \
    -DLLAMA_CURL=ON \
    -DCMAKE_CUDA_ARCHITECTURES=121a-real \
    && cmake --build build -j $(nproc)

# Stage 2: Runtime
FROM nvcr.io/nvidia/cuda:13.1.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libcurl4 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /llama.cpp
COPY --from=builder /tmp/llama.cpp/build/bin/ ./
