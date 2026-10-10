FROM lmsysorg/sglang:qwen38flashnext@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1 AS base
FROM base AS build
ARG VELO_COMMIT=a6ad23d60e082ff9cca2bb78adda4ecee771388c
RUN apt-get update && apt-get install -y --no-install-recommends curl git build-essential libibverbs-dev libssl-dev pkg-config && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.99.0 --profile minimal
ENV PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs
RUN git clone https://github.com/sf-stav/veloGB10.git /build && cd /build && git checkout "$VELO_COMMIT"
WORKDIR /build
COPY velogb10-exl3-yarn.patch velogb10-exl3-schema.patch ./
RUN git apply velogb10-exl3-yarn.patch && git apply velogb10-exl3-schema.patch && cargo build --release --locked --bin gb10_inference
RUN mkdir /engine && cp target/release/gb10_inference /engine/ && mkdir -p /engine/src && cp -a src/ptx /engine/src/
FROM base AS runtime
WORKDIR /opt/velogb10
COPY --from=build /engine/ ./
COPY launch.py release_weight_cache.py ./
ENV HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "/opt/velogb10/launch.py"]
