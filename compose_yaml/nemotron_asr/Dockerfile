# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=13.1.1
ARG UBUNTU_VERSION=24.04

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG NEMO_SPEECH_REF=4f9676226f667d14608487df744f375db87127f8
ARG CUDA_ARCH=121
ARG BUILD_JOBS=2
ARG NEMO_BUILD_ASR=ON
ARG NEMO_BUILD_TTS=OFF

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates cmake git ninja-build build-essential pkg-config \
      libsentencepiece-dev libssl-dev

WORKDIR /src
RUN git clone --filter=blob:none https://github.com/NVIDIA/NeMo-Speech.cpp.git . \
    && git checkout --detach "${NEMO_SPEECH_REF}" \
    && git submodule update --init --depth 1 ggml llama.cpp third_party/cpp-httplib

# Build only the CUDA ASR runtime and its HTTP server. TTS, NMT, gRPC,
# diarization, microphone capture and developer tools are intentionally absent.
RUN scripts/configure.sh cuda-server \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
      -DNEMO_SPEECH_BUILD_ASR="${NEMO_BUILD_ASR}" \
      -DNEMO_SPEECH_BUILD_DIAR=OFF \
      -DNEMO_SPEECH_BUILD_TTS="${NEMO_BUILD_TTS}" \
      -DNEMO_SPEECH_BUILD_NMT=OFF \
      -DNEMO_SPEECH_WITH_NMT=OFF \
      -DNEMO_SPEECH_BUILD_HTTP=ON \
      -DNEMO_SPEECH_BUILD_GRPC=OFF \
      -DNEMO_SPEECH_WITH_GRPC=OFF \
      -DNEMO_SPEECH_BUILD_MIC_CAPTURE=OFF \
      -DNEMO_SPEECH_WITH_FLASHLIGHT=OFF \
      -DNEMO_SPEECH_WITH_NORM=OFF \
      -DNEMO_SPEECH_BUILD_TESTS=OFF \
      -DNEMO_SPEECH_BUILD_TOOLS=OFF \
    && cmake --build --preset cuda-server --target nemo-speech -- -j"${BUILD_JOBS}"

# Conversion stays containerized as well. This target is not part of the
# runtime image and exists only to produce alternate GGUF weights on the host.
FROM python:3.11-slim AS converter
ARG NEMO_SPEECH_REF=4f9676226f667d14608487df744f375db87127f8
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone --filter=blob:none https://github.com/NVIDIA/NeMo-Speech.cpp.git . \
    && git checkout --detach "${NEMO_SPEECH_REF}" \
    && pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python", "/src/convert_model.py"]

# Optional host export, matching compose_yaml/llama.cpp's reproducible build
# pattern. CUDA-linked artifacts are operated through the runtime image below.
FROM scratch AS host-artifacts
COPY --from=builder /src/build/cuda-server/bin/ /
COPY --from=builder /src/LICENSE /licenses/LICENSE
COPY --from=builder /src/NOTICE /licenses/NOTICE
COPY --from=builder /src/THIRD_PARTY_NOTICES.md /licenses/THIRD_PARTY_NOTICES.md

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates libgomp1 libsentencepiece0 libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/nemo-speech
COPY --from=builder /src/build/cuda-server/bin/ ./
COPY --from=builder /src/LICENSE /src/NOTICE /src/THIRD_PARTY_NOTICES.md ./licenses/

ENV LD_LIBRARY_PATH=/opt/nemo-speech
EXPOSE 8693
ENTRYPOINT ["/opt/nemo-speech/nemo-speech"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8693", "--no-ui", "--threads", "1", "--asr.batching.enabled=false", "--asr.streaming.rnnt_right_context", "1", "--asr-model", "/models/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"]
