# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=13.1.1
ARG UBUNTU_VERSION=24.04

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

# PR 17 adds the v2607 factor-2 decoder and Korean/Arabic/Portuguese profiles.
# Keep the immutable commit here until the changes are available upstream.
ARG NEMO_SPEECH_REF=7752dabb3453ec7c5fd751e5e2052650aaa20497
ARG CUDA_ARCH=121
ARG BUILD_JOBS=4

ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates cmake git git-lfs ninja-build build-essential pkg-config \
      libsentencepiece-dev libssl-dev

WORKDIR /src
COPY patches/0001-fix-v2607-speaker-order.patch /tmp/0001-fix-v2607-speaker-order.patch
RUN git clone --filter=blob:none https://github.com/NVIDIA/NeMo-Speech.cpp.git . \
    && git fetch --depth 1 origin refs/pull/17/head \
    && test "$(git rev-parse FETCH_HEAD)" = "${NEMO_SPEECH_REF}" \
    && git checkout --detach "${NEMO_SPEECH_REF}" \
    && git apply /tmp/0001-fix-v2607-speaker-order.patch \
    && git submodule update --init --recursive --depth 1 \
      ggml llama.cpp third_party/cpp-httplib third_party/open_jtalk third_party/cppjieba \
    && git lfs install --local \
    && git lfs pull --include='src/tts/tokenizer/mandarin_data/**'

RUN scripts/configure.sh cuda-server \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
      -DNEMO_SPEECH_BUILD_ASR=OFF \
      -DNEMO_SPEECH_BUILD_DIAR=OFF \
      -DNEMO_SPEECH_BUILD_TTS=ON \
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
      -DNEMO_SPEECH_TTS_WITH_JA=ON \
      -DNEMO_SPEECH_TTS_WITH_ZH=ON \
    && cmake --build --preset cuda-server --target nemo-speech -- -j"${BUILD_JOBS}"

FROM python:3.11-slim AS converter
ARG NEMO_SPEECH_REF=7752dabb3453ec7c5fd751e5e2052650aaa20497
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY patches/0001-fix-v2607-speaker-order.patch /tmp/0001-fix-v2607-speaker-order.patch
RUN git clone --filter=blob:none https://github.com/NVIDIA/NeMo-Speech.cpp.git . \
    && git fetch --depth 1 origin refs/pull/17/head \
    && test "$(git rev-parse FETCH_HEAD)" = "${NEMO_SPEECH_REF}" \
    && git checkout --detach "${NEMO_SPEECH_REF}" \
    && git apply /tmp/0001-fix-v2607-speaker-order.patch \
    && pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python", "/src/convert_model.py"]

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates libgomp1 libsentencepiece0 libssl3 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/nemo-speech
COPY --from=builder /src/build/cuda-server/bin/ ./
COPY --from=builder /src/build/cuda-server/open_jtalk_dic/ /usr/local/share/nemo-speech/open_jtalk_dic/
COPY --from=builder /src/src/tts/tokenizer/mandarin_data/ /usr/local/share/nemo-speech/mandarin_g2p/
COPY --from=builder /src/LICENSE /src/NOTICE /src/THIRD_PARTY_NOTICES.md ./licenses/
ENV LD_LIBRARY_PATH=/opt/nemo-speech
EXPOSE 8692
ENTRYPOINT ["/opt/nemo-speech/nemo-speech"]
