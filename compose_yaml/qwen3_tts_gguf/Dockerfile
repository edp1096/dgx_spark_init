# syntax=docker/dockerfile:1.7
ARG CUDA_VERSION=13.1.1
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS builder
ARG QWENTTS_REF=51512f129a7419567f4b8abfb06801451789b8f1
ARG CUDA_ARCH=121
ARG BUILD_JOBS=2
ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git cmake ninja-build build-essential python3
WORKDIR /src
RUN git clone --filter=blob:none https://github.com/ServeurpersoCom/qwentts.cpp.git . \
    && git checkout --detach "${QWENTTS_REF}" \
    && git submodule update --init --depth 1 ggml
COPY patches/runtime_lifecycle.py /tmp/runtime_lifecycle.py
RUN python3 /tmp/runtime_lifecycle.py
RUN --mount=type=cache,target=/src/build,id=qwentts-51512f1-cuda13-sm121 \
    ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON -DGGML_CUDA_ENABLE_UNIFIED_MEMORY=ON \
      -DGGML_CUDA_FA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
      -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
    && cmake --build build --target tts-server -j"${BUILD_JOBS}" \
    && mkdir /runtime && cp build/tts-server /runtime/ \
    && find build -name 'libggml*.so*' -exec cp -P {} /runtime/ \; \
    && head -n 21 vendor/yyjson/yyjson.h > /runtime/yyjson-LICENSE
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/qwentts
COPY --from=builder /runtime/ ./
COPY --from=builder /src/LICENSE ./licenses/qwentts-LICENSE
COPY --from=builder /src/ggml/LICENSE ./licenses/ggml-LICENSE
COPY --from=builder /src/vendor/cpp-httplib/LICENSE ./licenses/cpp-httplib-LICENSE
ENV LD_LIBRARY_PATH=/opt/qwentts GGML_BACKEND=CUDA0 OMP_NUM_THREADS=4
EXPOSE 8692
ENTRYPOINT ["/opt/qwentts/tts-server"]
CMD ["--model", "/models/qwen-talker-0.6b-customvoice-Q8_0.gguf", "--codec", "/models/qwen-tokenizer-12hz-Q8_0.gguf", "--alias", "qwen3-tts-0.6b-q8", "--host", "0.0.0.0", "--port", "8692", "--lang", "Korean", "--max-batch", "1"]
