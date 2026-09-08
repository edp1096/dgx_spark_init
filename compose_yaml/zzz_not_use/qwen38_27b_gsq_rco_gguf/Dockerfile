FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c AS tools
ARG LLAMA_CPP_REF=e71b80510c848c00175924ecf3c40333ccae8eb5
RUN apt-get update && apt-get install -y --no-install-recommends cmake git build-essential libssl-dev && rm -rf /var/lib/apt/lists/*
RUN git init /opt/llama.cpp && cd /opt/llama.cpp && git remote add origin https://github.com/ggml-org/llama.cpp.git && git fetch --depth 1 origin "$LLAMA_CPP_REF" && git checkout --detach FETCH_HEAD && test "$(git rev-parse HEAD)" = "$LLAMA_CPP_REF"
RUN python3 -m venv --system-site-packages /opt/converter && /opt/converter/bin/pip install --no-cache-dir numpy==1.26.4 transformers==4.57.6 sentencepiece==0.2.1 protobuf==4.25.8
RUN /opt/converter/bin/pip install --no-deps /opt/llama.cpp/gguf-py && /opt/converter/bin/pip freeze > /opt/converter-requirements.txt
RUN cmake -S /opt/llama.cpp -B /opt/llama.cpp/build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF && cmake --build /opt/llama.cpp/build-cpu --target llama-quantize -j 4
ENV PATH=/opt/converter/bin:/opt/llama.cpp/build-cpu/bin:$PATH
WORKDIR /workspace
ENTRYPOINT []
CMD ["python3", "pipeline.py", "--help"]

FROM tools AS runtime
RUN cmake -S /opt/llama.cpp -B /opt/llama.cpp/build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=121a-real -DLLAMA_BUILD_TESTS=OFF && cmake --build /opt/llama.cpp/build-cuda --target llama-server llama-cli -j 3
ENV PATH=/opt/converter/bin:/opt/llama.cpp/build-cuda/bin:$PATH
CMD ["llama-server", "--help"]
