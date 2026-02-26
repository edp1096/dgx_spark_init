FROM nvcr.io/nvidia/pytorch:26.01-py3

ARG VLLM_VERSION=v0.16.0
ARG MAX_JOBS=16

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV MAX_JOBS=${MAX_JOBS}

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ninja-build ccache \
    && rm -rf /var/lib/apt/lists/*

# Fix missing headers
RUN for f in /usr/local/lib/python3.12/dist-packages/nvidia/cu13/include/*.h; do \
        fname=$(basename "$f"); \
        [ ! -f "/usr/local/cuda/include/$fname" ] && ln -sf "$f" "/usr/local/cuda/include/$fname"; \
    done

RUN git clone https://github.com/vllm-project/vllm.git -b ${VLLM_VERSION} /workspace/vllm
WORKDIR /workspace/vllm

RUN python3 use_existing_torch.py && \
    sed -i "/flashinfer/d" requirements/cuda.txt && \
    pip install -r requirements/build.txt

RUN pip install --no-build-isolation .

RUN pip install 'transformers>=4.56.0,<5.0' \
    'ray[default]' \
    fastsafetensors