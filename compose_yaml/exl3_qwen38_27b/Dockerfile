ARG BASE_IMAGE=nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c

FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517 AS source

ARG EXLLAMAV3_COMMIT=63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed

ADD https://github.com/MiaAI-Lab/exllamav3/archive/${EXLLAMAV3_COMMIT}.tar.gz /tmp/exllamav3.tar.gz

RUN mkdir -p /src/exllamav3 \
    && tar -xzf /tmp/exllamav3.tar.gz --strip-components=1 -C /src/exllamav3

FROM ${BASE_IMAGE}

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="12.0;12.1" \
    MAX_JOBS=8 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_EXTENSIONS_DIR=/cache/torch_extensions

COPY --from=source /src/exllamav3 /opt/src/exllamav3
COPY patches/serve-openai-config.patch /tmp/serve-openai-config.patch
COPY constraints.txt /opt/exl3/constraints.txt

RUN patch -d /opt/src/exllamav3 -p1 < /tmp/serve-openai-config.patch \
    && PIP_CONSTRAINT=/opt/exl3/constraints.txt python3 -m pip install --no-build-isolation /opt/src/exllamav3 \
    && python3 -m pip install aiohttp huggingface_hub \
    && install -D -m 0644 /opt/src/exllamav3/tools/serve_openai.py /opt/exl3/serve_openai.py \
    && python3 -c "import exllamav3, exllamav3_ext; print('exllamav3 CUDA extension ready')" \
    && rm -rf /root/.cache/pip /tmp/serve-openai-config.patch

WORKDIR /opt/exl3
