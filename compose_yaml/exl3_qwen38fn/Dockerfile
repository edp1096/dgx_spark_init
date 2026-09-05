ARG BASE_IMAGE=nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c

FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517 AS source

ARG EXLLAMAV3_COMMIT=499890c75d20d8e7c9d061f37189ae611a5c9f0b
ARG SERVER_COMMIT=63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed
ARG ARM_PORT_COMMIT=63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed

ADD https://github.com/turboderp-org/exllamav3/archive/${EXLLAMAV3_COMMIT}.tar.gz /tmp/exllamav3.tar.gz
ADD https://github.com/MiaAI-Lab/exllamav3/archive/${ARM_PORT_COMMIT}.tar.gz /tmp/arm-port.tar.gz
ADD https://raw.githubusercontent.com/MiaAI-Lab/exllamav3/${SERVER_COMMIT}/tools/serve_openai.py /tmp/serve_openai.py

RUN mkdir -p /src/exllamav3/tools /src/arm-port \
    && tar -xzf /tmp/exllamav3.tar.gz --strip-components=1 -C /src/exllamav3 \
    && tar -xzf /tmp/arm-port.tar.gz --strip-components=1 -C /src/arm-port \
    && for file in \
        exllamav3/exllamav3_ext/avx2_target.cpp \
        exllamav3/exllamav3_ext/avx2_target.h \
        exllamav3/exllamav3_ext/avx512_target.cpp \
        exllamav3/exllamav3_ext/avx512_target.h \
        exllamav3/exllamav3_ext/parallel/all_reduce_cpu_avx2.cpp \
        exllamav3/exllamav3_ext/parallel/all_reduce_cpu_avx512.cpp \
        exllamav3/exllamav3_ext/parallel/all_reduce_cpu.cu \
        exllamav3/exllamav3_ext/cpu/moe_handoff.cu \
        exllamav3/exllamav3_ext/cpu/moe_mul1.cpp \
        exllamav3/exllamav3_ext/cpu/moe_mul1.h; do \
          install -m 0644 "/src/arm-port/$file" "/src/exllamav3/$file"; \
       done \
    && install -m 0644 /tmp/serve_openai.py /src/exllamav3/tools/serve_openai.py

FROM ${BASE_IMAGE}

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="12.0;12.1" \
    MAX_JOBS=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_EXTENSIONS_DIR=/cache/torch_extensions

COPY --from=source /src/exllamav3 /opt/src/exllamav3
COPY ablit_runtime.py /opt/src/exllamav3/exllamav3/ablit_runtime.py
COPY prepare_direction.py /opt/exl3/prepare_direction.py
COPY patches/runtime-ablit.patch /tmp/runtime-ablit.patch
COPY patches/serve-openai-config.patch /tmp/serve-openai-config.patch
COPY constraints.txt /opt/exl3/constraints.txt

RUN patch -d /opt/src/exllamav3 -p1 < /tmp/runtime-ablit.patch \
    && patch -d /opt/src/exllamav3 -p1 < /tmp/serve-openai-config.patch \
    && PIP_CONSTRAINT=/opt/exl3/constraints.txt python3 -m pip install --no-build-isolation /opt/src/exllamav3 \
    && python3 -m pip install aiohttp huggingface_hub \
    && install -m 0644 /opt/src/exllamav3/tools/serve_openai.py /opt/exl3/serve_openai.py \
    && python3 -m py_compile /opt/exl3/serve_openai.py /opt/exl3/prepare_direction.py \
    && python3 -c "import exllamav3, exllamav3_ext; print('exllamav3 CUDA extension ready')" \
    && rm -rf /root/.cache/pip /tmp/runtime-ablit.patch /tmp/serve-openai-config.patch

WORKDIR /opt/exl3
