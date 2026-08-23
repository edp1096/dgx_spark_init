FROM nvcr.io/nvidia/pytorch:26.02-py3

ENV PIP_ROOT_USER_ACTION=ignore
ENV GITHUB_RAW=https://raw.githubusercontent.com/edp1096/dgx_spark_init/main/compose_yaml/zit
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /root/zit-ui

# [1/7] Clone Z-Image repo
ARG ZIMAGE_COMMIT=26f23ed
RUN git clone https://github.com/Tongyi-MAI/Z-Image.git /root/zit-ui/Z-Image && \
    cd /root/zit-ui/Z-Image && git checkout ${ZIMAGE_COMMIT}

# [2/7] Install Z-Image
RUN cd /root/zit-ui/Z-Image && pip install -q -e . --no-deps

# [3/7] Download UI files
RUN mkdir -p app/ui/videox_models app/ui/preprocessors/dwpose_utils app/ui/preprocessors/zoe logs && \
    curl -sfL ${GITHUB_RAW}/app/app.py -o app/app.py && \
    for f in zit_config.py pipeline_manager.py worker.py generators.py download_models.py i18n.py face_swap.py; do \
        curl -sfL ${GITHUB_RAW}/app/ui/$f -o app/ui/$f; \
    done && \
    for f in __init__.py attention_kernel.py attention_utils.py fp8_optimization.py pipeline_z_image_control.py z_image_transformer2d.py z_image_transformer2d_control.py; do \
        curl -sfL ${GITHUB_RAW}/app/ui/videox_models/$f -o app/ui/videox_models/$f; \
    done && \
    for f in __init__.py canny.py depth.py dwpose.py gray.py hed.py; do \
        curl -sfL ${GITHUB_RAW}/app/ui/preprocessors/$f -o app/ui/preprocessors/$f; \
    done && \
    for f in __init__.py onnxdet.py onnxpose.py wholebody.py util.py; do \
        curl -sfL ${GITHUB_RAW}/app/ui/preprocessors/dwpose_utils/$f -o app/ui/preprocessors/dwpose_utils/$f; \
    done

# [4/7] Python packages
RUN pip install -q gradio accelerate safetensors huggingface_hub psutil loguru einops fire transformers peft timm

# [5/7] q8_kernels (aarch64 FP8 GEMM)
ARG Q8_WHEEL_URL=https://raw.githubusercontent.com/edp1096/dgx_spark_init/main/compose_yaml/ltx-2.3/builder/q8_kernels-0.0.5-cp312-cp312-linux_aarch64.whl
RUN pip install -q ${Q8_WHEEL_URL}

# [6/7] bitsandbytes (source build for aarch64)
ARG BNB_COMMIT=925d83e
RUN git clone --depth 250 https://github.com/bitsandbytes-foundation/bitsandbytes.git /tmp/bnb-src && \
    cd /tmp/bnb-src && git checkout ${BNB_COMMIT} && \
    cmake -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY=121 -S . -B build && \
    cmake --build build -j$(nproc) && \
    pip install -q . && \
    rm -rf /tmp/bnb-src

# [7/7] Runtime packages used in entrypoint
RUN pip install -q diffusers sentencepiece opencv-python-headless fastsafetensors

# Download and run update.sh to apply latest updates
RUN wget -q https://raw.githubusercontent.com/edp1096/dgx_spark_init/main/compose_yaml/zit/builder/update.sh -O /root/zit-ui/update.sh && \
    chmod +x /root/zit-ui/update.sh && \
    /root/zit-ui/update.sh --no-restart

# Clean pip cache
RUN rm -rf /root/.cache/pip

WORKDIR /root/zit-ui