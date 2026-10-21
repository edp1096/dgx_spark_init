ARG BASE_IMAGE=sparktalk-qwen38fn_exl3:1.5.4-managed1
FROM ${BASE_IMAGE}

# The Python CUDA EXL3 implementation is present at this historical revision.
# Current native-only TensorFold main is a different implementation.
ARG TENSORFOLD_REVISION=db281878ddb836fd0df510d8771ecb7e0fe47d26
RUN git init /opt/tensorfold-exl3 \
    && git -C /opt/tensorfold-exl3 remote add origin https://github.com/ashhart/TensorFold.git \
    && git -C /opt/tensorfold-exl3 fetch --depth 1 origin ${TENSORFOLD_REVISION} \
    && git -C /opt/tensorfold-exl3 checkout --detach FETCH_HEAD

ENV PYTHONPATH=/opt/tensorfold-exl3/src \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TENSORFOLD_NO_UPDATE_CHECK=1 \
    TENSORFOLD_CUDA_MEMORY_LIMIT_GB=108 \
    TENSORFOLD_MEMORY_RESERVE_GIB=8 \
    MAX_JOBS=1 \
    TORCH_CUDA_ARCH_LIST=12.1 \
    TORCH_EXTENSIONS_DIR=/cache/torch-extensions \
    TRITON_CACHE_DIR=/cache/triton

COPY serve.py /opt/serve-qwen38-exl3.py
ENTRYPOINT ["python", "-u", "/opt/serve-qwen38-exl3.py"]
