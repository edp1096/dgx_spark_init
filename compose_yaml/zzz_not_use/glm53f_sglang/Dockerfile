# Reuse the validated ARM64 CUDA/PyTorch stack, replace Qwen-specific Python code.
ARG BASE_IMAGE=ghcr.io/edp1096/sglang-qwen38-qad:sm121-v5@sha256:c79141f03a89b8888920f629ac638a6915eec5abb0dabe75688977c3c58b75f5
FROM ${BASE_IMAGE}
USER root
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ARG SGLANG_COMMIT=425a1f8f247d0cc17f2a3f3c2dba6c0bdf936552
ENV PYTHONPATH="" SGLANG_BUILD_RUST_EXTS=none SETUPTOOLS_SCM_PRETEND_VERSION=0.5.20.dev0 \
    MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2 TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a PYTHONUNBUFFERED=1
ADD https://github.com/sgl-project/sglang/archive/${SGLANG_COMMIT}.tar.gz /tmp/sglang-source.tar.gz
RUN mkdir -p /opt/glm53/sglang \
    && tar -xzf /tmp/sglang-source.tar.gz --strip-components=1 -C /opt/glm53/sglang \
    && rm /tmp/sglang-source.tar.gz \
    && printf 'nvidia-cutlass-dsl[cu13]==4.7.0\n' > /opt/glm53/dependency-overrides.txt \
    && uv pip install --system --break-system-packages --no-build-isolation \
       --override /opt/glm53/dependency-overrides.txt /opt/glm53/sglang/python \
    && printf '%s\n' "$SGLANG_COMMIT" > /opt/glm53/sglang-commit
ENV UV_CONCURRENT_DOWNLOADS=1
RUN uv pip install --system --break-system-packages --no-deps flashinfer-cubin==0.6.18 --index-url https://flashinfer.ai/whl \
    && uv pip install --system --break-system-packages --no-deps flashinfer-jit-cache==0.6.18 --index-url https://flashinfer.ai/whl/cu130
WORKDIR /opt/glm53
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]
COPY patch_runtime.py /opt/glm53/runtime/
ENV PYTHONPATH=/opt/glm53/runtime
# The no-isolation bootstrap needs setuptools-scm for honest wheel metadata.
# Rebuild only SGLang after dependency bootstrap; preserve the pinned CUDA stack.
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.5.20.dev0+g425a1f8f \
    SGLANG_BUILD_COMMIT=425a1f8f247d0cc17f2a3f3c2dba6c0bdf936552 \
    SGLANG_BUILD_URL=https://github.com/sgl-project/sglang/commit/425a1f8f247d0cc17f2a3f3c2dba6c0bdf936552 \
    SGLANG_IMAGE_TAG=dgx-sglang-glm53:sm121-dev3
RUN uv pip install --system --break-system-packages setuptools-scm==8.3.1 \
    && uv pip install --system --break-system-packages --no-cache --no-deps --no-build-isolation --reinstall /opt/glm53/sglang/python \
    && python3 /opt/glm53/runtime/patch_runtime.py \
    && python3 -c 'from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration; from sglang.srt.server_args import ServerArgs; print("GLM imports verified")'
# SGLang's dependency pin predates the GLM5Next multimodal processor.
# Match the version already validated with this checkpoint on vLLM.
RUN uv pip install --system --break-system-packages transformers==5.17.0 \
    && python3 -c 'from transformers import Glm5NextProcessor, Glm5NextImageProcessor, Glm5NextVideoProcessor; from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration; print("GLM multimodal imports verified")'
COPY glm53_b12x.py /opt/glm53/runtime/
ENV SGLANG_IMAGE_TAG=dgx-sglang-glm53:sm121-dev6
