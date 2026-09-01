ARG SGLANG_BASE=lmsysorg/sglang:qwen38flashnext@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1

FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517 AS sglang-source

ARG SGLANG_PLE_COMMIT=0977d22bb005695fef0aee4bc59adfab45b7a496

ADD https://github.com/sgl-project/sglang/archive/${SGLANG_PLE_COMMIT}.tar.gz /tmp/sglang.tar.gz

RUN mkdir -p /src/sglang \
    && tar -xzf /tmp/sglang.tar.gz --strip-components=1 -C /src/sglang

FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517 AS spark-recipe

ARG SPARK_RECIPE_COMMIT=4f425ca561f767997738e894ee578673e79b01b1

ADD https://github.com/hashd1ve/qwen38-flash-next-one-dgx-spark/archive/${SPARK_RECIPE_COMMIT}.tar.gz /tmp/recipe.tar.gz

RUN mkdir -p /src/recipe \
    && tar -xzf /tmp/recipe.tar.gz --strip-components=1 -C /src/recipe

FROM ${SGLANG_BASE}

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# PR #37068 supplies the maintained file-backed PLE backend, including prefill
# hints and an RSS trimmer. The official day-zero image and this PR share the
# qwen4-main-squashed base, so replacing the Python package is deterministic.
COPY --from=sglang-source /src/sglang/python/sglang/ /sgl-workspace/sglang/python/sglang/

# tqdm redraws shard progress with carriage returns. Docker's line-oriented log
# API cannot stream those updates, so expose a sparse machine-readable progress
# line for SparkTalk without changing the loader's behavior.
COPY patches/structured-weight-progress.patch /opt/sparktalk-flash-next/patches/structured-weight-progress.patch
RUN patch -d /sgl-workspace/sglang/python -p1 \
      < /opt/sparktalk-flash-next/patches/structured-weight-progress.patch

# PR #36845's SM121 QSA kernel is already vendored and validated by the pinned
# single-Spark recipe. Apply it after the PLE source overlay because both are
# intentionally kept independent.
COPY --from=spark-recipe /src/recipe/patches/ /opt/sparktalk-flash-next/patches/

RUN set -eux; \
    qsa_path="$(python3 -c 'import sglang.srt.layers.attention.qwen_sparse_attn_backend as m; print(m.__file__)')"; \
    python3 /opt/sparktalk-flash-next/patches/qsa_sm121_kda.py "$qsa_path"; \
    install -D -m 0644 \
      /opt/sparktalk-flash-next/patches/qsa_sm121_varlen.py \
      "$(dirname "$qsa_path")/qsa/sm121_varlen.py"; \
    kernels_dir="$(python3 -c 'import sglang.kernels as k; print(k.__path__[0])')"; \
    rm -rf "$kernels_dir/kda_kernels"; \
    cp -R /opt/sparktalk-flash-next/patches/kda_kernels "$kernels_dir/kda_kernels"; \
    python3 - <<'PY'
import ast
import inspect

from sglang.srt.models import qwen4_exp
from sglang.srt.model_loader import weight_utils
from sglang.srt.server_args import ServerArgs

source = inspect.getsource(qwen4_exp)
assert "ple_offload_backend" in source
assert "make_ple_file_rss_trimmer" in source

weight_source = inspect.getsource(weight_utils.buffered_multi_thread_safetensors_weights_iterator)
assert "SGLANG_WEIGHT_PROGRESS" in weight_source

from sglang.srt.layers.attention import qwen_sparse_attn_backend

qsa_source = inspect.getsource(qwen_sparse_attn_backend)
ast.parse(qsa_source)
assert "kda_kernels.qwen38_qsa_sm121" in qsa_source
assert "qsa.sm121_varlen" in qsa_source

fields = ServerArgs.__dataclass_fields__
assert "ple_offload_backend" in fields
assert "ple_offload_dir" in fields
print("SparkTalk Flash-Next patches verified")
PY

ENV MAX_JOBS=1 \
    TORCHINDUCTOR_COMPILE_THREADS=4 \
    PYTHONUNBUFFERED=1
