ARG SGLANG_BASE=lmsysorg/sglang:qwen38flashnext@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1
ARG QWEN38FN_EXL3_BASE=sparktalk-qwen38fn_exl3:1.5.4-sgl1

FROM ${SGLANG_BASE} AS qwen38fn_exl3-build
USER root
ADD https://github.com/turboderp-org/exllamav3/archive/16a49792a3c93d8432d72e6c4bce800841566577.tar.gz /tmp/exllamav3.tar.gz
ADD https://github.com/theroyallab/tabbyAPI/archive/2fd6cc76203a66e13042daf7d76e5898b21c1ad8.tar.gz /tmp/tabbyapi.tar.gz
COPY constraints.txt /opt/exl3-constraints.txt
RUN mkdir -p /opt/exllamav3 /opt/tabbyAPI \
 && tar -xzf /tmp/exllamav3.tar.gz --strip-components=1 -C /opt/exllamav3 \
 && tar -xzf /tmp/tabbyapi.tar.gz --strip-components=1 -C /opt/tabbyAPI \
 && rm /tmp/exllamav3.tar.gz /tmp/tabbyapi.tar.gz \
 && python -c "from pathlib import Path; p=Path('/opt/tabbyAPI/pyproject.toml'); s=p.read_text(); assert s.count('fastapi-slim') == 1; p.write_text(s.replace('fastapi-slim', 'fastapi'))" \
 && python -c "import importlib.metadata as m; from pathlib import Path; pins=dict(line.split('==') for line in Path('/opt/exl3-constraints.txt').read_text().splitlines()); assert all(m.version(p)==v for p,v in pins.items()), 'Unexpected SGLang base dependencies'" \
 && EXLLAMA_NOCOMPILE=1 python -m pip install --no-cache-dir --no-build-isolation -c /opt/exl3-constraints.txt /opt/exllamav3 /opt/tabbyAPI uvloop
ENV MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=12.1 PYTHONUNBUFFERED=1
RUN python -m pip wheel --no-cache-dir --no-deps --no-build-isolation --wheel-dir /opt/exl3-wheels /opt/exllamav3 \
 && python -m pip install --no-deps --force-reinstall /opt/exl3-wheels/exllamav3-1.5.4*.whl \
 && python -c "import importlib.util as u; assert u.find_spec('exllamav3_ext').origin.endswith('.so')"

# An already qualified native image can supply the same engine and extension.
FROM ${QWEN38FN_EXL3_BASE} AS reuse
COPY launch.py release_weight_cache.py /opt/sparktalk-qwen38fn_exl3/
WORKDIR /opt/tabbyAPI
ENTRYPOINT ["python", "/opt/sparktalk-qwen38fn_exl3/launch.py"]

# First installation builds from the pinned public SGLang image without a GPU.
FROM qwen38fn_exl3-build AS qwen38fn_exl3-runtime
COPY launch.py release_weight_cache.py /opt/sparktalk-qwen38fn_exl3/
ENV EXL3_INT8_GEMV=0 EXL3_NGRAM_STREAM=1 EXL3_MOE_CPU_OFFLOAD=0 EXL3_MOE_CPU_SPLIT=0
WORKDIR /opt/tabbyAPI
ENTRYPOINT ["python", "/opt/sparktalk-qwen38fn_exl3/launch.py"]
