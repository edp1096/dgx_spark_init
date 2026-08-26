FROM dgx-ltx2.5-nvfp4-api:latest

ARG SOL_ENGINE_COMMIT=d0c0a4685ab5dc2336d18b7213d85f13def92418

RUN git clone --filter=blob:none --no-checkout https://github.com/NVlabs/Sana.git /opt/sol-engine && \
    git -C /opt/sol-engine checkout "${SOL_ENGINE_COMMIT}" && \
    python -m pip install --no-cache-dir -e /opt/sol-engine/techniques/sparse_backends

ENV PYTHONPATH=/opt/sol-engine \
    SOL_ATTN_STRICT=1

WORKDIR /opt/ltx-sol-experiment
COPY ltx-2.5_sol_experiment/benchmark_sol.py ./benchmark_sol.py

ENTRYPOINT ["python", "/opt/ltx-sol-experiment/benchmark_sol.py"]
