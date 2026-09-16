# Local immutable baseline: sha256:8e7e062186f841453ef0ec6f713043c5b65447decc3835206685128c18e42262
# Locally hydrated rootfs: sha256:ae52198a0f4c814c3185bf0937e639b72b906f55e1359381529146d50c9b5045
# eugr/spark-vllm-docker, vLLM b40673cd0 (2026-09-13), transformers 5.17.0.
ARG BASE_IMAGE=dgx-vllm-moe-tp1:b40673cd0-v3
FROM ${BASE_IMAGE}
COPY patch_runtime.py patch_gemma_spec.py launch.py /opt/sparktalk-moe/
COPY draft_vocab/ /opt/sparktalk-moe/draft_vocab/
RUN python3 /opt/sparktalk-moe/patch_runtime.py \
    && python3 /opt/sparktalk-moe/draft_vocab/install_shortlist.py \
    && python3 /opt/sparktalk-moe/patch_gemma_spec.py
ENTRYPOINT ["python3", "/opt/sparktalk-moe/launch.py"]
