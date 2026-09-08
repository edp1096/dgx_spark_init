FROM vllm/vllm-omni:v0.26.0

# The official runtime also carries its source checkout, documentation and UI
# examples under /app.  The installed runtime lives in site-packages, so the
# checkout is unnecessary for serving and is excluded from this deployment.
RUN python3 -c "import shutil; shutil.rmtree('/app/vllm-omni')"

ENTRYPOINT []
