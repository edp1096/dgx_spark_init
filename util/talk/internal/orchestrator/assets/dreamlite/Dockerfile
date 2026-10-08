FROM nvcr.io/nvidia/pytorch:26.07-py3

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    PATH=/opt/dreamlite-venv/bin:$PATH
RUN python -m venv --system-site-packages /opt/dreamlite-venv && \
    pip install --no-cache-dir diffusers==0.39.0 transformers==4.57.3 \
      huggingface-hub==0.36.2 accelerate==1.12.0 \
      sentencepiece==0.2.1 protobuf==6.33.5 fastapi==0.116.1 uvicorn==0.35.0
WORKDIR /opt/dreamlite-api
COPY api.py ./
ENTRYPOINT ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8691"]
