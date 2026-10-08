FROM nvcr.io/nvidia/pytorch:26.07-py3

RUN python -m pip install --no-cache-dir \
      fastapi==0.116.1 \
      huggingface-hub==0.34.4 \
      pillow==11.3.0 \
      python-multipart==0.0.20 \
      safetensors==0.6.2 \
      transformers==4.56.2 \
      uvicorn==0.35.0

WORKDIR /opt/garment-extractor
COPY api.py ./api.py

EXPOSE 8705
ENTRYPOINT ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8705"]
