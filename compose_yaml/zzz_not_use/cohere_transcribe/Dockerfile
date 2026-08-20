FROM nvcr.io/nvidia/pytorch:26.07-py3

RUN python -m pip install --no-cache-dir \
    accelerate==1.12.0 \
    fastapi==0.136.3 \
    librosa \
    numpy \
    python-multipart==0.0.32 \
    soundfile==0.14.0 \
    transformers==5.14.1 \
    uvicorn==0.51.0

WORKDIR /app
COPY api.py /app/api.py

EXPOSE 8696
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8696"]
