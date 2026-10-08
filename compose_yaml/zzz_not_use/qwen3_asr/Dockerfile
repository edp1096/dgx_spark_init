FROM nvcr.io/nvidia/pytorch:26.07-py3

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libsndfile1 sox \
    && rm -rf /var/lib/apt/lists/*

# qwen-asr의 기본 의존성에는 Gradio가 포함되므로 API 실행에 필요한 패키지만 설치한다.
RUN python -m pip install --no-cache-dir \
    accelerate==1.12.0 \
    fastapi==0.136.3 \
    librosa \
    nagisa==0.2.11 \
    python-multipart==0.0.32 \
    qwen-omni-utils \
    soundfile==0.14.0 \
    sox \
    soynlp==0.0.493 \
    transformers==4.57.6 \
    uvicorn==0.51.0 \
    && python -m pip install --no-cache-dir --no-deps qwen-asr==0.0.6

WORKDIR /app
COPY api.py /app/api.py

EXPOSE 8694
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8694"]
