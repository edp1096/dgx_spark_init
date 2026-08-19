# Qwen3-ASR native API

Gradio와 vLLM 없이 Transformers native pipeline을 OpenAI 호환
`POST /v1/audio/transcriptions` API로 제공합니다.

```bash
docker volume create media-hf-cache
docker compose build
docker compose up -d
curl http://127.0.0.1:8694/health
```

기본 모델은 `Qwen/Qwen3-ASR-1.7B-hf`이며 한 번에 한 요청을 처리합니다.
업로드는 libsndfile이 읽을 수 있는 WAV, FLAC, OGG 등의 음성 파일을 지원합니다.
