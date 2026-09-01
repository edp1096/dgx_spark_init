# Qwen3-ASR timestamp API

Gradio와 vLLM 없이 공식 `qwen-asr` Transformers wrapper를 OpenAI 호환
`POST /v1/audio/transcriptions` API로 제공합니다.

```bash
docker volume create media-hf-cache
docker compose build
docker compose up -d
curl http://127.0.0.1:8694/health
```

기본 모델은 `Qwen/Qwen3-ASR-1.7B`, 타임스탬프 모델은
`Qwen/Qwen3-ForcedAligner-0.6B`이며 한 번에 한 요청을 처리합니다. 응답의
`timestamps`에는 어절별 `text`, `start`, `end`(초)가 포함됩니다. 긴 미디어는
Media Access API에서 최대 180초 WAV로 나눠 요청합니다.
한 구간의 비정상 반복 생성을 제한하기 위해 최대 생성 토큰은 256으로 제한합니다.
