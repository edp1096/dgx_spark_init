# Nemotron 3.5 ASR native API

Transformers native offline pipeline을 OpenAI 호환
`POST /v1/audio/transcriptions` API로 제공합니다. 한국어는 `language=ko` 또는
`language=ko-KR`로 지정할 수 있고 `language=auto`도 지원합니다.

```bash
docker volume create media-hf-cache
docker compose build
docker compose up -d
curl http://127.0.0.1:8697/health
```

기본 모델은 `nvidia/nemotron-3.5-asr-streaming-0.6b`이며 한 번에 한 요청을
처리합니다. 현재 API는 정확도·처리량 비교를 위한 오프라인 추론 경로입니다.
