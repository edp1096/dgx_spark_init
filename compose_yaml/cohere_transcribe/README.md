# Cohere Transcribe native API

공식 권장 오프라인 경로인 Transformers native pipeline을 OpenAI 호환
`POST /v1/audio/transcriptions` API로 제공합니다. 한국어는 요청의 `language=ko`로
명시해야 하며 자동 언어 감지는 지원하지 않습니다.

모델은 Hugging Face에서 접근 조건에 동의한 토큰이 필요합니다.

```bash
docker volume create media-hf-cache
HF_TOKEN=... docker compose build
HF_TOKEN=... docker compose up -d
curl http://127.0.0.1:8696/health
```

기본 모델은 `CohereLabs/cohere-transcribe-03-2026`이며 한 번에 한 요청을
처리합니다.
