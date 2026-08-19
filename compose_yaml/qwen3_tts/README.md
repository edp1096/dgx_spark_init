# Qwen3-TTS vLLM-Omni API

별도 웹 UI 없이 vLLM-Omni의 OpenAI 호환 음성 API만 실행합니다.
프로젝트에서 사용하는 프리셋 화자용 CustomVoice만 운영합니다.
공통 런타임 이미지는 먼저 `../vllm_omni`에서 빌드합니다.

```bash
docker compose -f ../vllm_omni/compose.yaml build
docker volume create media-hf-cache
docker compose up -d custom
curl http://127.0.0.1:8692/health
```

생성 API는 `POST /v1/audio/speech`입니다. 기본 모델은
`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`입니다.
선택적인 `instructions` 필드로 감정, 말투, 강조와 같은 발화 스타일을 제어합니다.
선택적인 음이 아닌 정수 `seed`로 동일 조건의 생성을 재현할 수 있습니다.
Stage 0은 요청별 seed를 유지하면서 정적 연산을 가속하도록 `PIECEWISE` CUDA Graph를 사용합니다.
모델은 `media-hf-cache` Docker 볼륨에 저장됩니다.
