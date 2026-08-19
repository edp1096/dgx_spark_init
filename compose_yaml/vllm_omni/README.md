# DGX Spark vLLM-Omni runtime

공식 ARM64 `vllm/vllm-omni:v0.26.0`을 기반으로 서빙에 불필요한 소스 체크아웃,
문서 및 UI 예제를 제외한 공통 런타임 이미지입니다.

```bash
docker compose --profile check build
docker compose --profile check run --rm runtime-check
```

`flux2_klein`과 `qwen3_tts`가 이 이미지 하나를 공통으로 사용합니다.
