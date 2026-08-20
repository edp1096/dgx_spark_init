# DGX Spark vLLM-Omni runtime

공식 ARM64 `vllm/vllm-omni:v0.26.0`을 기반으로 서빙에 불필요한 소스 체크아웃,
문서 및 UI 예제를 제외한 공통 런타임 이미지입니다.

```bash
docker compose --profile check build
docker compose --profile check run --rm runtime-check
```

현재 `qwen3_tts`가 이 이미지를 사용합니다. `flux2_klein_nvfp4`는 별도의
ComfyUI 기반 이미지이므로 vLLM-Omni를 사용하지 않습니다. 이 compose의
`runtime-check`는 런타임 이미지를 빌드하고 CUDA·vLLM-Omni import를 확인하는
용도이며, 포트를 열어 상시 실행하는 API 서비스가 아닙니다.
