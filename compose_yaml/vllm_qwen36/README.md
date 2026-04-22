# vLLM - Qwen3.6-35B-A3B-FP8

DGX Spark (GB10, sm_121a) 전용. eugr/spark-vllm-docker Dockerfile 기반.

## 빌드

```sh
cd builder
docker build -t edp1096/vllm-qwen36:v1 .
```

캐시 무시 재빌드:

```sh
docker build -t edp1096/vllm-qwen36:v1 \
  --build-arg CACHEBUST_VLLM=$(date +%s) \
  --build-arg CACHEBUST_FLASHINFER=$(date +%s) .
```

## 실행

```sh
cd single
docker compose up -d
docker compose logs -f vllm
```

## 참고

- 첫 빌드 1~2시간 (FlashInfer + vLLM 소스 컴파일), ccache 있으면 재빌드 빠름
- 모델은 HF cache에서 자동 다운로드, 사전 다운로드: `huggingface-cli download Qwen/Qwen3.6-35B-A3B-FP8`
