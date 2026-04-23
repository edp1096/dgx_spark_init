# vLLM - Gemma-4-26B-A4B NVFP4

DGX Spark (GB10, sm_121a) 전용. AEON-7 커뮤니티 이미지 기반.

- 모델: `RedHatAI/gemma-4-26B-A4B-it-NVFP4` (compressed-tensors NVFP4, ~16GB)
- MoE: 128 experts, top-8 routing, 4B active params
- 예상 성능: Single ~45 tok/s, Dual ~55+ tok/s


## 사전 준비 (선택)

모델 사전 다운로드:
```sh
huggingface-cli download RedHatAI/gemma-4-26B-A4B-it-NVFP4
```


## 실행

### Single 노드

```sh
cd single
docker compose up -d
docker compose logs -f vllm
```

### Cluster (2-Spark, TP=2)

Worker 먼저:
```sh
# worker 노드에서
cd cluster/worker
docker compose up -d
```

Head:
```sh
# head 노드에서
cd cluster/head
docker compose up -d
```


## 모델 변경

compose.yaml에서 모델 ID 교체:

| 모델 | 설명 |
|------|------|
| `RedHatAI/gemma-4-26B-A4B-it-NVFP4` | RedHat NVFP4 (기본) |
| `AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4` | Uncensored NVFP4 |


## 참고

- `gpu-memory-utilization` 0.80 초과 금지 (DGX Spark 시스템 프리즈 위험)
- `--quantization` 플래그 없음 — compressed-tensors 포맷은 자동 감지
- `--quantization modelopt`는 nvidia NVFP4 전용, RedHat compressed-tensors와 다름
- cluster 미테스트


## 출처

- https://github.com/ZengboJamesWang/dgx-spark-vllm-gemma4-26b-uncensored
- https://github.com/eugr/spark-vllm-docker
- https://forums.developer.nvidia.com/t/46tok-s-with-redhatai-gemma-4-26b-a4b-it-nvfp4/365870
- https://forums.developer.nvidia.com/t/gemma-4-day-1-inference-on-nvidia-dgx-spark-preliminary-benchmarks/365503
