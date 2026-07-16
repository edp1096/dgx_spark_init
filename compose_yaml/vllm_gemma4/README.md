# vLLM - Gemma 4 NVFP4

DGX Spark (GB10, sm_121a) 전용. `eugr/spark-vllm-docker`의 TF5 빌드(`vllm-node-tf5`) 기반.

- 기본 모델: `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4`
- 소형 모델: `bg-digitalservices/Gemma-4-E4B-it-NVFP4`
- 26B A4B: MoE, 25.2B total / 3.8B active, 256K context, modelopt NVFP4
- E4B: Dense + PLE, 128K context, modelopt NVFP4


## vLLM 이미지 준비

`spark-vllm-docker`를 head 노드에서 빌드한다.

```sh
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
./build-and-copy.sh --tf5
```

클러스터까지 이미지 배포:

```sh
./build-and-copy.sh --tf5 -c
```

기본 compose는 로컬 이미지 `vllm-node-tf5:latest`를 사용한다. 다른 태그를 쓸 때:

```sh
VLLM_IMAGE=my-vllm-node-tf5:latest docker compose up -d
```


## 모델 사전 다운로드 (선택)

```sh
huggingface-cli download bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4
huggingface-cli download bg-digitalservices/Gemma-4-E4B-it-NVFP4
```


## 실행

### Single 노드

```sh
cd single
docker compose up -d
docker compose logs -f vllm
```

E4B:

```sh
cd single
docker compose --profile e4b up -d vllm-e4b
docker compose logs -f vllm-e4b
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

기본 compose는 26B A4B를 실행한다. E4B는 `single`의 `vllm-e4b` profile로 실행한다.

| 모델 | 설명 |
|------|------|
| `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` | MoE, 256K, 26B total / 3.8B active, 기본 |
| `bg-digitalservices/Gemma-4-E4B-it-NVFP4` | Dense + PLE, 128K, 가벼운 단일 노드용 |


## 참고

- `gpu-memory-utilization` 0.80 초과 금지 (DGX Spark 시스템 프리즈 위험)
- `bg-digitalservices` 모델은 NVIDIA Model Optimizer 포맷이므로 `--quantization modelopt` 사용
- 26B A4B는 MoE expert scale key 로딩 패치가 필요해서 `patches/gemma4_patched.py`를 컨테이너 내부 `vllm/model_executor/models/gemma4.py`에 덮어쓴다.
- 26B A4B는 `VLLM_NVFP4_GEMM_BACKEND=FLASHINFER_CUTLASS`, `--moe-backend marlin`, `--attention-backend TRITON_ATTN`, `--kv-cache-dtype fp8` 조합을 사용한다.
- E4B는 모델 카드 기준으로 vanilla vLLM에서 패치 없이 동작한다.
- cluster는 미테스트


## 출처

- https://github.com/eugr/spark-vllm-docker
- https://huggingface.co/bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4
- https://huggingface.co/bg-digitalservices/Gemma-4-E4B-it-NVFP4
