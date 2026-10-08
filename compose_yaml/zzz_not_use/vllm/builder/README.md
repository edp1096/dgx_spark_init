## 이미지 빌드

* [builder/Dockerfile](builder/Dockerfile) - FlashInfer + vLLM 소스 빌드 (PyTorch는 NGC 이미지 포함분 사용), 첫 빌드 1~2시간

```sh
cd builder
docker build -t vllm-node .
```

* 캐시 무시하고 재빌드

```sh
docker build -t vllm-node \
  --build-arg CACHEBUST_VLLM=$(date +%s) \
  --build-arg CACHEBUST_FLASHINFER=$(date +%s) .
```

* 내꺼

```sh
docker build -t edp1096/vllm-node:v1 \
  --build-arg CACHEBUST_VLLM=$(date +%s) \
  --build-arg CACHEBUST_FLASHINFER=$(date +%s) .

docker push edp1096/vllm-node:v1
```
