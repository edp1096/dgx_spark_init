## 이미지 빌드

* [builder/Dockerfile](builder/Dockerfile) - FlashInfer + vLLM 소스 빌드 (PyTorch는 NGC 이미지 포함분 사용), 첫 빌드 1~2시간

```sh
cd builder
docker build -t vllm-node .
```
