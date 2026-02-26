## whl, image 직접 빌드

* [builder/Dockerfile](builder/Dockerfile) - 1~2시간 걸림
* 
* 빌드
```sh
cd builder

VLLM_VERSION=v0.16.0
docker build -t edp1096/vllm-spark:${VLLM_VERSION}-v1 --build-arg VLLM_VERSION=${VLLM_VERSION} --build-arg MAX_JOBS=20 .
```
