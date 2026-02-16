## 이미지 실행

```sh
cd runtime
docker compose up -d
```


## 이미지 빌드

```sh
TAG=$(curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/latest | grep -oP '"tag_name": "\K[^"]+')
TAG+="-v1"

cd builder
docker compose up -d
docker commit llama-cpp-spark edp1096/llama.cpp-spark:$TAG
docker compose down -v

docker push edp1096/llama.cpp-spark:$TAG
```
