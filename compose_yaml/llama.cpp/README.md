## 이미지 실행

```sh
docker compose up -d
```


## 이미지 빌드

```sh
./build.sh
```

또는 수동:

```sh
TAG=$(curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/latest | grep -oP '"tag_name": "\K[^"]+')
TAG+="-v1"

docker build -t edp1096/llama.cpp-spark:$TAG .
docker push edp1096/llama.cpp-spark:$TAG
```
