## 이미지 실행

```sh
cd runtime
docker compose up -d
```


## 이미지 빌드

```sh
cd builder
docker compose up -d
docker commit llama-cpp-spark edp1096/llama.cpp-spark:b8061-v1
docker compose down -v

docker login -u edp11096
docker push edp1096/llama.cpp-spark:b8061-v1
```
