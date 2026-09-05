# EXL3 Qwen3.8 27B

단일 DGX Spark에서 `Lygodactylus/Qwen3.8-27B-Uncensored-exl3-4bpw`를
MTP와 NVFP4 KV cache로 실행한다. 기본 컨텍스트는 262,144토큰이며 API는
`http://서버-IP:8000/v1`이다.

```sh
cd /home/edp1096/workspace/dgx_spark_init/compose_yaml/exl3_qwen38_27b
./manage.sh setup
./manage.sh start
./manage.sh logs
```

첫 `setup`은 GB10 CUDA extension을 이미지에 컴파일하고 모델 약 16GB를 받는다.
설정 변경은 숨김 파일 `.env`에서 하며 기본값은 `env.sample`에 있다. 컨테이너를
내려도 모델과 컴파일 cache는 호스트에 남는다.

```sh
./manage.sh status
./manage.sh stop
```

요청별 thinking 제어는 `chat_template_kwargs.enable_thinking`을 사용한다. 서버는
batch-1이므로 동시 요청은 순서대로 처리된다.

## 이 장비 실측

- 262K MTP/NVFP4 준비 시간: 약 30초
- 정상 상주: 23.966GiB
- 모델 로딩 순간 피크: 24.094GiB
- 기존 4개 프롬프트, thinking off, 요청당 512토큰: 25.857 tok/s

Docker 통계는 통합 메모리의 CUDA 할당을 전부 표시하지 않으므로 `free -h`의 기동
전후 차이를 실제 상주량으로 본다.
