# Qwen3.8 Flash-Next EXL3

단일 DGX Spark에서 `turboderp/Qwen3.8-Flash-Next-exl3` 4.05bpw를 실행한다.
공개 BF16 abliterated 모델의 rank-1 거부 방향을 준비 단계에서 복원하고, 실행 중
151개 residual writer와 동등한 위치에 λ=1.5 투영을 적용한다. EXL3 모델 파일은
수정하거나 재양자화하지 않는다.

```sh
cd /home/edp1096/workspace/dgx_spark_init/compose_yaml/exl3_qwen38fn
./manage.sh setup
./manage.sh start
```

첫 setup은 이미지를 빌드하고 방향 검증용 데이터 약 60MiB를 받은 뒤 모델 약
100GiB를 받는다. 기본 컨텍스트는 262,144토큰이고 API는
`http://서버-IP:8000/v1`이다. 설정은 `.env`에 있으며, 파일이 없으면
`env.sample`에서 자동 생성한다.

Flash-Next의 QSA attention은 ExLlamaV3 1.4.6에서 양자화 KV cache를 지원하지 않아
fp16 cache를 사용한다. 262K 기준 main cache 약 7.4GB와 MTP cache 약 0.5GB가 필요하다.

```sh
./manage.sh status
./manage.sh logs
./manage.sh stop
```

SparkTalk의 `low`, `medium`, `xhigh`, `none`이 Flash-Next의
`reasoning_effort`/`enable_thinking`으로 전달된다. 현재 내장 서버는 batch-1 텍스트와
도구 호출용이다. vision 가중치는 포함되지만 이미지 입력 파이프라인은 연결하지 않았다.

근거 모델은 `windowsxp811203/Qwen3.8-Flash-Next-Abliterated`이며, 준비 과정은 한
writer의 BF16 차이가 rank-1인지와 λ=1.5 결과의 잔류 비율이 0.5인지 검증한 뒤에만
방향 파일을 남긴다. 이는 그 모델의 공개 평가 결과를 참고한 실험적 런타임 패치이며,
EXL3 양자화 상태에서 동일한 거부율을 보장하는 별도 평가 결과는 아니다.

ExLlamaV3 본체는 Flash-Next 지원이 들어간 upstream 1.4.6을 고정한다. upstream의
x86 전용 CPU 소스가 ARM64에서 컴파일되지 않는 부분만 기존 27B에서 사용 중인
MiaAI ARM64 fallback 파일로 대체하며, 단일 GPU CUDA 추론 경로는 바꾸지 않는다.
