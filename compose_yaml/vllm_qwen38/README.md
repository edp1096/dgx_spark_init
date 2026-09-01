# vLLM - Huihui Qwen3.8 27B FP8 검증

로컬에서 변환한 `Huihui-Qwen3.8-27B-abliterated-FP8`을 vLLM으로
서빙하고 Open WebUI에서 대화하기 위한 DGX Spark용 구성이다. 디코딩에는
`Doopeworld/Qwen3.8-27B-DSpark-vLLM` drafter를 probabilistic 방식으로 사용한다.

## 실행

```sh
cd /home/edp1096/workspace/dgx_spark_init/compose_yaml/vllm_qwen38
docker compose up -d
docker compose logs -f vllm
```

vLLM 준비가 끝나면 Open WebUI가 자동으로 시작된다.

- Open WebUI: `http://서버-IP:12000` (`0.0.0.0:12000`에 바인딩)
- vLLM API: http://127.0.0.1:8000/v1
- 모델 이름: `Huihui-Qwen3.8-27B-abliterated-FP8`

Open WebUI는 vLLM의 `http://vllm:8000/v1`에 자동 연결되며, 별도 검증용
데이터 디렉터리 `~/.cache/openwebui-qwen38`을 사용한다. 검증 편의를 위해
로그인은 비활성화되어 있으므로 외부에 포트를 공개할 때는 주의한다.

### Qwen3.8 Flash Next

DGX Spark에서 `dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4`와 내장 MTP를
사용할 때는 전용 구성을 실행한다. 이 구성은 64K 컨텍스트와 동시 요청 2개를
기준으로 하며, 영상은 요청당 하나, 2fps·최대 96프레임으로 제한한다. 멀티모달
전처리 캐시는 0.5GiB로 줄여 TTS와 함께 실행할 메모리 여유를 남긴다.
KV 캐시는 동시 64K 요청 2개와 여유분에 맞춘 7GiB로 고정한다. 자동 비율로
할당하면 실제 동시 요청 제한보다 훨씬 큰 캐시가 예약되어 FLUX 이미지 엔진과
공존할 메모리가 사라지므로 임의로 `--gpu-memory-utilization`로 되돌리지 않는다.

```sh
docker compose -f compose.flash-next.yaml up -d
docker compose -f compose.flash-next.yaml logs -f vllm
```

모델과 PLE를 처음 적재할 때는 시간이 오래 걸릴 수 있다. 준비 여부는 다음으로
확인한다.

```sh
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

## 자동 스모크 테스트

vLLM 상태, 모델 목록, 실제 한국어 생성 요청을 한 번에 검사한다.

```sh
python3 smoke_test.py
```

에디터의 REST Client를 사용한다면 `request.http`로도 같은 검사를 할 수 있다.
스모크 테스트는 생성 요청 이후 `/metrics`에서 `vllm:spec_decode_*` 지표가
실제로 생성됐는지도 확인한다. 승인 토큰 수와 전체 draft 토큰 수의 비율은
다음과 같이 확인할 수 있다.

```sh
curl -s http://127.0.0.1:8000/metrics | grep 'vllm:spec_decode_'
```

DSpark는 원본 Qwen3.8 FP8을 대상으로 학습됐으므로 Huihui 모델에서도 결과의
정확성은 target 검증으로 보존되지만 승인률은 달라질 수 있다. 실제 요청에서
DSpark 적용 전후의 decode tok/s와 승인률을 비교해 가속 효과를 판단한다.

정식 Qwen3.8 FP8을 같은 조건으로 실행할 때는 override compose를 함께 지정한다.

```sh
docker compose down
docker compose -f compose.yaml -f compose.official.yaml up -d vllm
python3 benchmark_dspark.py --label official
```

## 종료

```sh
docker compose down
```

Flash-Next 구성의 컨텍스트 길이는 이미지·음성 엔진과의 공존을 검증한 64K로 설정했다.
모델의 네이티브 최대 길이는 262,144이며, 길이를 늘릴 때는 동시 요청 수와
GPU 메모리 사용률을 함께 조정해야 한다.
