# vLLM · Qwen3.8 27B FP8 검증

로컬에서 변환한 `Huihui-Qwen3.8-27B-abliterated-FP8`을 vLLM으로
서빙하고 Open WebUI에서 대화하기 위한 DGX Spark용 구성이다. 디코딩에는
`Doopeworld/Qwen3.8-27B-DSpark-vLLM` drafter를 probabilistic 방식으로 사용한다.

## 실행

```sh
cd /home/edp1096/workspace/dgx_spark_init/compose_yaml/vllm_qwen38_27b
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

Qwen3.8 Flash-Next의 vLLM 런타임과 GB10 전용 이미지 재구축 자료는
`../vllm_qwen38fn`에 완전히 분리되어 있다.
