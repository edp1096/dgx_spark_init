# SGLang Qwen3.8 27B + DSpark

DGX Spark에서 Qwen3.8 27B target과 RadixArk DSpark 조합을 실행한다.

- Draft: `RadixArk/Qwen3.8-27B-DSpark`
- SGLang image: `lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1`
- API: `http://127.0.0.1:8000/v1`
- Open WebUI: `http://서버-IP:12000`

## 전체 스택 최초 실행

명령은 이 디렉터리에서 실행한다.

```bash
cd ~/workspace/dgx_spark_init/compose_yaml/sgl_qwen38
```

Huihui FP8 target, DSpark, Open WebUI를 함께 실행한다.

```bash
docker compose \
  -f compose.yaml \
  -f compose.huihui-fp8-dspark-local.yaml \
  up -d
```

직접 양자화한 Huihui-RadixArk NVFP4 target, DSpark, Open WebUI를 함께 실행한다.

```bash
docker compose \
  -f compose.yaml \
  -f compose.huihui-radixark-nvfp4-local.yaml \
  up -d
```

Open WebUI는 SGLang이 healthy 상태가 된 뒤 기동된다.

## Open WebUI를 유지하며 모델만 전환

이미 전체 스택이 실행 중이라면 다음 명령으로 SGLang 컨테이너만 FP8 설정으로
교체한다.

```bash
docker compose \
  -f compose.yaml \
  -f compose.huihui-fp8-dspark-local.yaml \
  up -d --force-recreate sglang
```

NVFP4 설정으로 교체하려면 다음을 실행한다.

```bash
docker compose \
  -f compose.yaml \
  -f compose.huihui-radixark-nvfp4-local.yaml \
  up -d --force-recreate sglang
```

이 경우 기존 Open WebUI 컨테이너는 유지된다. Compose의 `-f` 옵션은 반드시 `up`
또는 `down`보다 앞에 둔다.

## 로그 및 준비 상태

기동 로그를 계속 확인한다.

```bash
docker logs -f sglang-qwen38
```

로그의 다음 문구가 나오면 엔진 초기화가 끝난 것이다.

```text
The server is fired up and ready to roll!
```

API 준비 상태를 직접 확인한다.

```bash
curl -fsS http://127.0.0.1:8000/health
```

준비될 때까지 반복해서 기다리려면 다음을 사용한다.

```bash
until curl -fsS http://127.0.0.1:8000/health; do
  sleep 5
done
echo "SGLang ready"
```

현재 제공 중인 모델 이름과 실제 생성을 검사한다.

```bash
curl -fsS http://127.0.0.1:8000/v1/models
python3 smoke_test.py
```

Open WebUI 상태도 함께 확인할 수 있다.

```bash
docker ps --format '{{.Names}}\t{{.Status}}' \
  | grep -E 'sglang-qwen38|open-webui-sgl-qwen38'
```

## 완전히 종료한 뒤 다시 실행

현재 FP8 구성을 완전히 내린다.

```bash
docker compose \
  -f compose.yaml \
  -f compose.huihui-fp8-dspark-local.yaml \
  down
```

현재 NVFP4 구성을 완전히 내린다.

```bash
docker compose \
  -f compose.yaml \
  -f compose.huihui-radixark-nvfp4-local.yaml \
  down
```

그 후 `전체 스택 최초 실행`에 있는 원하는 모델의 `up -d` 명령을 실행한다. 서비스
이름을 붙이지 않아야 Open WebUI도 함께 올라온다. 두 override를 한 명령에 동시에
지정하지 않는다.

## 2026-08-17 DGX Spark 단일 실행 실측

| 모델 | code_en | math_en | technical_ko | prose_ko | 전체 |
|---|---:|---:|---:|---:|---:|
| Radix NVFP4 + DSpark | 28.676 | 25.714 | 13.805 | 10.860 | 16.787 tok/s |
| Huihui NVFP4 + DSpark | 29.859 | 42.349 | 15.037 | 12.009 | 19.464 tok/s |

각 케이스는 thinking 비활성화, 최대 512 출력 토큰이다. 결과 원본은 `results/`에 있다.
한 번씩 실행한 비교이므로 절대 성능 보증이 아니라 현재 구성의 동작 검증 자료로 본다.

`behavior_test.py`에서 일반 질문과 천안문·시진핑 비판 질문 모두 거부하지 않는 것을
확인했다. 다만 천안문 답변에 고르바초프 방문 시점·호칭 논쟁 같은 부정확한 내용이
섞였으므로, abliterated 여부와 사실 정확성은 별개로 평가해야 한다.

기존 vLLM 구성과 8000/12000 포트를 공유하므로 두 구성을 동시에 실행하지 않는다.
