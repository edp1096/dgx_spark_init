# SGLang Qwen3.8 27B + DSpark

DGX Spark에서 Qwen3.8 27B target과 RadixArk DSpark 조합을 실행한다.

- Draft: `RadixArk/Qwen3.8-27B-DSpark`
- SGLang image: `lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1`
- API: `http://127.0.0.1:8000/v1`
- Open WebUI: `http://서버-IP:12000`

## 전체 스택 최초 실행

명령은 이 디렉터리에서 실행한다.

```bash
cd ~/workspace/dgx_spark_init/compose_yaml/sglang_qwen38
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

동일한 Huihui-RadixArk NVFP4 target에 DFlash2를 붙여 SGLang만 실행한다.
DFlash2 지원을 위해 upstream SGLang 고정 커밋을 기존 Qwen3.8 이미지 위에 적용한
전용 이미지를 자동으로 빌드한다.

```bash
docker compose \
  -f compose.yaml \
  -f compose.huihui-radixark-nvfp4-dflash2-local.yaml \
  up -d --build --force-recreate sglang
```

첫 실행에는 DFlash2 BF16 draft 약 3.85GB 다운로드와 target/draft CUDA graph
컴파일이 필요하다. DFlash는 `extra_buffer_lazy`를 지원하지 않으므로 전용 override는
Mamba radix cache를 `extra_buffer`로 사용한다.

SparkTalk 전용 32K 저메모리 프로필은 SparkMedia의 중량 모델을 모두 내린 뒤 다음처럼
실행한다. Open WebUI는 함께 올리지 않는다.

```bash
docker compose \
  -f compose.yaml \
  -f compose.huihui-radixark-nvfp4-dflash2-sparktalk.yaml \
  up -d --build --force-recreate sglang
```

이 프로필은 동시 요청 2, `mem-fraction-static=0.38`, FP8 target/draft KV,
Mamba cache 8슬롯, 32K context, CUDA graph batch 2를 사용한다. FlashInfer autotune을
끄고 GB10의 Cortex-X5 코어에 고정하며 idle scheduler를 재운다. SparkTalk의 live
설정은 `http://127.0.0.1:8000`과 제공 모델 ID를 가리켜야 한다.

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

NVFP4 DFlash2 설정으로 교체하려면 다음을 실행한다.

```bash
docker compose \
  -f compose.yaml \
  -f compose.huihui-radixark-nvfp4-dflash2-local.yaml \
  up -d --build --force-recreate sglang
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

## 트러블슈팅
`Error ... : Conflict. the container name ... is already in use ...`
`docker compose down`으로 내리지 못하고 재부팅하거나 여러 이유로 컨테이너가 살아있고 이 컨테이너를 재사용하기 싫은 경우
```sh
docker rm -f sglang-qwen38
```
* 컨테이너? 아몰랑~ 걍 실행
```sh
docker compose down --remove-orphans
docker compose up -d --force-recreate
# or
docker compose rm -sf && docker compose up -d --force-recreate --remove-orphans
```

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

## 2026-08-21 DFlash2 실측

Huihui-RadixArk Qwen3.8 27B abliterated NVFP4 target에서 동일한 4개 프롬프트와
thinking 비활성화, 요청당 최대 512토큰 조건으로 비교했다.

| 방식 | code_en | math_en | technical_ko | prose_ko | 전체 |
|---|---:|---:|---:|---:|---:|
| DSpark + compile | 27.072 | 37.483 | 15.058 | 11.181 | 18.226 tok/s |
| DFlash2 + compile | 40.983 | 58.978 | 27.392 | 14.533 | 27.272 tok/s |

DFlash2가 전체 기준 약 49.6% 빨랐다. accept length는 코드 4.65, 수학 6.75,
기술 한국어 6.75, 한국어 산문 2.575였다. DFlash2 기동 후 유휴 GPU 프로세스
할당은 약 52.5GiB였으며 텍스트와 이미지 입력을 모두 확인했다. 원본 결과는
`results/huihui-radixstyle-nvfp4-dflash2-compile-no-thinking.jsonl`에 있다.

## 2026-08-24 SparkTalk 32K 저메모리 프로필 실측

SparkMedia 컨테이너와 TensorRT Edge-LLM을 내리고 ASR, TTS, SparkTalk Extra만 남긴
상태에서 측정했다. SGLang 기동 직전 가용 메모리는 98.57GiB, 준비 직후는 약
55.6GiB로 SGLang 추가 상주는 약 43GiB였다. 31,912-token 동일 입력은 첫 요청
20.26초, 두 번째 요청 0.43초였으며 prefix-cache hit rate는 99.87%였다.

| 방식 | code_en | math_en | technical_ko | prose_ko | 전체 |
|---|---:|---:|---:|---:|---:|
| SparkTalk 32K DFlash2 | 38.456 | 54.749 | 22.413 | 12.946 | 24.077 tok/s |

WebP vision 요청은 4.37초에 완료됐고 SSE 41개 chunk, reasoning/content 분리도
정상 동작했다. 장문 대화의 체감 응답성은 TensorRT Edge-LLM DFlash가 아직 지원하지
않는 prefix reuse에서 큰 차이가 난다. 대신 이 프로필도 TensorRT 구성보다 약
6~9GiB 더 상주하므로 SparkMedia 중량 모델 전체와 동시에 올리는 기본 구성으로는
사용하지 않는다.
