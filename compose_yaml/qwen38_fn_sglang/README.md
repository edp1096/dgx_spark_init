# Qwen3.8 Flash-Next · 독립 SGLang TP1 Compose

이 디렉터리만으로 모델 다운로드·검증·이미지 빌드·실행을 할 수 있습니다.
Talk 설치, Talk YAML, `util/talk` 소스나 변환 작업 디렉터리는 필요하지 않습니다.
Docker Compose와 NVIDIA GPU 런타임이 준비된 DGX Spark 1대를 사용합니다.
런타임 이미지는 `dgx-sglang-qwen38-qad:sm121-v5-memory`입니다.

## 모델 선택

| 선택 파일 | 모델 | 고정 리비전 |
| --- | --- | --- |
| `model.official.env` | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` (기본값) | `7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd` |
| `model.abliterated.env` | `huginnfork/Qwen3.8-Flash-Next-NVFP4-Abliterated` | `93a1b466ce773185f21a49d1649b7933ce0fc910` |
| `model.huihui-lil.env` | `edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD` | `d8a260c0792cf2db93c5aaf8d397fd837b69901b` · **step-5500** |

Huihui는 [공개된 step-5500 체크포인트](https://huggingface.co/edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD/commit/d8a260c0792cf2db93c5aaf8d397fd837b69901b)를 내려받습니다.
별도 로컬 변환 작업은 필요하지 않습니다. 모델 ID·리비전·컨테이너 모델 경로는
선택한 env 파일 하나로 지정하며, 다운로드와 런타임에 동일하게 전달합니다.

## 준비·다운로드·빌드

**Dockerfile만 받지 말고 이 디렉터리 전체를 준비**한 뒤 여기에서 실행합니다.
아래 예시는 Huihui step-5500입니다. 다른 모델은 env 파일만 바꿉니다.

```sh
cd compose_yaml/qwen38_fn_sglang
export QWEN_UID="$(id -u)" QWEN_GID="$(id -g)"
export QWEN_HF_CACHE="${QWEN_HF_CACHE:-$HOME/.cache/huggingface}"
mkdir -p "$QWEN_HF_CACHE"
docker compose --env-file model.huihui-lil.env build model-prepare
docker compose --env-file model.huihui-lil.env run --rm model-prepare
docker compose --env-file model.huihui-lil.env build sglang
```

`model-prepare`는 GPU 없이 실행하는 이 디렉터리 자체 다운로드 도구입니다.
HF CLI나 호스트 Python 패키지 설치도 필요하지 않습니다. 런타임을 `up`할 때도
먼저 준비 서비스가 성공해야 GPU 서비스가 기동됩니다.

Huihui 파일 위치는 `$QWEN_HF_CACHE/edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD`입니다.
`checkpoint.huihui-lil.json`에 고정한 config·가중치 인덱스·양자화 설정·변환 기록의
SHA-256과 인덱스에 포함된 모든 safetensors 파일의 구조·크기를 확인합니다.
이전 리비전의 파일이 남아 있거나 샤드가 누락·잘린 상태를 준비 완료로 처리하지 않습니다.
검증된 step-5500이 이미 있으면 네트워크 없이 재사용합니다.
다운로드 없이 확인만 하려면:

```sh
docker compose --env-file model.huihui-lil.env run --rm model-prepare --check
```

`env.sample`의 캐시 위치를 바꾸려면 해당 변수를 export하거나,
`--env-file .env --env-file model.huihui-lil.env`처럼 두 설정 파일을 지정합니다.
독립 런타임 캐시 기본값은 `~/.local/share/qwen38-fn/`입니다.
기존 `SPARKTALK_HF_CACHE`·`SPARKTALK_DATA_DIR` 변수는 명시적으로 설정된 경우에만
호환용 대체 값으로 사용하며, Talk 서비스나 설정 파일을 읽지 않습니다.

## 실행

| 구성 | 컨텍스트 | 동시 요청 | MTP |
| --- | --- | --- | --- |
| 기본 `compose.yaml` | 64K | 2 | 사용 |
| `compose.context1m.yaml` 추가 | 1M | 1 | 사용 · LLM 우선 |
| `compose.context1m.shared.yaml` 추가 | 1M | 1 | 끔 · 과거 동시 상주용 대안 |

**기본 64K — Huihui step-5500:**

```sh
docker compose --env-file model.huihui-lil.env up -d sglang
```

**1M + MTP — Huihui step-5500:**

```sh
docker compose --env-file model.huihui-lil.env \
  -f compose.yaml -f compose.context1m.yaml up -d sglang
```

현재 권장 구성은 위의 **1M + MTP**입니다. `shared` 파일은 과거에 여러 GPU
서비스를 동시 상주시킬 때 사용한 MTP-off 대안이며, 현재 Talk의 작업 교대
방식과는 별개입니다. 이를 명시적으로 선택하려는 경우에만 추가 파일을 `compose.context1m.shared.yaml`로
바꿉니다. 두 1M 파일을 동시에 넣지 않습니다. 기본 구성으로 돌아가려면 추가
`-f` 없이 실행합니다. 이 Compose는 Qwen과 CPU 모델 준비 서비스만 관리합니다.
다른 GPU 서비스의 실행 순서·메모리 확보는 별도로 관리합니다.

## 확인·종료

```sh
docker compose --env-file model.huihui-lil.env logs -f sglang
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/server_info
docker compose --env-file model.huihui-lil.env down
```

API 요청의 `model`에는 선택한 모델 ID를 넣습니다. 첫 기동은 적재·컴파일로
수 분 걸립니다. 1M 구성은 `/server_info`의 `max_total_num_tokens`가 실제로
`1048576` 이상인지 확인합니다. 컨텍스트 상한과 할당된 KV 용량은 다릅니다.

기본 MTP는 한국어 초안 어휘 `ko64k`를 사용합니다. 전체 어휘를 쓰려면:

```sh
SPARKTALK_FLASH_NEXT_DRAFT_VOCAB=off docker compose --env-file model.huihui-lil.env up -d sglang
```

공유 1M 구성은 MTP와 초안 어휘 제한을 모두 끕니다.

## Cached RadixArk TP1, 1M KV

SparkTalk의 `Qwen 3.8 Flash-Next NVFP4` 세트는 캐시의
`edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4`를 사용한다.
LLM, 요청 시 실행하는 Nemotron ASR, Extra Media·SSH·Collector·Documents를 포함하며 이미지·TTS는 포함하지 않는다. FP8 KV와 실제 context/KV
상한은 모두 1,048,576토큰, MTP는 3단계다. GPU SGLang의 standard
`modelopt_fp4` 경로로 실행하며, QAD 전용 GDN 경로는 끈다.
기존 `dgx-sglang-qwen38-qad:sm121-v5-memory` 이미지는 두 형식을 지원한다.
ASR을 요청 시 실행하는 이 세트에서는 `mem-fraction-static=0.92`를 사용한다. 공용 0.86 예약 비율은
이 체크포인트의 MTP 로딩 뒤 KV 풀을 약 632K로 제한하므로 사용하지 않는다.
`max-total-tokens` 상한은 1M이며 Talk는 실제 풀 용량이 1M보다 작으면 기동을 실패 처리한다.

독립 Compose 구성은 `compose.yaml`과 `compose.radixark1m.yaml`을 함께 사용한다:

```sh
docker compose -f compose.yaml -f compose.radixark1m.yaml up -d --no-build
```

모델 준비는 네트워크 없이 캐시의 pinned metadata와 모든 206개 shard의
safetensors 구조/전체 길이를 확인한다. 캐시가 불완전하면 실패하며 모델을
다운로드하지 않는다. `checkpoint.radixark.json`이 release identity다.
`radixark_launch.py`는 MTP index의 3개 shard만 draft loader에 전달한다.
QAD 세트와 TP2 세트의 설정은 별개다.

2026-10-08 실기동 검증: `/server_info`의 context와 실제 KV pool은 모두
1,048,576토큰, FP8 KV/QSA는 13.8134GiB였다. 한국어 한 문장, 덧셈,
강제 function tool call을 같은 컨테이너/PID로 처리했다. OOM과 재시작은 없었다.
CUDA 점유 95.27GiB + CUDA 외 host 점유 6.03GiB = 상주 101.31GiB,
검증 후 시스템 가용 14.37GiB, 로딩·검증 중 최저 가용 9.46GiB였다.
짧은 요청의 기동·응답·메모리 검증이며 1M 전체 입력의 장문 품질 시험은 아니다.

2026-10-08 Extra 구성: NVFP4 세트는 Extra Media·SSH·Collector·Documents를
사용한다. Media·Collector·Documents는 공통 작업 큐와 메모리 확인을 거쳐 요청 시
시작하며, SSH는 기존 공용 서비스를 사용한다. WAV 변환, 실제 Chromium 웹 수집,
PDF 생성과 SSH API를 확인했고 LLM 컨테이너/PID 및 실제 1M KV가 유지됐다.
반복 Extra 시험 중 최저 시스템 가용 메모리는 13.19GiB였다.

NVFP4 세트의 Nemotron 3.5 ASR(Q5_K, 화자 구분 Q8_0)은 요청 시 GPU로 시작한다.
Extra Media가 영상에서 16kHz mono PCM을 추출한 뒤 입력 길이별 ASR 작업 예산을
확인한다. 새 ASR CUDA 컨텍스트 기동 전에는 SGLang을 in-place pause/resume하여
사용하지 않는 PyTorch allocator 블록만 반환한다. 본체, 1M KV 풀과 prefix 상태는
유지하며, 즉시 여유 6GiB와 작업 후 최소 여유를 각각 검사한다.
실제 MP4의 한국어 전사·화자 구분·전사 캐시 재사용 및 동일 LLM PID/1M KV를 확인했다.

운영 Talk의 실제 `/api/asr/transcribe`에서도 MP4 오디오 전사를 검증했다.
자동 정리를 사용하는 새 ASR 기동 중 가용 메모리는 최저 12.22GiB,
완료 후 약 12.02GiB였고 LLM의 컨테이너/PID 및 실제 1M KV가 유지됐다.

## TP2 이미지 재생성

새 설치에서도 공개 베이스와 고정 소스에서 TP2 이미지를 만들 수 있다.
이 디렉터리에서 다음을 실행한다. 모델 가중치는 빌드에 포함하지 않는다.

```bash
docker build -f Dockerfile.tp2-first-install --target tp2 \
  -t dgx-sglang-qwen38-fn:sm121-tp2-vocab-v1 .
```

Talk에 포함된 TP2 준비 레시피도 같은 Dockerfile과 패치로 빌드한 뒤 워커에
동일 이미지를 전송한다.
