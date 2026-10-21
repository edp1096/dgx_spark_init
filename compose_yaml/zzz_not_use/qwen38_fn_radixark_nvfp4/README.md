# Archived TP1 RadixArk NVFP4

Removed from the current Talk model-set list. This standalone recipe is retained for reference. The separate TP2 model set still shares the RadixArk checkpoint.

# Qwen Flash-Next: RadixArk and TP2

The QAD standalone recipe is archived at [qwen38_fn_qad](../qwen38_fn_qad/README.md). The active TP2 recipe and its build context remain at [qwen38_fn_sglang](../../qwen38_fn_sglang/README.md). The following notes describe the retired TP1 deployment.

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

독립 Compose 구성은 `compose.radixark-base.yaml`과 `compose.radixark1m.yaml`을 함께 사용한다:

```sh
docker compose -f compose.radixark-base.yaml -f compose.radixark1m.yaml up -d --no-build
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

NVFP4에 ASR·의미 검색을 함께 상주시킬 때는 ASR을 먼저 기동한다.
Talk는 이 순서와 메모리 검사를 적용한다. 독립 Compose는 아래 구성을 사용한다.

```bash
docker compose -f compose.radixark-base.yaml -f compose.radixark1m.yaml -f compose.radixark-resident.yaml up -d --no-build
```

Nemotron ASR Q5_K·화자 구분 Q8_0와 EmbeddingGemma 2 BF16 가중치는 미리 준비한다.
이미지·TTS는 이 NVFP4 구성에 포함하지 않는다.

## TP2 이미지 재생성

새 설치에서도 공개 베이스와 고정 소스에서 TP2 이미지를 만들 수 있다.
이 디렉터리에서 다음을 실행한다. 모델 가중치는 빌드에 포함하지 않는다.

```bash
docker build -f Dockerfile.tp2-first-install --target tp2 \
  -t dgx-sglang-qwen38-fn:sm121-tp2-vocab-v1 .
```

Talk에 포함된 TP2 준비 레시피도 같은 Dockerfile과 패치로 빌드한 뒤 워커에
동일 이미지를 전송한다.
