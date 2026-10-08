# TensorRT Edge-LLM on DGX Spark

DGX Spark에서 `Qwen3.8-27B NVFP4 + DFlash2 DDTree + vision encoder`를 OpenAI 호환
API로 실행하는 구성이다. TensorRT Edge-LLM은 v0.10.0의 정확한 커밋
`71dd1bae032e70771265917ec74d3ff4cad07a10`에 고정한다. Dockerfile에 이 SHA를
직접 기록했으며 Compose 환경변수로 다른 ref를 주입할 수 없다.

기본 target은 Huihui abliterated NVFP4, draft는 incoai DFlash2, 문맥 길이는
32K다. 분기형 DDTree의 draft Top-K는 실측이 가장 좋았던 2를 사용한다. Huihui와
원본 RadixArk target에 DSpark를 붙인 엔진도 fallback으로 보존한다. Qwen3.8의
DFlash2와 DSpark는 현재 NVIDIA 공식 검증 조합이 아니므로 이 저장소의 호환
패치와 실측 결과를 전제로 사용한다.

## 실행

ONNX와 TensorRT 엔진은 용량이 크고 장치 종속적이므로 `workspace/`에 두며 Git에
포함하지 않는다. 변환된 기본 Huihui 32K 엔진이 있으면 다음 명령만 필요하다.

```bash
cd /home/edp1096/workspace/dgx_spark_init/compose_yaml/trt-edge-llm
docker compose up -d --no-build server
curl -fsS http://127.0.0.1:8696/v1/models
```

Huihui DSpark 엔진으로 되돌릴 때는 기존 산출물을 다시 만들 필요가 없다.

```bash
TRT_EDGE_WORKSPACE_DIR=./workspace/dspark-huihui \
TRT_EDGE_ENGINE_SUBDIR=dspark-huihui-32k \
TRT_EDGE_MULTIMODAL_ENGINE_SUBDIR=dspark-huihui-32k \
TRT_EDGE_DRAFT_TOP_K=1 \
TRT_EDGE_VERIFY_TREE_SIZE=8 \
docker compose up -d --no-build --force-recreate server
```

런타임은 `nvcr.io/nvidia/pytorch:26.07-py3`의 TensorRT 11.1/CUDA 13.3을
사용한다. TensorRT 10.14에서는 speculative base engine 빌드가 `PLUGIN_V3`
optimizer 오류로 실패했다.

## 변환 및 엔진 빌드

먼저 exporter와 runtime 이미지를 만든다.

```bash
docker compose --profile tools build exporter runtime
```

기본 Huihui NVFP4 target + incoai DFlash2 DDTree 32K:

```bash
./scripts/build-dflash.sh all
./scripts/build-dflash.sh vision
```

`build-dflash.sh`는 `all`, `export`, `base-export`, `draft-export`, `build`,
`base-build`, `draft-build`, `vision` 단계를 지원한다. Top-K 2 이상의 분기형
검증에는 base ONNX를 반드시 `--dflash-tree-base`로 export해야 한다. 이 스크립트는
이를 기본 적용하고 base/draft tree 크기를 16으로 빌드한다.

원본 RadixArk NVFP4 target + RadixArk DSpark 32K fallback:

```bash
TRT_EDGE_MAX_INPUT_LEN=32768 \
TRT_EDGE_MAX_KV_CACHE_CAPACITY=32768 \
TRT_EDGE_ENGINE_SUBDIR=dspark-radixark-32k \
./scripts/build-dspark.sh all

TRT_EDGE_MAX_INPUT_LEN=32768 \
TRT_EDGE_MAX_KV_CACHE_CAPACITY=32768 \
TRT_EDGE_ENGINE_SUBDIR=dspark-radixark-32k \
./scripts/build-dspark.sh vision
```

Huihui abliterated NVFP4 target + RadixArk DSpark 32K:

```bash
TRT_EDGE_DSPARK_TARGET_REPO=/home/edp1096/workspace/heretic_models/Huihui-RadixArk-Qwen3.8-27B-abliterated-NVFP4 \
TRT_EDGE_DSPARK_WORKSPACE_DIR="$PWD/workspace/dspark-huihui" \
TRT_EDGE_MAX_INPUT_LEN=32768 \
TRT_EDGE_MAX_KV_CACHE_CAPACITY=32768 \
TRT_EDGE_ENGINE_SUBDIR=dspark-huihui-32k \
./scripts/build-dspark.sh all

TRT_EDGE_DSPARK_TARGET_REPO=/home/edp1096/workspace/heretic_models/Huihui-RadixArk-Qwen3.8-27B-abliterated-NVFP4 \
TRT_EDGE_DSPARK_WORKSPACE_DIR="$PWD/workspace/dspark-huihui" \
TRT_EDGE_MAX_INPUT_LEN=32768 \
TRT_EDGE_MAX_KV_CACHE_CAPACITY=32768 \
TRT_EDGE_ENGINE_SUBDIR=dspark-huihui-32k \
./scripts/build-dspark.sh vision
```

`build-dspark.sh`는 `all`, `export`, `base-export`, `draft-export`, `build`,
`base-build`, `draft-build`, `vision` 단계를 지원한다. Hugging Face 캐시의
snapshot symlink는 exporter가 안전하게 읽도록 작업공간에 일반 파일로
staging한다.

## v0.10.0 호환 패치와 제약

- Qwen3.8 hybrid recurrent state와 DSpark projector를 exporter와 C++ runtime에
  연결한다.
- 체크포인트에 내장된 혼합 정밀도 메타데이터를 해석해 Float8/Half 충돌을
  방지한다.
- 공개 DSpark 체크포인트에 없는 `lm_head`를 무작위로 만들지 않고 target의
  NVFP4 head를 공유한다. 이 수정 전 평균 수용 토큰은 약 1.05였지만 수정 후
  3.24~3.32가 됐고 draft engine도 약 5.27GB에서 3.45GB로 줄었다.
- speculative decoding과 vision encoder를 함께 연결한다.
- SparkTalk의 OpenAI `image_url` data URL과 허용된 로컬 이미지 경로를 C++
  런타임 입력으로 전달한다. 기본 디코더가 직접 받지 못하는 WebP는 서버
  어댑터에서 무손실 PNG로 변환한다.
- Qwen reasoning을 SSE delta로 스트리밍하고 OpenAI `reasoning_effort`를
  `none`, `low`, `medium`, `xhigh` 단계에 연결한다.
- speculative decoding이 활성화되면 Edge-LLM runtime은 temperature, top-k,
  top-p를 무시하고 greedy sampling으로 강제한다. SparkTalk의 reasoning 단계와
  SSE 스트리밍은 정상 동작하지만 확률적 샘플링이 필요하면 speculative engine이
  아닌 별도 engine 또는 SGLang 계열 runtime을 사용해야 한다.

패치는 위의 정확한 v0.10.0 커밋을 기준으로 한다. 후속 릴리즈로 기준을 바꾸면
`git apply --check`가 빌드를 중단하므로 패치를 명시적으로 재검토해야 한다.

## 2026-08-24 DGX Spark 실측

통합 메모리는 컨테이너의 RSS나 `nvidia-smi` 합계가 아니라 호스트
`/proc/meminfo`의 `MemAvailable` 감소량과 요청 중 최솟값으로 측정했다.

| 항목 | 결과 |
|---|---:|
| Huihui DFlash2 DDTree 32K + vision 상주 | 약 36.7 GiB |
| Huihui DSpark 32K + vision 상주 | 약 36.55 GiB |
| 원본 RadixArk DSpark 32K + vision 상주 | 약 36.57 GiB |
| 이전 SGLang NVFP4 + DFlash2 상주 참고값 | 약 52.5 GiB |
| DFlash2 DDTree Top-K 2 native · 107 input + 256 output | 25.0 tok/s · 수용 2.88 |
| DFlash2 DDTree Top-K 4 native | 24.3 tok/s · 수용 2.81 |
| DFlash2 DDTree Top-K 8 native | 23.5 tok/s · 수용 2.72 |
| Huihui native C++ · 107 input + 256 output | 17.1 tok/s |
| Huihui DSpark 평균 수용 토큰 | 3.32 |
| 원본 native C++ · 107 input + 256 output | 16.7 tok/s |
| 원본 DSpark 평균 수용 토큰 | 3.24 |
| Huihui OpenAI HTTP · 256 output | 10.58 tok/s |
| DFlash2 OpenAI HTTP · 256 output | 22.87 tok/s · 11.19초 |
| DFlash2 SSE · 256 output | 10.40초 · 증분 스트리밍 성공 |
| DFlash2 WebP 이미지 인식 | 성공 · 9.05초 |
| DFlash2 27,404-token 입력 회수 시험 | 성공 · 11.51초 |
| WebP 이미지 인식 | 성공 · 10.84 tok/s |
| 14,951-token 입력 회수 시험 | 성공 |
| SparkMedia API + SparkTalk Extra 상주 | 약 28.42 GiB |
| 위 서비스 + Huihui TRT 상주 | clean 대비 약 65.08 GiB |
| Krea가 상주한 최악 조건의 LTX 기본 생성 | 90.19초 · HTTP 200 |
| 위 LTX 요청 전 가용 메모리 | 32.303 GiB |
| 위 LTX 요청 중 최소 가용 메모리 | 13.386 GiB |
| 위 LTX 요청 자체의 추가 피크 | 18.917 GiB |
| DFlash 기본 구성 + Krea 상주 후 LTX cold 생성 | 168초 · HTTP 200 |
| 위 DFlash/LTX 요청 전 가용 메모리 | 27.845 GiB |
| 위 DFlash/LTX 요청 중 최소 가용 메모리 | 13.167 GiB |
| 위 DFlash/LTX 요청 자체의 추가 피크 | 14.678 GiB |
| 모든 모델 상주 후 DFlash 256-token HTTP | 7.33초 · 34.94 tok/s · HTTP 200 |

DFlash2 DDTree Top-K 2는 DSpark와 상주 메모리가 사실상 같으면서 native 속도가
약 46% 빨랐고, OpenAI HTTP 실효 속도는 2배 이상이었다. 따라서 DFlash2를
기본으로 선택하고 DSpark 엔진은 회귀 시 즉시 되돌릴 fallback으로 보존한다.
SGLang 참고값보다 Edge-LLM 구성은 약 16GiB 적게 상주했다.

SparkTalk과 SparkMedia 전체의 동시 상주와 실제 이미지·업스케일·영상 생성은
통과했다. DFlash 기본 구성에서도 Krea 생성 후 LTX cold 생성과 256-token LLM
응답이 모두 HTTP 200이었고 최저 가용 메모리는 약 13.2GiB였다. 다만 최악 조건의
물리 여유가 크지는 않으므로 Krea 2, LTX-2.5,
SeedVR2 같은 생성 작업은 계속 공통 중량 작업 큐에서 한 번에 하나씩 실행한다.
