# TensorRT Edge-LLM on DGX Spark

DGX Spark에서 `Qwen3.8-27B NVFP4 + DFlash2 + vision encoder`를 OpenAI 호환
API로 실행하는 실험 구성이다. TensorRT Edge-LLM v0.10.0의 커밋
`71dd1bae032e70771265917ec74d3ff4cad07a10`에 고정한다. Dockerfile에 이 SHA를
직접 기록했으며 Compose 환경변수로 다른 ref를 주입할 수 없다.

현재 Qwen3.8과 이 DFlash2 체크포인트 조합은 NVIDIA의 공식 검증 조합이 아니다.
실측에서는 정상 동작하지만 SGLang보다 DFlash 수용률과 생성 속도가 낮았다.

## 실행

ONNX와 TensorRT 엔진은 용량이 크고 장치 종속적이므로 `workspace/`에 두며 Git에
포함하지 않는다. 현재 변환된 엔진이 있으면 다음 명령만 필요하다.

```bash
cd /home/edp1096/workspace/dgx_spark_init/compose_yaml/trt-edge-llm
docker compose up -d --no-build server
curl -fsS http://127.0.0.1:8696/v1/models
```

런타임은 `nvcr.io/nvidia/pytorch:26.07-py3`의 TensorRT 11.1/CUDA 13.3을
사용한다. TensorRT 10.14에서는 DFlash base engine 빌드가 `PLUGIN_V3`
optimizer 오류로 실패했다.

## 변환 및 엔진 빌드

```bash
docker compose --profile tools build exporter runtime

DRAFT=/draft/snapshots/dedf8df68adfb1afeaf7b7480c0a0243108177b4

# 비전 ONNX
docker compose --profile tools run --rm exporter \
  /model /workspace/onnx/multimodal

# DFlash target/draft ONNX
docker compose --profile tools run --rm exporter \
  /model /workspace/onnx/dflash-base \
  --dflash-base --dflash-draft-dir "$DRAFT"
docker compose --profile tools run --rm exporter \
  /model /workspace/onnx/dflash-draft \
  --dflash-draft --dflash-draft-dir "$DRAFT"

# Target와 draft engine
docker compose --profile tools run --rm runtime \
  ./build/examples/llm/llm_build \
  --onnxDir /workspace/onnx/dflash-base/llm \
  --engineDir /workspace/engines/dflash-trt11 \
  --maxBatchSize 1 --maxInputLen 32768 --maxKVCacheCapacity 32768 \
  --maxVerifyTreeSize 16 --specBase
docker compose --profile tools run --rm runtime \
  ./build/examples/llm/llm_build \
  --onnxDir /workspace/onnx/dflash-draft/dflash_draft \
  --engineDir /workspace/engines/dflash-trt11 \
  --maxBatchSize 1 --maxInputLen 32768 --maxKVCacheCapacity 32768 \
  --maxDraftTreeSize 16 --specDraft

# Vision engine
docker compose --profile tools run --rm runtime \
  ./build/examples/multimodal/visual_build \
  --onnxDir /workspace/onnx/multimodal/visual \
  --engineDir /workspace/engines/dflash-trt11 \
  --minImageTokens 128 --maxImageTokens 4096 --maxImageTokensPerImage 512
```

`openai-image-url.patch`는 SparkTalk의 OpenAI `image_url` data URL을 C++
런타임 입력으로 연결한다. Edge-LLM의 기본 이미지 디코더가 WebP를 받지 못하므로
WebP data URL은 서버 어댑터에서 무손실 PNG로 변환한다. NVFP4 vision 모델 자체는
JPEG, PNG, WebP 변환 입력을 모두 정상 인식했다.

`qwen-streaming-reasoning.patch`는 v0.10.0의 실험 서버에 다음 호환 계층을
추가한다.

- Qwen XML 도구 호출은 일반 답변·리즈닝을 즉시 SSE로 보내고, 실제
  `<tool_call>...</tool_call>` 블록만 완성될 때까지 보류한다. 다른 형식은
  v0.10.0의 전체 응답 파서로 폴백한다. 런타임이 바깥 특수 토큰을 제거하는
  경우의 `<function=...>...</function>` 단독 블록도 처리한다. 종료 프로토콜
  토큰이 여러 SSE delta에 나뉘어도 사용자 본문에 노출하지 않는다.
- OpenAI `reasoning_effort`를 Qwen3.8의 네이티브 단계에 연결한다.
  `none`은 리즈닝을 끄고, `minimal/low → low`, `medium → medium`,
  `high/xhigh/max → xhigh`로 매핑한다. 0~1 숫자도 같은 세 구간으로 매핑한다.
  체크포인트가 `</think>` 없이 종료하면 리즈닝 스트림은 유지하면서 본문이
  빈 응답이 되지 않도록 같은 텍스트를 최종 content로 폴백한다.
- 패치는 위에 적힌 정확한 v0.10.0 커밋을 기준으로 생성했다. 후속 릴리즈로
  기준을 바꾸면 `git apply --check`가 빌드를 중단하므로 패치를 명시적으로
  재검토해야 한다.

이 패치는 Python 서버 계층만 바꾸므로 ONNX export나 TensorRT 엔진 재생성은
필요하지 않다.

## 2026-08-24 DGX Spark 실측

| 항목 | 통합 메모리/성능 |
|---|---:|
| Edge-LLM DFlash2 + vision 상주 | 약 39.0 GiB |
| NVIDIA compute process | 34,496 MiB |
| 이전 SGLang 동일 target+draft 상주 | 약 52.5 GiB |
| Native C++ 104 input + 256 output | 17.3 tok/s |
| DFlash 평균 수용 토큰 | 2.06 |
| Edge + SparkMedia 전체 상주 | 약 90.8 GiB |
| 위 상태에서 LTX-2.5 최소 생성 피크 | 108.7 GiB |
| 피크 시 남은 물리 메모리 | 약 12.9 GiB |

SparkTalk과 SparkMedia의 동시 상주는 가능하다. 다만 LTX-2.5, Krea 2,
SeedVR2처럼 순간 메모리를 크게 쓰는 생성 작업은 한 번에 하나씩 실행하는 전제를
유지한다.
