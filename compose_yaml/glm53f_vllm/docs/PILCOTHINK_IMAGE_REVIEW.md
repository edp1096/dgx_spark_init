# pilcothink GLM 0.28 이미지 조사

2026-09-27. 자체 vLLM 이미지는 소스 빌드로 재구성할 수 있는 근거를 확보했다. SGLang도 최신 upstream에 GLM-5.3 Flash·ModelOpt NVFP4 구현이 있어 개발 가능성이 있지만, vLLM 패치를 그대로 옮기는 방식은 맞지 않는다. 이번 작업은 이미지·소스 조사이며 새 이미지 빌드나 SGLang GPU 실행을 완료한 것은 아니다.

## 확인한 이미지

- `pilcothink/vllm_spark_glm53:0.28`
- 태그 index: `sha256:e99cb670acdedd3234c595cd81dbefa7f091507268c78f659b3286a957a5536f`
- ARM64 manifest: `sha256:09da6eb394216d174ab8692758d90f9f458398d9c8fbc11ba6f04e93d5cf6392`
- 압축 레이어 합계 10.06GiB. 전체 CUDA OS 레이어를 설치하지 않고 런타임·패치 레이어 약 2.90GiB를 단일 스트림으로 받아 SHA-256을 검증했다.
- `/workspace/provenance/`에 vLLM·FlashInfer 소스 차이와 빌드 JSON이 있고, `/opt/glm53-dflash2-v31/`, `/opt/glm53-dflash2-v33/`에 패치·Dockerfile·검사 코드가 있다.
- 소스 차이 파일의 해시는 빌드 기록과 일치한다. v33 기대 매니페스트의 최종 소스 22개도 모두 일치한다.
- 이것은 레이어 파일 조사다. 이미지 전체를 실행하거나 해당 이미지의 성능을 측정하지 않았다.

## 기반 구성

| 항목 | 확인값 |
|---|---|
| OS / CUDA | Ubuntu 24.04 / CUDA 13.0.2 devel |
| PyTorch | 2.13.0, cu130 |
| vLLM | `6fbb00b18874e27ba7d7adc0a3b8e93fee763ab1` |
| FlashInfer | `27d5b029818e530a9fd0c6b3b356217b5bef1226`, 0.6.18 |
| B12X | `06b4de7c723e6f166d65abf5909c5b7d0f8acc68` |
| NCCL | `fd168324a3dc0c9080fd4881b6c7f4bb252a95a2`, SM121 소스 빌드 |
| CUTLASS DSL / TVM FFI | 4.7.0 / 0.1.11 |
| eugr 참조 | `841fdcc4bde9f84c0abbed72c6df2d435401942e` |
| DeepGEMM | `a6b593d2826719dcf4892609af7b84ee23aaf32a` |

vLLM PR 55277(SM120 sparse MLA/cache), 54788(draft MoE override), 55736(GLM KDA decode), FlashInfer PR 4947 변경이 기록돼 있다. SM12x 타깃 보존, KV 정리, page-size 64 계약, BF16 MTP 처리도 추가되어 있다. 현재 우리 검증 이미지는 `b40673cd0-v5`이므로 이미지 이름만 바꿔 같은 환경이라고 볼 수 없다.

## DFlash2 추가 부분

- v31: mHC hidden-state 다섯 지점 연결, multimodal wrapper, target과 draft의 KV group 분리, native LBHNC 물리 slot의 strided view 사용, DFlash K=4~7.
- v33: `KpoolTailSpec`을 일반 절대 위치 slot mapper에서 제외하고 전용 원형 tail mapper를 유지한다. 변경 대상은 V2 `model_runner.py` 한 파일이다.
- 이미지 제작자의 label에는 GPU 실행 검증이 `false`로 남아 있다. 외부 사용자의 실행 보고와 구분해야 한다.
- v33에는 Apache-2.0 LICENSE/NOTICE가 있지만, v31 디렉터리에는 독립적인 라이선스 선언이 확인되지 않았다. 해당 추가 코드를 그대로 자체 배포물에 복사하는 방안은 확정하지 않았다.
- 이미지 기록의 NVIDIA checkpoint revision은 `423acf...`다. 우리 변환 기준은 `09b04e...`이며 `layers.45`에 MTP/NextN 가중치가 있다. 이전의 가중치 부재 설명은 잘못이었다. MTP 설정을 그대로 가져오면 안 된다. DFlash2도 별도 draft 가중치·검증이 필요하다.

## 자체 Dockerfile 경로

1. CUDA 13.0.2 ARM64를 기반으로 NCCL·PyTorch·CUTLASS/TVM 버전을 고정한다.
2. vLLM·FlashInfer·B12X를 고정 커밋에서 빌드한다. 공식 소스의 필요한 수정과 직접 구현할 통합 부분을 구분하고 라이선스·출처를 유지한다.
3. 빌드 단계에서 wheel을 만들고 실행 단계에는 wheel·필요한 라이브러리·빌드 출처만 넣는다. GPU 커널은 SM121로 고정한다.
4. 먼저 현재처럼 draft 없이 TP2 / FP8 KV / 1M을 검증하고, DFlash는 별도 단계로 추가한다.

소스 pins와 차이는 확보했지만 최종 이미지 history에는 중간 builder stage 전체가 없고, 기록에도 완전한 전이 의존성 lock이 아니라고 명시돼 있다. 따라서 기능상 재현은 가능성이 높지만 동일 바이너리의 byte-for-byte 재현을 확인한 상태는 아니다. `FROM pilcothink/...`만 작성하는 것은 독립적인 소스 빌드가 아니다.

## SGLang 경로

확인한 upstream: `425a1f8f247d0cc17f2a3f3c2dba6c0bdf936552`.

- `Glm5NextForConditionalGeneration`, NVIDIA ModelOpt FP4 exclusion/weight-name mapping, 비양자화 shared expert의 fusion 제외 처리가 있다.
- DFlash용 mHC hidden-state capture와 테스트가 있다. vLLM의 v31 adapter를 복사할 필요부터 전제할 이유가 없다.
- SGLang은 paged KV와 KDA state pool을 나눠 관리하므로 vLLM의 group/slot allocator를 붙이는 대신 자체 pool 용량·prefix state 수명·speculative rollback을 검증해야 한다.
- 구체적인 막힘도 확인했다. `flash_mla_sm120.py`의 SM120/SM121 FP8 KV용 `flashinfer_sparse_mla` 허용 목록은 `GlmMoeDsaForCausalLM`/`NextN` 두 가지뿐이고, 이번 모델인 `Glm5NextForConditionalGeneration`은 없다. 따라서 이 backend를 그대로 지정하면 초기 검사에서 거부된다. 목록만 늘리는 것이 아니라 GLM5Next의 KV packing·head dimension·KPOOL index 형식이 커널 계약과 맞는지 검증해야 한다.
- KDA는 Triton 경로가 있고 FlashInfer KDA는 SM100 경로로 명시되어 있다. 같은 이름의 backend를 무조건 선택하면 안 된다.
- SM121의 NVFP4 MoE·Linear, 실제 DSA/MLA·KDA·mHC 커널 실행과 FP8 KV를 먼저 확인해야 한다. 공식 cookbook의 Blackwell 서버 GPU 결과는 GB10 TP2 검증을 대신하지 않는다.
- 현재 QAD 이미지 `sm121-v5`의 SGLang `d91c3682b`에는 `glm5_next.py`가 없다. QAD 전용 launch를 재사용하기보다 최신 upstream 기반 GLM 전용 이미지를 만드는 편이 명확하다.

우선순위는 검증된 현재 vLLM의 독립 빌드를 확보한 뒤, SGLang 신규 이미지를 작은 문맥 → 256K → 512K → 1M 순서로 비교하는 것이다. 새 런타임에서 같은 1M 용량·속도가 나온다는 보장은 아직 없다.

출처: [0rand 구성](https://github.com/0rand/glm-5.3-flash-nvidia-nvfp4-dflash-2x-dgx-sparks), [SGLang 모델](https://github.com/sgl-project/sglang/blob/425a1f8f247d0cc17f2a3f3c2dba6c0bdf936552/python/sglang/srt/models/glm5_next.py), [ModelOpt 검사](https://github.com/sgl-project/sglang/blob/425a1f8f247d0cc17f2a3f3c2dba6c0bdf936552/test/registered/unit/models/test_glm5_next_modelopt.py), [SGLang cookbook](https://github.com/sgl-project/sglang/blob/425a1f8f247d0cc17f2a3f3c2dba6c0bdf936552/docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx).
