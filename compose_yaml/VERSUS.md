# Qwen3.8 Flash-Next · SGLang vs vLLM

> DGX Spark 한 대에서 같은 모델 실측 결과 · 2026-09-02

## 결론

**SparkTalk 기본 엔진은 SGLang이 낫다.**

* 단일 생성 속도는 사실상 동률
* 첫 응답·동시 요청·긴 입력·메모리 여유는 SGLang이 약간 우세

| 핵심 지표 | SGLang | vLLM | 우세 |
|---|---:|---:|:---:|
| API 준비 시간 | **13분 27초** | 14분 15초 | SGLang |
| 단일 생성 평균 | 22.79 tok/s | **23.21 tok/s** | 동률 |
| 첫 토큰 평균 | **0.287초** | 0.392초 | SGLang |
| 동시 2요청 합산 | **82.41 tok/s** | 31.34 tok/s | SGLang |
| 8K Prefill | **1,767 tok/s** | 1,333 tok/s | SGLang |
| 32K Prefill | **2,358 tok/s** | 2,286 tok/s | 근소 SGLang |
| 유휴 시 호스트 사용량 | **98.9 GiB** | 100.5 GiB | SGLang |

## 시험 조건

| 항목 | SGLang | vLLM |
|---|---|---|
| 구성 | [`sglang_qwen38_flash_next`](sglang_qwen38_flash_next/) | [`vllm_qwen38_flash_next`](vllm_qwen38_flash_next/) |
| 모델 | `dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4` | 동일 |
| 문맥 | 64K | 64K |
| 동시 요청 | 2 | 2 |
| 병렬화 | TP 1 | TP 1 |
| 추측 디코딩 | NEXTN · 3단계 · draft 4 | MTP · draft 2 |
| PLE | 파일 offload | NVMe mmap |
| KV | 자동 | 고정 7 GiB |

- 다른 컨테이너를 모두 내리고 엔진을 한 번에 하나씩 실행했다.
- 각 엔진의 현재 운영용 최적 구성을 비교했다. 추측 디코딩 알고리즘만 따로 비교한 시험은 아니다.
- 디스크 캐시는 강제로 비우지 않았다. SGLang을 먼저, vLLM을 나중에 실행했다.

## 단일 요청

256토큰 생성, thinking 끔, temperature 0 기준이다.

| 작업 | SGLang TTFT / 속도 | vLLM TTFT / 속도 |
|---|---:|---:|
| 일반 문장 | **0.288초** / 18.66 tok/s | 0.545초 / **20.72 tok/s** |
| 코드 | 0.359초 / **29.17 tok/s** | **0.308초** / 24.96 tok/s |
| 추론 | **0.213초** / 20.55 tok/s | 0.325초 / **23.96 tok/s** |

속도 차이는 1.8%뿐이라 사실상 동률이다. 첫 토큰은 SGLang이 평균 약 27% 빨랐다.

## 동시 2요청

256토큰 요청 두 개를 동시에 보냈다.

| 엔진 | 완료 시간 | 합산 처리량 |
|---|---:|---:|
| SGLang | **6.21초** | **82.41 tok/s** |
| vLLM | 16.34초 | 31.34 tok/s |

SGLang이 약 **2.63배** 빨랐다.

여기서 동시 2요청은 TP 2가 아니다. **TP 1 서버에 서로 다른 HTTP 요청 두 개가 겹쳐 들어간 것**이다. SparkTalk에서도 두 대화방의 답변 생성 시간이 겹치면 같은 상황이 된다.

vLLM 로그에는 다음 fallback이 기록됐다.

```text
Fused multi-step draft decode is not supported by QSA state.
Falling back to rebuilding attention metadata between draft steps.
```

현재 vLLM의 동시 처리 손실을 설명하는 가장 유력한 원인이다.

## 긴 입력

출력은 1토큰으로 제한해 prefill만 비교했다.

| 입력 | SGLang | vLLM | 차이 |
|---|---:|---:|---:|
| 약 8K | **1,767 tok/s** | 1,333 tok/s | SGLang +32.5% |
| 약 32K | **2,358 tok/s** | 2,286 tok/s | SGLang +3.2% |

## 기능 확인

| 기능 | SGLang | vLLM |
|---|:---:|:---:|
| 이미지 인식 | 정상 | 정상 |
| OpenAI형 도구 호출 | 정상 | 정상 |
| 리즈닝 분리·스트리밍 | 정상 | 정상 |
| 결정론적 수열 출력 | 동일 | 동일 |

vLLM도 SparkTalk의 `low`, `medium`, `xhigh` 리즈닝 요청과 `reasoning` 필드를 정상 처리했다. 영상은 이번 비교에서 제외했고, 음성은 별도 ASR을 사용한다.

## 메모리와 기동

| 항목 | SGLang | vLLM |
|---|---:|---:|
| 호스트 사용량 | **98.9 GiB** | 100.5 GiB |
| 시스템 가용 | **22.7 GiB** | 21.1 GiB |
| swap 사용 | **약 0.7 GiB** | 약 3.1 GiB |

통합 메모리 환경이라 Docker의 cgroup 수치보다 호스트 전체 메모리를 기준으로 기록했다. 두 엔진 모두 OOM은 없었다.

기동 중 SGLang의 206개 샤드 ETA는 후반부를 심하게 과소평가했다. 90% 이후 숫자가 오래 멈춘 것처럼 보여도 실제로는 CPU와 디스크를 사용하며 적재 중이었다.

## 선택

- **기본 운영:** SGLang
- **보존 용도:** vLLM 비교·복구 구성
- **재평가 시점:** vLLM에 QSA용 fused multi-step MTP가 적용되거나, 동시성 8 이상이 필요할 때

시험 후 두 엔진은 모두 중지했고 호스트 메모리는 약 117 GiB 가용 상태로 돌아왔다.
