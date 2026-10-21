# TP1 최적화 실측 — 2026-09-16

Compose와 Talk 모두 `dgx-vllm-moe-tp1:b40673cd0-v5` 적용. 61번은 Ornith, 60번은 Gemma이며 Talk 기본 모델은 Ornith다.

## 결과

짧은 입력은 한국어·코드·수학 각각 3회, 출력 512토큰의 tg 중앙값이다. TP1, batch 1, FP8 KV 32 GiB, prefill chunk 1024, temperature 0 조건이다.

| 모델 | 일반 tg | 최종 tg | 개선 | 일반 → 최종 ttft |
|---|---:|---:|---:|---:|
| Ornith | 80.51 | 107.30 | 33.3% | 0.107 → 0.104초 |
| Gemma | 51.03 | 76.71 | 50.3% | 0.055 → 0.059초 |

80K 입력 뒤 256토큰을 생성하는 별도 시험:

| 모델 | 일반 → 최종 tg | 일반 → 최종 ttft | 일반 → 최종 총시간 |
|---|---:|---:|---:|
| Ornith | 61.68 → 81.46 | 24.45 → 27.50초 | 28.58 → 30.63초 |
| Gemma | 40.81 → 58.02 | 75.70 → 77.66초 | 81.95 → 82.06초 |

단위는 tok/s다. 생성 속도는 개선됐지만 긴 입력의 prefill/전체 응답 시간이 개선된 것은 아니다. 숨겨 둔 문자열 검색은 모두 통과했다.

## 선택한 설정

- Ornith: MTP 1토큰 + ko64k. 최초 비교의 MTP 105.86, DFlash2 96.35 tok/s로 MTP를 선택했다. DFlash는 Triton/BLHNC가 필요했다.
- Gemma: 공식 assistant 1토큰 + ko64k. 전체 어휘 assistant1 71.59, assistant3 72.01로 3토큰의 이득이 작고 ttft가 증가했다.
- ko64k만 비교한 별도 실행: Ornith 103.43 → 109.39, Gemma 71.59 → 74.81 tok/s. 실행 시점 차이가 있어 최종 표와 같은 표본은 아니다.
- 초안 문맥을 본체와 맞춰 64K 경계 이후 fallback을 피했다. Gemma assistant의 위치 보간도 본체와 맞췄다.
- Gemma에서 투기 검증용 짧은 query의 full attention에 기존 split-KV 경로를 사용하고, KV를 소유하지 않는 assistant의 중간 prefill 계산을 생략했다. 기존 경로의 80K tg 14.89가 최종 58.02로 회복됐다.

## 검증과 한계

한국어·코드·산술·도구 호출·이미지 및 최종 80K 검색을 통과했다. Gemma 일반 디코딩은 1,039,984 입력 토큰 검색도 통과했으나 11,021.6초가 걸렸다. 최종 assistant1+ko64k+커널 최적화 설정도 1,039,984 입력 토큰에서 암호 3개 검색을 통과했다(2026-09-16 14:37, 11,224.18초). 검색 성공이 장문 생성 품질 전반을 보증하지 않는다.

ko64k는 초안 출력 head만 줄인다. 본체 head와 가중치는 유지한다. 특수 토큰·byte fallback·한글/Jamo를 보존하며 tokenizer 해시를 검사한다. Ornith NVFP4 head는 원래 packed 값과 scale을 사용한다. 어휘에서 빠진 토큰도 본체 검증/생성 대상에서 제외되지 않는다.

Gemma split-KV는 지원 shape·dtype·full attention 조건에만 적용한다. 14개 수치/버퍼 검사와 FP8 기준 계산 비교를 통과했다. FP8 연산의 분할에 따라 반올림 차이가 있어 bitwise 동일하지 않다. 80K 기준 정규화 RMS 오차는 기존 2.597%, 변경 2.607%였다.

실행 옵션은 `MTP_TOKENS=0`으로 투기 디코딩을 끄거나 `DRAFT_VOCAB=off`로 전체 초안 어휘를 쓸 수 있다. Gemma 추가 최적화는 `SPARKTALK_TRITON_SPEC_SPLIT`, `SPARKTALK_GEMMA_PREFILL_SKIP`로 제어한다.

런처 검사, Talk config/orchestrator 테스트 통과. 원시 측정은 `~/.cache/model-download-jobs/ornith-gemma26/`의 `ornith-long-draft-*`, `gemma-kernel-*`, `*-sustained-*`, `*-shortlist-*`, `*-spec-*`에 보관했다.
