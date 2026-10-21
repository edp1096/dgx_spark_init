# GLM 초기 엔진 비교 · 2026-09-27

> 아래는 DFlash2 도입 전의 보존 기록입니다. 현재 설정과 결과는 상위 README 및 qualification.json을 기준으로 합니다.

이번 시험은 현재 로컬 구성의 비교다. 최적화된 vLLM 전체보다 SGLang이 우수하다는 검증이 아니다. 양쪽 모두 같은 Huihui NVFP4, TP2, 1,048,576 문맥, 동시 요청 1개, CPU 70%·GPU 2100MHz를 사용했다.

| 설정 | 로컬 vLLM | 로컬 SGLang |
|---|---|---|
| 이미지 | `dgx-vllm-moe-tp1:b40673cd0-v5` | `dgx-sglang-glm53:sm121-dev5` |
| 디코드 그래프 | 끔(eager) | batch 1 켬 |
| 입력 묶음 | 1,024 | 4,096 |
| 추측 디코딩 | 끔 | 끔 |
| MoE·FP4 Linear | B12X | FlashInfer CUTLASS |

숫자 나열은 속도 측정과 지시 준수 점검용이다. 이 한 종류의 합성 요청으로 모델 전반의 품질이나 특정 커널의 결함을 단정하지 않는다. 단문은 SGLang의 64토큰 캐시 페이지보다 짧아 재사용되지 않으므로, 6,183토큰 공통 문맥도 별도 측정한다. 로딩과 추론의 스왑 쓰기를 분리한다. 저장된 스왑 용량과 누적 쓰기량은 다르다.

## 실측 결과

| 항목 | vLLM | SGLang |
|---|---:|---:|
| 짧은 새 요청 TTFT 중앙값 | 0.510초 | 0.227초 |
| 짧은 새 요청 TG 중앙값 | 4.19t/s | 15.62t/s |
| 기능 / RGB / 다중 턴 계산 | 9 / 3 / 24 통과 | 9 / 3 / 24 통과 |
| 동일 숫자 나열 요청 27개 실패 | 15 | 19 |
| 로딩 스왑 누적 쓰기(헤드 / 워커) | 3.29 / 2.28GiB | 26.04 / 23.02GiB |
| 추론 스왑 쓰기(헤드 / 워커) | 0 / 4KiB | 0 / 0 |

각각 10분 이상 직렬 부하를 완료했다. 서버 종료·재부팅은 없었으나 vLLM 로딩 중 복구된 NVIDIA 메모리 할당 실패 경고가 1건 있었다. 현재 SGLang도 포럼의 최적화된 성능을 재현하지 못했고 로딩 스왑이 크므로 Talk 기본 엔진 전환은 보류한다. 상세 측정·범위: [engine-comparison.json](engine-comparison.json).

## 포럼 전체 확인

[원문](https://forums.developer.nvidia.com/t/lets-optimize-nvidia-glm-5-3-flash-nvfp4-for-2x-dgx-spark/382939)의 Discourse 공개 글 ID 132개를 전부 조회했다. 마지막 글은 135번(2026-09-27 09:20 UTC)이다. 처음 렌더링되는 20개 외에 112개를 추가로 읽고 측정 이미지도 확인했다.

- [26번](https://forums.developer.nvidia.com/t/lets-optimize-nvidia-glm-5-3-flash-nvfp4-for-2x-dgx-spark/382939/26): DFlash2 외에 async scheduling, graph·Mamba·batch 설정도 중요하다.
- [80~82번](https://forums.developer.nvidia.com/t/lets-optimize-nvidia-glm-5-3-flash-nvfp4-for-2x-dgx-spark/382939/82): 0rand는 0.29의 전환 지연을 보고하고 0.28을 유지한다. 1M, KV 11GiB, batch 4096에서 30~35t/s 보고.
- [108번](https://forums.developer.nvidia.com/t/lets-optimize-nvidia-glm-5-3-flash-nvfp4-for-2x-dgx-spark/382939/108): 우리 원본과 같은 `09b04e...` revision으로 DFlash K5·그래프·async를 사용한 재현 보고가 있다.
- [118~121번](https://forums.developer.nvidia.com/t/lets-optimize-nvidia-glm-5-3-flash-nvfp4-for-2x-dgx-spark/382939/118): DFlash draft-cache 보존과 매칭이 충돌하는 prefix 재사용 실패 사례. `--prefix-cache-retention-interval 4608` 제안과 후속 개선 보고가 있지만 모든 환경의 해결책은 아니다.
- [127번](https://forums.developer.nvidia.com/t/lets-optimize-nvidia-glm-5-3-flash-nvfp4-for-2x-dgx-spark/382939/127): 일반 문장 30~35, 코드·구조화 40~45t/s 보고. 디스플레이 RAM 회수는 headless·별도 구현 전제이며 이번 구성에는 적용하지 않았다.
- [132번](https://forums.developer.nvidia.com/t/lets-optimize-nvidia-glm-5-3-flash-nvfp4-for-2x-dgx-spark/382939/132): 공개 전 개선안은 53.2t/s, 기준 41.0t/s. 첨부 점수는 각각 75.4%, 76.6%이며 품질 동등성을 우리가 검증한 것은 아니다.

따라서 로컬 eager vLLM과의 차이를 이 포럼 구성 대비 성능 향상으로 표현해서는 안 된다. 외부 수치와 직접 비교하려면 모델(원본/abliterated), 입력, 출력 길이, 가속 기능과 벤치 버전도 맞춰야 한다.

## 정정

NVIDIA 원본 `09b04e...`와 현재 Huihui 파생 모델 모두 `num_hidden_layers=45`, `num_nextn_predict_layers=1`이며 `layers.45`에 NextN/MTP 가중치가 있다. 기존의 “MTP 가중치가 없다”는 설명은 오류다. 현 레시피의 MTP 제한은 TP2/1M 사용 미검증에 따른 것이며, 가중치 부재 때문이 아니다.
