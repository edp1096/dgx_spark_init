> **운영 결정 변경:** 사용자 지시에 따라 QAT 교체를 취소하고 자체 모델 + 공식 assistant1/ko64k/청크1024로 복원한다. 아래는 QAT 실험과 이전 적용의 기록이다. 두 모델의 포괄적 품질 우열은 입증되지 않았다.

# Gemma 교체 평가 — QAT 교체 완료 (2026-09-16)

목적은 현재 자체 변환 모델을 유지할지, sakamakismile 배포본으로 교체할지 결정하는 것이다. 가중치 형식 변경 자체를 성과로 보지 않는다. 검증 중인 후보를 운영 기본값으로 승격하지 않는다.

| 후보 | 기반 | expert intermediate |
|---|---|---:|
| edp1096/Huihui-Gemma-4-26B-A4B-it-NVFP4 | 비QAT, ModelOpt | 704 |
| sakamakismile/Huihui-gemma-4-26B-A4B-it-abliterated-NVFP4-vLLM | 비QAT, compressed-tensors | 768 |
| sakamakismile/Huihui-gemma-4-26B-A4B-it-qat-abliterated-MTP-NVFP4 | QAT, compressed-tensors, 전용 assistant 포함 | 768 |

## 확보된 결과

- 60번 assistant1/전체 초안 어휘/Marlin/청크1024: 자체 모델 74.81, 비QAT 패딩 71.41 tok/s. 한국어·코드·산술·도구·이미지·80K 검색 모두 통과. 5% 이내 차이만으로 확정적인 우열을 선언하지 않는다.
- 61번 QAT 동일 설정 속도 중앙값 74.56 tok/s. 호스트가 다르므로 미세한 순위 비교 근거로 사용하지 않는다.
- 자체 모델/QAT 각각 한국어 3건·일본어 3건, 최대2048 출력: 기계적 연속 반복 검출 0건. 자체 모델 한국어에 `흐-으로으로`, `던집으로` 등 깨진 표현 관찰. 소규모 관찰이며 광범위한 품질 우열을 입증하지 않는다.
- 자체 모델의 운영 설정(assistant1+ko64k+커널 최적화+청크1024)은 1,039,984 입력 토큰 검색 통과(11,224.18초). 다른 후보·설정에는 이 결과를 전용하지 않는다.
- 세 tokenizer.json의 model/어휘/토큰 ID 등은 동일. 후보에는 truncation/padding 설정만 추가되어 있다. 이 설정을 해제한 한국어·일본어·코드·특수 토큰·15,601토큰 입력 인코딩 5건이 일치했다. 후보별 원본 파일 SHA256을 지정한 ko64k 목록을 별도로 준비했으며 전역 해시 검사를 해제하지 않았다.

## 비QAT 패딩 후보 제외 (16:14)

- 전용 FlashInfer CUTLASS: 68.19 tok/s로 Marlin 71.41보다 느림. 한국어 `여get`, `겨last`, `미칩으로` 등 문장 훼손.
- Marlin에서도 한국어 세 seed에서 깨진 표현 재현. 일본어 seed2026은 `おあつ` 연속 반복으로 2048토큰 제한까지 붕괴, 기계적 반복 검사도 실패.
- 따라서 이 비QAT 체크포인트는 교체 후보에서 제외한다. QAT 체크포인트에는 이 실패를 일반화하지 않는다.

## 진행 순서

1. 60번에서 세 후보의 커널 자동 선택 경로와 품질·속도 확인. 기존 Marlin 속도 결과 재사용. 현재 vLLM에서는 자체 704 모델도 FlashInfer CUTLASS 기동·기능·80K 검색에 성공했다.
2. 검증 통과 후보를 대상으로 청크1024 결과와 4096/8192/32768 비교. 동일80K 입력의 pp·ttft·tg·정답을 기록한다.
3. 최적화 후보의 ko64k, 기능·한일 장문·구조화 출력 검사. JSON 의미 정답과 코드 펜스 등 형식 준수를 분리해 기록한다.
4. 후보의 최종 설정으로 1M 검색 검증. 이미 검증한 자체 모델의 동일 설정이면 결과 재사용.
5. 전체 출력과 실패 원인을 검토해 유지/교체 결정, 적용 후 서비스 확인. 기존 자체 모델은 그 전에 삭제하지 않는다.

상태와 원시 출력: 60번 `~/.cache/model-download-jobs/ornith-gemma26/sequential/`의 `state.json`, `tune.json`, `final1m.json`. 앞 단계 실패를 성공으로 간주해 다음 단계로 넘기지 않는다. 튜닝용 임시 후보 선택은 운영 교체 결정과 구분한다.

## 참고 구현

- [kelnei/vllm-gemma4](https://github.com/kelnei/vllm-gemma4): 자동 NVFP4 커널, 청크32768, assistant/DFlash 측정. 다른 체크포인트·실행 조건의 수치를 우리 성능으로 대체하지 않는다.
- [eugr 레시피](https://github.com/eugr/spark-vllm-docker/blob/main/recipes/gemma4-26b-a4b-nvfp4.yaml): 청크8192, assistant4, instanttensor. 현재 검증에는 로더를 추가 변경하지 않는다.
- [nabe2030](https://github.com/nabe2030/gemma4-vllm-nvfp4-dgx-spark): 일본어 추론·이미지 구조화 출력 평가 참고. 오래된 커널 제약 설명보다 현재 설치된 소스와 실제 실행 결과를 우선한다.

## 최종 QAT 설정 실측 (20:16)

- 최종 설정: TP1, Marlin, FP8 KV 32 GiB, 문맥1,048,576, 청크4096, 전용 assistant1, 전체 초안 어휘, v6.
- 1,039,980 입력 + 512 출력: 검색 정답 `482913 / 571026 / 839405`의 순서까지 일치. 총10,978.61초, TTFT10,953.60초, tg20.43 tok/s. 단일 검색·설명 요청이며 전체1M 품질 보증은 아니다.
- 생성문에 `한꺼l에`, `시간적 지역인` 등의 오류가 있다. 검색 통과와 한국어 표현 품질을 분리한다. 한영 혼입이 해결됐다고 주장하지 않는다. 512토큰 출력 제한으로 마지막 문장이 잘렸다.
- 자체 모델의 이전1M 결과는 25토큰 출력이므로 해당 tg와 비교하지 않는다. TTFT는 두 모델 모두 약3시간 수준이다.

| 비교 조건 | 설정 | TTFT (초) | tg (tok/s) |
|---|---|---:|---:|
| QAT, 80K+256, 전체 어휘 | 청크1024 | 78.21 | 55.67 |
| QAT, 80K+256, 전체 어휘 | 청크4096 | 74.68 | 55.14 |
| QAT, 80K+256, ko64k | 청크4096 | 73.50 | 63.58 |
| QAT, 짧은 입력512출력 | 전체 어휘/청크4096 | — | 71.72 |
| QAT, 짧은 입력512출력 | ko64k/청크4096 | — | 75.87 |
| QAT, 일본어 장문 | 전체 어휘/청크4096 | — | 64.13 |
| QAT, 일본어 장문 | ko64k/청크4096 | — | 48.36 |

- 같은60번에서 얻은 결과다. 청크4096은1024 대비80K TTFT 약4.5% 개선. 8192/32768은 이를 넘지 못했다.
- ko64k는 한국어 속도에 유리하지만 일본어 장문에서 약25% 느렸다. 기본은 전체 어휘, ko64k는 선택 옵션으로 유지한다.
- 현재 vLLM의 native FlashInfer CUTLASS 결과: 자체71.67 / 비QAT768 68.19 / QAT69.21 tok/s. Marlin 대비 이득이 없어 채택하지 않는다.
- 기존 운영 자체 모델(ko64k/청크1024)의 짧은 입력76.71,80K58.02 tok/s와 비교하면 새 전체 어휘 설정이 더 빠른 것은 아니다. 교체 근거는 작은 한일 표본에서의 문장 훼손 감소와 QAT 기반·전용 assistant 결합이며, 속도 우위나 모든 문장의 정확성을 보장하지 않는다.
- 정식 Compose 재기동 후 이미지의 좌우 색을 strict JSON schema와 자동 도구 호출로 각각 정확히 반환했다. 컨테이너 healthy, v6, restart unless-stopped 확인.
- Talk 배포 완료: revision6에서 Gemma QAT ID로 마이그레이션. 실제 설정 API 전후 비교로 세트 이름·순서·호스트·실행 옵션·음성 설정 보존 확인. 기본 Ornith 유지, 전체 health 정상.

## 결론과 운영 상태

**Gemma 운영 항목은 sakamakismile QAT768 + 전용 assistant1로 교체한다.** 비QAT768은 제외한다. 자체 변환본은 운영 기본 경로에서 제외하고 기존 파일은 복구용으로 보존했다. 공개 HF 저장소 삭제나 변경은 하지 않았다.

- 60번: 정식 Compose의 Gemma QAT TP1, 전체 어휘, 청크4096, FP8 KV32GiB, 1M 문맥.
- 61번: Ornith 기존 서비스 유지. Talk 기본 선택도 Ornith이며 Gemma 선택 시 새 QAT 설정 사용.
- 소규모 품질 표본에 따른 선택이며 QAT도 극장문에서 글자 오류가 남는다. 모든 품질 문제 해결이나 속도 향상을 의미하지 않는다.
- 런타임 tokenizer 파일별 ko64k 목록 선택을 엄격한 해시 검사로 지원. v6와 v5의 vLLM 파이썬 커널 파일은 동일하다.
- Talk 자동 문맥 요약의 고정3분 제한을 제거하고 상위 요청의 기한·취소를 유지한다. 실제 장문 입력의 prefill이 수분을 넘길 수 있기 때문이다.
- 검증: launcher 3개 테스트, config/orchestrator/llm/server Go 테스트, 실제 저장 설정 읽기 전용 마이그레이션 검사, 지원 자산68개 검사, Compose 구문 검사, 배포 후 API 비교 통과.

원시 결과는 `~/.cache/model-download-jobs/ornith-gemma26/sequential/`에 보존했다. 주요 파일: `final1m-result.json`, `final-vision-json.json`, `qat768-final-full-speed.json`, `qat768-final-full-ja.json`, `quality-review.json`.
