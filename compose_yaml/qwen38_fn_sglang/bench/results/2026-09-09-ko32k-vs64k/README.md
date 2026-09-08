# ko32k 실제 실행과 기존 ko64k 비교

동일한 24개 문항을 두 번씩 실행한 48쌍이다. 32k는 새 실행, 64k는 재부팅 전 저장된 본 실험이다. 후보 어휘 외 생성 설정과 입력 토큰은 일치 여부를 검사했다.

**판단: 이번 표본에서는 범용 기본값을 ko64k에서 ko32k로 바꿀 뚜렷한 이점이 확인되지 않았다.**

- 합산 decode 처리량은 30.18→30.22 tok/s로 약 0.13% 차이다. 문항·반복별 속도비 기하평균은 32k가 2.30% 높고, 양쪽 답변 완료 32쌍에서는 4.04% 높다. 서로 다른 가중 방식의 지표이므로 구분해서 읽어야 한다.
- 이번 32k 실행의 관측 SM 클럭 중앙값은 2,086MHz, 기존 64k는 1,989MHz로 약 4.9% 차이가 있다. 실행 시점·부팅 세션도 다르므로 작은 속도 차이를 후보 어휘 축소의 순수 효과로 귀속할 수 없다. 클럭 비율로 성능을 단순 보정하지도 않았다.
- 초안 수락률은 51.77%→48.40%로 3.38%p 낮아졌다. 후보 출력층은 가벼워졌지만 실제 합산 처리량 개선은 거의 없었다.
- 고정된 3개 기준의 합산 점수는 99/144→97/144, 전체 기준 통과는 24/48→22/48이다. 같은 요청끼리 비교하면 37쌍 동점, 32k 우세 5쌍, 64k 우세 6쌍이다. 품질 차이는 작으며 이 자료로 일반적인 품질 우열을 확정하지 않는다. 서술형은 assistant 평가이고 독립 전문가 검증이 아니다.
- 두 설정 모두 Python 코드 작성 4건과 메모리 분석 2건에서 최종 답변을 내지 못했다. 실제 제출 코드가 없으므로 생성 코드의 기능 검사 실패율로 표현하지 않는다. 원장 집계도 양쪽 모두 정답을 맞추지 못했다. 32k의 연구 비교 한 응답은 최종 설명 도중 잘려 마지막 연구 평가를 빠뜨렸다.
- 후보 출력층 가중치 절감은 정확히 160MiB(320→160MiB)다. 모델 전체 메모리 절반 절감이 아니다. 32k 실행은 재부팅·OOM·감시 중단 없이 끝났고, 실험 서버는 제거했다. 운영 설정은 변경하지 않았다.

| 항목 | ko64k | ko32k |
|---|---:|---:|
| 루브릭 점수 | 99/144 | 97/144 |
| 3개 기준 모두 통과 | 24 | 22 |
| 서술형 중요 오류 표시 | 14 | 14 |
| 출력 한도 도달 | 14 | 15 |
| 최종 답변 없음 | 6 | 6 |
| 생성 토큰 수 | 73226 | 72178 |
| 응답 대기시간 합계(분) | 41.05 | 40.39 |
| 합산 decode tok/s | 30.18 | 30.22 |
| 합산 전체 응답 tok/s | 29.73 | 29.78 |
| 첫 토큰 지연 중앙값(초) | 0.293 | 0.263 |
| draft 수락률 | 51.77% | 48.40% |
| 검증당 출력 토큰 | 2.553 | 2.452 |

문항·반복별 생성 속도비(32k/64k)의 기하평균은 **1.0230**이다. 1보다 크면 32k가 빠르다.
중요 오류 표시는 서술형 검토의 별도 지표다. 코드 미제출과 집계 오답은 루브릭 점수에 반영되며 이 표시 수에는 자동으로 포함되지 않는다.
양쪽 모두 한도에 걸리지 않고 답변을 낸 32쌍의 속도비는 1.0404이다.
완전히 동일한 출력 토큰열은 0/48쌍이다.

## 분야별 비교

| 분야 | 쌍 수 | 64k 점수 | 32k 점수 | 속도비 32k/64k |
|---|---:|---:|---:|---:|
| coding | 6 | 4/18 | 4/18 | 1.038 |
| documents | 4 | 6/12 | 6/12 | 1.020 |
| economics | 4 | 11/12 | 10/12 | 0.984 |
| everyday | 6 | 16/18 | 16/18 | 1.049 |
| humanities | 2 | 5/6 | 5/6 | 0.992 |
| language | 4 | 12/12 | 12/12 | 1.060 |
| medicine | 10 | 22/30 | 21/30 | 1.020 |
| planning | 4 | 6/12 | 8/12 | 1.026 |
| research | 4 | 6/12 | 3/12 | 0.971 |
| science | 4 | 11/12 | 12/12 | 1.041 |

## 문항별 비교

| 문항 | 반복 | 64k 점수/3 | 32k 점수/3 | 속도비 |
|---|---:|---:|---:|---:|
| code_html_review | 1 | 2 | 2 | 1.057 |
| code_html_review | 2 | 2 | 2 | 1.202 |
| code_shortlist | 1 | 0 | 0 | 0.919 |
| code_shortlist | 2 | 0 | 0 | 1.047 |
| code_telemetry | 1 | 0 | 0 | 1.064 |
| code_telemetry | 2 | 0 | 0 | 0.964 |
| contract_reading | 1 | 3 | 3 | 1.083 |
| contract_reading | 2 | 3 | 3 | 1.137 |
| dialogue_constraints | 1 | 3 | 3 | 1.105 |
| dialogue_constraints | 2 | 3 | 3 | 1.082 |
| finance_budget | 1 | 3 | 3 | 0.957 |
| finance_budget | 2 | 2 | 2 | 0.982 |
| finance_cost | 1 | 3 | 2 | 1.036 |
| finance_cost | 2 | 3 | 3 | 0.962 |
| human_history | 1 | 2 | 3 | 0.983 |
| human_history | 2 | 3 | 2 | 1.002 |
| human_translation | 1 | 3 | 3 | 0.981 |
| human_translation | 2 | 3 | 3 | 1.153 |
| human_writing | 1 | 3 | 3 | 1.043 |
| human_writing | 2 | 3 | 3 | 1.072 |
| life_energy | 1 | 3 | 3 | 1.056 |
| life_energy | 2 | 3 | 3 | 1.052 |
| life_food | 1 | 2 | 2 | 1.023 |
| life_food | 2 | 2 | 2 | 1.100 |
| life_router | 1 | 3 | 3 | 1.008 |
| life_router | 2 | 3 | 3 | 1.055 |
| life_schedule | 1 | 0 | 1 | 0.902 |
| life_schedule | 2 | 0 | 1 | 1.028 |
| long_ledger | 1 | 0 | 0 | 0.942 |
| long_ledger | 2 | 0 | 0 | 0.934 |
| med_cancer | 1 | 2 | 2 | 1.027 |
| med_cancer | 2 | 2 | 2 | 1.038 |
| med_cold | 1 | 1 | 1 | 1.009 |
| med_cold | 2 | 2 | 1 | 1.054 |
| med_diabetes | 1 | 2 | 2 | 1.119 |
| med_diabetes | 2 | 2 | 2 | 0.918 |
| med_gout | 1 | 3 | 3 | 0.958 |
| med_gout | 2 | 3 | 3 | 1.004 |
| med_rhinitis | 1 | 3 | 2 | 1.072 |
| med_rhinitis | 2 | 2 | 3 | 1.020 |
| research_memory | 1 | 0 | 0 | 0.803 |
| research_memory | 2 | 0 | 0 | 1.052 |
| science_bayes | 1 | 2 | 3 | 1.026 |
| science_bayes | 2 | 3 | 3 | 1.102 |
| science_lab | 1 | 3 | 3 | 1.037 |
| science_lab | 2 | 3 | 3 | 1.002 |
| science_trial | 1 | 3 | 2 | 1.204 |
| science_trial | 2 | 3 | 1 | 0.875 |

## 측정 범위와 한계

- Reference measured before host reboot, candidate after it; sessions/order/thermal conditions differ.
- Reference includes a process restart; prompts/settings match but this is not an interleaved trial.
- 48 paired requests are two repeats of 24 tasks, not 48 independent task types.
- Three binary checks per task; prose is assistant judgment, not independent expert validation.
- Token-limit and missing-final failures are included; no supplementary longer-budget trial is pooled.
- Different output token sequences and lengths can influence speed; no causal vocabulary-only claim.
- Speculative shortlist changes draft candidates, not the target model vocabulary or context length.

## 응답 생성 중 온도와 클럭

| 모드 | GPU 온도 중앙값/최대(°C) | SM 클럭 중앙값(MHz) |
|---|---:|---:|
| ko64k | 63.0/66.0 | 1989 |
| ko32k | 64.0/67.0 | 2086 |

10초 간격 관측이다. 손상된 이전 온도 로그 행은 제외했다. 클럭·온도 차이가 있다면 속도 차이의 교란 요인으로 보아야 한다.

## 32k 메모리 관측

로딩을 포함한 최소 가용 메모리 21.44GiB, 시작 대비 최대 추가 스왑 228.47MiB. 감시 중단과 컨테이너 OOM은 없었다.
후보 출력층 가중치는 160MiB로, 같은 BF16 64k 출력층의 320MiB보다 160MiB 작다. 이는 모델 전체 메모리 절반 감소를 뜻하지 않는다.

## 자료

- [개별 점수와 판정 이유](quality-summary.json)
- [코드 제출·기능 검사 결과](code-grades.json)
- [문서 집계 답안과 정답 대조](ledger-grades.json) / [정답 독립 검산](ledger-reference-audit.json)
- [전체 집계·토큰 범위·메모리 측정](comparison-summary.json)
- [32k 원문](ko32k-benchmark.json) / [64k 원문](ko64k-benchmark.json)
- [실행 계획](PLAN.md)
