# ko16k · ko32k · ko64k 실제 비교

동일한 24개 문항을 두 번씩 실행한 48쌍이다. 16k만 새로 실행하고 32k·64k의 저장 결과를 대조했다. 입력 토큰·생성 설정·이미지와 기존 64k 채점 유지 여부를 검증했다.

16k 재시험은 2026-09-09 03:15~04:11 KST에 48개 응답을 모두 생성하고 정상 종료했다. 이번 재시험 중 재부팅은 없었다. 앞선 로딩 실패 기록은 별도 보존되어 있으며, 이번 성공만으로 호스트 안정성이 입증되지는 않는다.

16k는 합산 decode 30.09 tok/s로 32k 30.22, 64k 30.18과 거의 같다. 같은 문항 속도비의 기하평균은 32k 대비 +2.02%, 64k 대비 +4.37%지만, 긴 응답까지 합산한 처리량 이득은 없다. 특히 64k의 관측 클럭 중앙값은 1989 MHz, 16k는 2093 MHz로 달라 어휘 축소만의 인과적 속도 개선이라 결론 내릴 수 없다.

고정 기준 점수는 16k 98/144, 32k 97/144, 64k 99/144로 좁은 범위다. 16k는 코드 2/18로 다른 두 설정의 4/18보다 낮고, 서술형 중요 오류도 16건으로 각각 14건보다 많다. 점수 1~2점 차이로 품질 우열을 확정할 규모는 아니다. 모든 설정의 최종 답변 미제출 6건도 그대로 실패로 포함했다. 코드 실행 대상 네 응답은 모두 추론 중 한도에 도달하여 제출 코드가 없었고 실행 검사는 생략됐다.

확실한 차이는 초안 후보 출력층의 가중치 크기다. 16k는 80 MiB로 32k보다 80 MiB, 64k보다 240 MiB 작다. 반면 초안 수락률은 46.10%로 내려간다. 이는 모델 전체 어휘나 65,536 토큰 컨텍스트 축소가 아니며, 전체 메모리가 같은 비율로 줄어드는 것도 아니다.

메모리 관측: 로딩 중 최소 가용 21.42 GiB, 생성 중 최소 26.64 GiB, 최대 추가 swap 269.46 MiB. 임시 서버를 종료한 뒤 가용 메모리는 약 117 GiB로 회복했다. 운영 컨테이너는 기존 정지 상태이고 지원 서비스 네 개는 정상 상태다.

채점 보충: 저혈당 두 번째 응답은 고정 세 기준을 통과했지만 의식 소실 시 반좌위 지시를 별도 중요 오류로 표시했다. [NHS 저혈당 안내](https://www.nhs.uk/conditions/low-blood-sugar-hypoglycaemia/)의 회복자세 지침을 추가 대조했으며, 세 기준 및 기존 32k·64k 점수는 변경하지 않았다.

| 항목 | ko16k | ko32k | ko64k |
|---|---:|---:|---:|
| 고정 기준 점수 | 98/144 | 97/144 | 99/144 |
| 3개 기준 모두 통과 | 23/48 | 22/48 | 24/48 |
| 서술형 중요 오류 표시 | 16 | 14 | 14 |
| 최종 답변 미제출 | 6 | 6 | 6 |
| 출력 한도 도달 | 13 | 15 | 14 |
| 합산 decode tok/s | 30.09 | 30.22 | 30.18 |
| 합산 전체 응답 tok/s | 29.67 | 29.78 | 29.73 |
| 문항 속도의 산술평균 tok/s | 34.23 | 33.39 | 32.62 |
| 첫 토큰 지연 중앙값(초) | 0.262 | 0.263 | 0.293 |
| 초안 수락률 | 46.10% | 48.40% | 51.77% |
| 생성 토큰 수 | 73499 | 72178 | 73226 |
| 응답 대기시간 합계(분) | 41.28 | 40.39 | 41.05 |
| 후보 출력층 가중치(MiB) | 80 | 160 | 320 |
| 관측 SM 클럭 중앙값(MHz) | 2093 | 2086 | 1989 |
| 관측 GPU 온도 중앙값/최대(°C) | 64/66 | 64/67 | 63/66 |

합산 처리량과 문항별 평균은 가중 방식이 다르다. 중요 오류 표시는 수동 서술형 검토의 별도 지표이며 코드/집계 실패는 점수로 반영한다.

## 같은 요청끼리 속도·점수 비교

| 비교 | 속도비 기하평균 | 양쪽 완료 쌍 수 | 완료 쌍 속도비 | 후보 점수 우세/열세/동점 |
|---|---:|---:|---:|---:|
| ko16k_over_ko64k | 1.0437 | 33 | 1.0618 | 5/5/38 |
| ko16k_over_ko32k | 1.0202 | 32 | 1.0232 | 8/7/33 |
| ko32k_over_ko64k | 1.0230 | 32 | 1.0404 | 5/6/37 |

## 분야별 비교

| 분야 | 쌍 수 | 16k 점수 | 32k 점수 | 64k 점수 | 속도비16/32 | 속도비16/64 |
|---|---:|---:|---:|---:|---:|---:|
| coding | 6 | 2/18 | 4/18 | 4/18 | 0.943 | 0.979 |
| documents | 4 | 6/12 | 6/12 | 6/12 | 1.054 | 1.075 |
| economics | 4 | 10/12 | 10/12 | 11/12 | 1.090 | 1.072 |
| everyday | 6 | 16/18 | 16/18 | 16/18 | 1.038 | 1.088 |
| humanities | 2 | 5/6 | 5/6 | 5/6 | 0.993 | 0.986 |
| language | 4 | 12/12 | 12/12 | 12/12 | 1.004 | 1.065 |
| medicine | 10 | 23/30 | 21/30 | 22/30 | 1.023 | 1.044 |
| planning | 4 | 8/12 | 8/12 | 6/12 | 1.094 | 1.122 |
| research | 4 | 6/12 | 3/12 | 6/12 | 0.980 | 0.952 |
| science | 4 | 10/12 | 12/12 | 11/12 | 1.008 | 1.050 |

## 한계

- Sequential runs across boot sessions; another host reboot occurred after the 32K trial. Clock/thermal/order/output-length effects remain.
- An earlier 16K load ended with host resets and zero responses. These metrics describe the explicitly requested retry only, not proof of hardware stability.
- 48 requests are two repeats of 24 task types, not 48 independent scenarios.
- Prose uses frozen assistant-rated rubrics, not independent expert judgments.
- Only draft vocabulary changes; target vocabulary/context is not reduced.
- Token-limit and missing-final failures retained. No selected extended-budget requests are pooled.

## 상세 자료

- [3개 설정 집계 JSON](three-way-summary.json)
- [3개 설정의 144개 응답 점수·판정 근거](three-way-quality.json)
- [16k·64k 개별 점수와 근거](quality-summary.json)
- [32k 개별 점수와 근거](../2026-09-09-ko32k-vs64k/quality-summary.json)
- [16k 원문](ko16k-benchmark.json) / [32k 원문](../2026-09-09-ko32k-vs64k/ko32k-benchmark.json) / [64k 원문](ko64k-benchmark.json)
- [16k 메모리 관측](ko16k-memory.json) / [코드 검사](code-grades.json) / [원장 대조](ledger-grades.json)
- [실행 계획](PLAN.md)
