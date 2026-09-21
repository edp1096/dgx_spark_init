# QAD TP1 MTP 최적화 검증

DGX Spark GB10에서 작은 출력 행렬과 MTP 상태 저장 비용을 비교한다.
운영 기본값은 유지하고 아래 두 경로를 실험 옵션으로 연결했다.
공개 `sm121-v5` 이미지는 그대로다. 아래 옵션에는 수정 소스로 만든 이미지가 필요하다.

- `SGLANG_QAD_DRAFT_GEMV=1`: BF16 `[65536, 2560]` 초안 헤드의 1행 계산에 B12X GEMV 사용.
- `SGLANG_QAD_REPLAY_VERIFY=1` + `--enable-linear-replayssm-spec --mamba-ssm-dtype float32`:
  SGLang의 GDN ring-write 검증과 accepted-prefix fold 사용. B12X prefill/decode는 유지한다.
  decode용 ReplaySSM ring과 디스크 HiCache는 이 검증 대상이 아니다.

Qwen Compose 디렉터리에서 시험 이미지를 빌드한다:

```sh
docker build -f experiments/ninfer-study/Dockerfile -t dgx-sglang-qad-ninfer:trial .
```

## 구현 참고

- NInfer `9e163eee4b8acec21ab0ac765107b6a3f287b217`: 작은 토큰 수별 Linear 튜닝, raw-input ReplaySSM 설계.
- NInfer-4090 `5c60b7c9b455231795c09da21a9fbb6aa53f08e5`: 작은 MTP 행렬의 Tensor Core 경로.
- 실제 실행: 기존 `sm121-v5`의 B12X BF16 GEMV와 SGLang GDN ReplaySSM.

## 단위 검사

`bench_head.py`: BF16 출력 헤드에서 PyTorch와 B12X를 CUDA Graph로 비교한다.
`bench_gdn.py`: GDN 한 레이어의 B12X snapshot과 SGLang ring/fold를 비교한다.
전체 모델 속도나 전체 레이어 지연으로 환산하지 않는다.
`test_replay.py`: 32라운드의 모든 수락 길이와 접두사 추적 상태를 native snapshot과 비트 단위 비교한다.

원시 결과: `results/head.jsonl`, `results/gdn.jsonl`.
합성 입력에서 B12X와 native GDN 사이에는 반올림 차이가 있으며,
native snapshot과 native replay의 일치는 기존 B12X 출력과의 일치를 뜻하지 않는다.

## 서버 검사

`serve_probe.py --label NAME --output-dir DIR`: 한국어·코드·JSON·영문 응답,
반복 요청, 긴 접두사 재사용 및 대화 분기. 전용 시험 서버의 URL을 지정한다.
`media_tools_probe.py`: 도구 호출 파싱과 이미지·WebM 응답. 반환된 도구는 실행하지 않는다.

## 실측 결과 (GB10, 2026-09-22 KST)

| 항목 | 기존 | 실험 옵션 2개 활성화 |
| --- | ---: | ---: |
| 한국어 256토큰 HTTP 완료 중앙값 | 8.458초 | 8.169초 |
| 코드 256토큰 HTTP 완료 중앙값 | 6.010초 | 5.789초 |
| MTP 중간 SSM 상태 | 약 0.844GiB | 없음 |
| ReplaySSM 기록 | 없음 | 약 0.020GiB |

첫 요청을 제외한 3회 중앙값이다. 출력 길이만 고정했고 생성 내용·MTP 수락률은 달랐다.
반복 변동이 있어 일반적인 속도 향상이나 두 변경 각각의 서버 기여도를 확정하지 않는다.
모델 가중치와 KV 정밀도는 유지했다.

- 기존 GDN 회귀 검사 7개 통과.
- native snapshot/replay의 출력·상태·접두사 추적을 32라운드 비트 단위 비교: 통과.
- 기존/실험 모델 각각 한국어 산술·JSON·코드 문법·접두사 회상/재사용/분기·도구·이미지·WebM 검사 통과.
- 1M KV 풀을 할당했으며 이번 최대 실제 입력은 32,041토큰이다.
  1M 전체 입력과 Flux·ASR·TTS 동시 실행은 이번 변경으로 재검증하지 않았다.
- 운영 기본값과 공개 이미지는 유지했다. 시험 서버는 종료했다.

서버 요약은 `results/serving.json`, 설정·이미지·소스 식별자는 `results/provenance.json`과
`results/*-config.json`에 있다. 원시 응답을 같은 폴더에 보관했다.
