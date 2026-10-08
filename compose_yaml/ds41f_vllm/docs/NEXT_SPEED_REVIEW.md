# 원본 DS41 추가 속도 후보 — 2026-09-13

## 실행 후 갱신

1. [Engram native reader](ENGRAM_READER.md): TP2 24개 요청 검증 통과,
   pp 중앙값 +5.45% / +6.62%. 소스 기본값에 채택했다.
2. [새 compact MoE](COMPACT_MOE_QUALIFICATION.md): 현재 TP2 크기에 해당
   경로가 적용되지 않는다. 전체 교체 후보도 수치 차이와 M=5 CUDA 오류로 제외했다.
3. [8192토큰 실행 범위와 dense 튜닝](PREFILL_WINDOW_8192.md)을 채택했다.
   reader와 합쳐 pp +52.77% / +47.85%, 긴 반복 입력의 초기 회귀도 해소했다.
   전체 층별 중단·재개 스케줄러를 이식한 것은 아니다.

아래는 실행 승인을 받기 전의 후보 조사 기록이다.

EXL3/Q2 전환은 사용자가 취소했다. 원본 MXFP4/E8M0, MXFP8 activation,
TP2, DSpark5, 224/128 expert 슬롯, Engram SSD를 유지하는 방향만 검토한다.
이번 작업은 코드·기존 결과 조사이며 새 GPU 벤치마크는 실행하지 않았다.
현재 실행 중인 Qwen/FLUX 세트도 중단하지 않았다.

## 이미 적용되거나 제외된 것

현재 고정 b12x는 `789bbb3c846565c41f3404af3e0d7c9ce8702f7f`다.
4096 scheduler / 2048 expert kernel / shared I/O / 384MiB scratch,
224개 startup preload는 이미 적용됐다.

- `081b2359` 패키지 교체는 실제 TP2 target/draft 비교에서 실패했다.
  5/6토큰과 2048토큰 연산이 느려졌고, 원본이 유지됐다.
- 실제 route staging은 약 1.2~1.5% pp 향상에 rank당 1.401GiB가 필요해 기본에서 제외했다.
- 예측 prefetch, 전체 모델 graphs, NCCL→RoCEnante 교체는 기존 측정에서
  이득이 없거나 주요 조건에서 느렸다. 같은 설정을 새 아이디어로 제안하지 않는다.

근거: [이전 커널 비교](PREFILL_ARCHITECTURE.md),
[4096 검증](PREFILL_4096_RETRY.md), [preload](EXPERT_PRELOAD.md),
[통신 비교](PERFORMANCE_ROUND2.md).

## 우선 후보: Engram의 CPU 읽기 처리

현재 `patches/engram.py`는 중복 행 제거 후 Python ThreadPoolExecutor로
작업을 나누고 각 행마다 `os.preadv`를 호출한다. 기본 스레드 32개,
prefill 작업당 최대 16행이다. 작은 읽기의 제출·완료 처리 비용을 줄일 여지가 있다.

[b12x c51d3c09](https://github.com/local-inference-lab/b12x/commit/c51d3c09fc7a)는
고정 fd 등록, io_uring 완료 일괄 처리, 큰 요청의 radix 정렬, 기존 scratch 재사용을
추가했다. 독립 reader 구현이므로 MoE 연산 방식을 바꾸지 않고 아이디어를 이식할 수 있다.

먼저 현재 읽기·CPU 정렬/제출·복원 시간을 분리한다. C에서 기존 buffered positional
read를 묶는 방식과 bounded io_uring 방식을 따로 비교한다. 현 구현은 페이지 캐시의
도움을 받으므로 O_DIRECT로 바꾸면 warm 조건에서 오히려 느려질 수 있다.
두 Engram 층과 두 rank offset에서 행/스케일 바이트가 같아야 하며,
cold/warm 및 1/5/6/4096토큰에 대해 실제 시간과 물리 읽기 바이트를 측정한다.

## 두 번째 후보: 새 compact MoE 커널의 선택적 이식

GitHub API로 확인한 master는 `323107ff948c` (2026-09-12 15:59 UTC)였다.
웹 커밋 목록 캐시에는 더 오래된 head가 보여 API로 재확인했다.
이전 불합격 `081b2359` 이후의 새 변경:

- [ac93b1e7](https://github.com/local-inference-lab/b12x/commit/ac93b1e7bb38):
  compact W4A8 direct micro를 복구, 8토큰 이하 micro / 이후 grouped M16 선택.
- [a55e9213](https://github.com/local-inference-lab/b12x/commit/a55e921324aa):
  compact MXFP8 scale 정렬 수정.
- [6475dacf](https://github.com/local-inference-lab/b12x/commit/6475dacfa1e4):
  W4A8 prefill의 SwiGLU limit 적용 수정.

다만 중간의 [15ec4b45](https://github.com/local-inference-lab/b12x/commit/15ec4b45f7a0)는
기존 `deepseek_v41` FP32 MoE recipe와 관련 reference를 제거한다.
따라서 최신 패키지를 통째로 설치해서 기존 수치 계약이 유지된다고 가정할 수 없다.
원본 가중치 보존과 중간 연산/합산 경계 보존은 별개다.
실제 M=1/5/6/8, target/draft, TP2 intermediate1152에서 먼저 배치와 수치 계약을
대조하고, 유지 가능한 커널/스케줄 변경만 별도 후보로 만든다.
성능 개선 폭은 아직 측정되지 않았다.

## 큰 변경 후보: layer-major prefill

기존 [아키텍처 조사](ARCHITECTURE_RESEARCH.md)의 Layered Prefill은 아직
완전한 V4.1 구현으로 이식하지 않았다. 긴 입력을 층별로 처리해 같은 expert의
반복 SSD 읽기를 줄이는 후보이나 mHC, KV 진행 상태, DSpark, TP 동기화를
다뤄야 한다. 단순 배치 크기 옵션 변경과 다르며 우선순위는 reader/커널 검사 뒤다.

진행 순서는 Engram 비용 계측 → 독립 reader A/B → compact 커널 수치·시간 비교다.
새 동작은 별도 테스트 이미지에서 검증하고 원본 serving에 바로 덮어쓰지 않는다.
