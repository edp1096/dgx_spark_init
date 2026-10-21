# Pilcothink 가속 구성 재현

2026-09-27 측정. 기존 eager/no-spec vLLM 4.2t/s 대 SGLang 15.6t/s 비교는 고속 공개 구성과의 비교가 아니었다.

NVIDIA 원본 리비전 `09b04e5e74bca08ca8549fc736d4cdd8624bfde3`, Pilcothink 0.28 ARM64 이미지 `sha256:09da6eb394216d174ab8692758d90f9f458398d9c8fbc11ba6f04e93d5cf6392`, incoai DFlash2 리비전 `bf582e4eacc1810f76656d1811693ff6c6737d2a`로 재현했다. CPU/GPU 클럭 제한은 기존 그대로 유지했다.

TP2, 1,048,576 문맥, FP8 KV 10,240,000,000바이트/호스트, DFlash K5, async scheduling, CUDA graph 최대 16, batch 1024, seqs 4. 실제 KV 용량은 1,172,644토큰, 메인 graph 메모리는 0.67GiB. AGPL display-KV 코드는 추가하지 않았다.

워밍업 뒤 출력 128토큰의 생성 속도(TTFT 제외)는 일반 문장 27.5–32.4t/s, 코드 36.8–44.2t/s, 구조화 출력 50.1–52.4t/s였다. [원시 보고서](spec-measured.md). 도구는 tool-eval-bench `6be685f0e6b9e0df05ed024848cf7fe1eca48752`이다. 문맥 깊이 0/2048/8192는 filler에만 적용되며 코드/구조화는 보고서의 depth=0이다. 출력이 짧으므로 장시간 생성 성능이나 품질 검증을 대신하지 않는다.

이는 NVIDIA 원본 결과이며 Huihui 변환본과 SGLang의 속도·품질을 보장하지 않는다. 원본과 이전 Huihui 측정의 차이를 가속 옵션 하나의 효과로 단정하지 않는다. 수정된 입력 형식으로 실제 1,047,622토큰 cold 입력에서 5개 코드를 모두 정확한 JSON으로 회수하고 정상 종료했다(1,129.95초, 출력 61토큰). 첫 시도는 원본 템플릿이 enable_thinking=false를 무시하여 설명 출력 중 512토큰 한도에 걸렸고 JSON 형식 검증에 실패했다. 재검증은 assistant `<think></think>` prefill과 continue_final_message를 사용했다. 이후 기동 레시피에는 기존 템플릿 보정기를 연결했다. 서버가 cached_tokens 상세를 반환하지 않아 cold 조건은 요청마다 고유 cache_salt를 사용해 확보했다. 시작 과정에 양쪽 스왑 쓰기가 관찰되어 무스왑 구성으로도 판정하지 않았다. 아직 production compose/Talk 기본값 변경 근거로 삼지 않는다.

재현 중 CLI 차이: 이 이미지의 `--per-request-spec-decode-metrics`는 `detailed` 등의 값을 요구한다. prefix retention은 포럼의 재사용 문제 해결 제안에 따라 4608을 사용했으며 실제 긴-prefix 재사용 검증은 별도로 필요하다.

시작부터 첫 1M까지 누적 스왑 쓰기는 메인 약 5.84GiB, 워커 약 3.69GiB. 수정된 두 번째 1M 실행에서는 추가 쓰기가 없었으나 최초 실행의 스왑 문제는 해결되지 않았다. 메인 최소 가용 메모리는 약 1.18GiB였다.

14,423토큰 동일 입력 두 번의 총 응답시간은 11.85초→4.20초로 감소했고 두 출력은 일치했다. 응답에는 cached_tokens 상세가 없으므로 이 측정만으로 정확한 재사용 토큰 수를 단정하지 않는다.
