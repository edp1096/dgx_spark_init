# SGLang TP2와 긴 문맥 검증

기존 `compose.yaml`은 TP1용이다. TP2는 원본 NVFP4 체크포인트와 출력층 최적화 이미지를 사용하는 `compose.tp2.yaml`과 `manage_tp2.py`로 기동한다. 두 노드의 NCCL RoCE 통신과 TP별 PLE 파일 분할을 사용한다. SparkTalk에는 `Qwen3.8 Flash-Next TP2 · 1M` 세트로도 등록되어 있다. API 포트는 8012다.

TP1은 이제 QAD 체크포인트를 사용하지만, 이 TP2 경로는
`edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4`를 유지한다.

## 준비

두 노드에서 같은 경로에 체크포인트와 같은 ID의 `dgx-sglang-qwen38-fn:sm121-tp2-vocab-v1` 이미지가 필요하다. 워커 기본 주소는 `edp1096@192.168.100.60`, 통신망은 `10.200.0.1/2`, 인터페이스/HCA는 `enp1s0f1np1`/`rocep1s0f1`이다. 관리 스크립트는 소스만 동기화한다. 모델과 이미지의 최초 복사는 별도로 한다.

## 기동·종료

이 디렉터리의 상위 모델 디렉터리에서 실행한다. 다른 LLM을 먼저 종료한다.

```bash
python3 manage_tp2.py start --context 262144
python3 manage_tp2.py status
python3 manage_tp2.py stop
```

`--context`는 `262144`, `524288`, `1048576`을 받는다. 512K/1M은 YaRN 배율 2/4를 런타임 오버라이드로 적용하는 실험 설정이다. 체크포인트 파일은 읽기 전용이며 변경하지 않는다. 설정 수락·KV 풀 할당만으로 실제 긴 입력 처리나 품질이 검증되는 것은 아니다.

KV는 BF16, 동시 요청 1개, NEXTN은 3/1/4다. ko64k 초안 어휘는 전역 토큰 소유권에 따라 선택한 뒤 shortlist 순서로 32,768개씩 재분할한다. 본체의 전체 어휘 검증과 모델 가중치는 유지한다. `SPARKTALK_FLASH_NEXT_DRAFT_VOCAB=off`로 비활성화할 수 있다. 프리필 청크 기본값은 1024다. 2048·4096·8192는 검색 응답 오류가 나와 채택하지 않았다. [실측 기록](TP2_TUNING_20260915.md)을 참고한다.

## 메모리 감시

기동 전 두 노드에서 cold filesystem cache를 반환하고 각각 110 GiB 이상의 MemAvailable을 요구한다. 호스트 watchdog은 0.5초 간격으로 관측하며 가용 8 GiB 미만 또는 컨테이너 cgroup 98 GiB 초과 시 해당 시험 컨테이너를 종료한다. Docker 메모리 한도는 104 GiB, swap 추가 사용은 금지한다. GPU 통합 메모리가 모두 cgroup에 반영된다고 가정하지 않으므로 호스트 가용량을 함께 감시한다. 급격한 할당이나 하드웨어 장애까지 예방하는 보장은 아니다.

로그는 각 노드 `~/.local/state/qwen38-tp2/<token>/`에 남는다. 한 노드가 실패하면 `manage_tp2.py stop`으로 양쪽을 정리한다. 자동 재시작은 꺼져 있다.

## 실제 입력 시험

`tp2/long_probe.py`는 실제 모델 tokenizer로 문맥 한도보다 2,048토큰 적은 입력을 만들고 10%, 50%, 90% 지점의 임의 값을 조회한다. 서버의 prompt token 수가 제출한 수와 같은지 검증한다. 자동 입력 절단은 허용하지 않는다. TTFT는 요청 송신 시작부터 첫 응답 텍스트까지이며, pp_estimate는 입력 토큰/TTFT라 순수 커널 prefill 처리율과 다르다. 세 값 회수 성공은 제한된 검색 시험 통과이며 전반적인 장문 품질 보장은 아니다.

기동 완료를 기다리고 시험·로그 수집을 실행하려면 같은 문맥 값으로 다음을 실행한다. 실패 시 양쪽 TP2 컨테이너를 종료한다. 결과는 `bench/results/<token>/`에 저장한다.

```bash
python3 tp2/run_probe.py --context 262144
```

## 2026-09-14 실측

원본 NVFP4 가중치, BF16 KV, TP2, NEXTN 3/1/4, prefill chunk 1024, cold prefix 입력으로 측정했다. 각 문맥 크기당 1회다.

| 설정 | 실제 입력 토큰 | TTFT | 입력/TTFT | 조회 정확도 | 최소 가용 메모리 헤드/워커 |
|---|---:|---:|---:|---:|---:|
| 256K | 260,096 | 137.35초 | 1,893.7 tok/s | 3/3 | 51.38 / 53.53 GiB |
| 512K · YaRN 2 | 522,240 | 339.41초 | 1,538.7 tok/s | 3/3 | 47.82 / 50.11 GiB |
| 1M · YaRN 4 | 1,046,528 | 1,098.87초 | 952.4 tok/s | 3/3 | 39.50 / 41.85 GiB |

세 시험 모두 `cached_tokens=0`, `num_retractions=0`, 입력 절단 없음, watchdog trip/OOM 없음. 약 1초 길이의 짧고 예측하기 쉬운 JSON 답변에서 서버 decode_throughput은 각각 59.65 / 63.65 / 58.27 tok/s였다. 일반 대화·코딩의 생성 속도로 일반화하지 않는다.

원시 결과: [256K](../bench/results/tp2-256k-20260914/retrieval.json), [512K](../bench/results/tp2-512k-20260914/retrieval.json), [1M](../bench/results/tp2-1m-20260914/retrieval.json). 각 폴더에 서버 설정·랭크별 로그·메모리 관측을 함께 보관한다.

## FP8 / NVFP4 KV 후보

BF16은 TP2·문맥 확장의 기준 시험값이다. KV 양자화를 금지하는 모델이라는 뜻은 아니다. 같은 계열의 [별도 Spark TP2 구현](https://github.com/maci0/qwen3.8-flash-next-spark/blob/main/docs/sglang-deployment.md)은 `fp8_e4m3`와 별도 QSA 패치의 NVFP4 KV를 사용한다. 현재 고정 이미지의 일반 dtype 파서가 `nvfp4`를 받아도 QSA 경로까지 지원한다는 증거는 아니므로, 옵션만 바꿔 운영 설정으로 채택하지 않는다.

1M 시험의 풀 1,114,112토큰 기준, 랭크당 본체 K/V 12.75 GiB + MTP K/V 1.0625 GiB다. 이 부분만 BF16→FP8이면 이론상 약 6.91 GiB/rank 절약된다. Mamba 상태와 QSA 인덱스, 모델 가중치, 그래프·작업 버퍼는 별도다. FP8·NVFP4의 실제 속도와 정확도는 이 BF16 결과로 대신 검증할 수 없다.


세 단계 검증 후 시험용 TP2 컨테이너는 관리 명령으로 양쪽 모두 종료했다. 최종 종료 코드는 head 137 / worker 143이며, 측정 중 실패가 아닌 시험 완료 후 stop 단계에서 기록됐다. 두 컨테이너의 OOMKilled는 false다. TP1 Compose는 유지하고 SparkTalk에는 별도 TP2 세트를 추가했다. TP2 기본값은 네이티브 256K이며, 1M을 선택하려면 명시적으로 실행한다.

```bash
python3 manage_tp2.py start --context 1048576
```

관리 스크립트는 매번 새 시험 식별자를 사용한다. 수동 `--token`도 이전 기동과 겹치면 거절한다. 마지막 기동 정보는 소스 폴더가 아닌 `~/.local/state/qwen38-tp2/last-start.json`에 보관한다.


## B12X 출력층 적용 (2026-09-14)

이 최적화의 기반 이미지는 `sm121-b12x-head-v1`이다. 기존 `sm121-vocab1` 이미지를
준비한 뒤 이 디렉터리에서 다음으로 빌드하고 워커에도 같은 이미지를 복사한다.

```sh
docker build -f Dockerfile.b12x-head -t dgx-sglang-qwen38-fn:sm121-b12x-head-v1 .
```

BF16 출력층의 1토큰 계산만 B12X로 바꾼다. 가중치/출력 dtype과 TP 처리는
유지하고 MoE·QSA·GDN은 기존 경로를 사용한다. 512/4K/16K 반복 시험에서
출력 문장은 같았으며 tg가 약 3.5~3.8% 개선됐다. pp 개선은 없었다.
1M 입력 1,046,528토큰의 세 위치 값도 3/3 회수했고 TTFT는 18분 21초였다.
[상세 실측](B12X_SGLANG_REVIEW.md)을 참고한다.

기존 동작으로 되돌리려면 `SPARKTALK_FLASH_NEXT_DRAFT_VOCAB=off QWEN_TP2_IMAGE=dgx-sglang-qwen38-fn:sm121-vocab1`
환경변수로 관리 명령을 실행한다. TP1 기본 이미지는 변경하지 않는다.

현재 TP2 이미지는 위 기반에 TP2 shortlist 처리를 추가한 것이다.

```sh
docker build -f Dockerfile.tp2-vocab -t dgx-sglang-qwen38-fn:sm121-tp2-vocab-v1 .
```

워커에도 같은 이미지 ID를 복사한다. TP1 설정은 이 변경 대상이 아니다.

## 재부팅 후 RoCE 주소 복원

관리 스크립트의 `start`는 모델 실행 전에 양쪽 RoCE IPv4 주소를 확인하고,
주소가 없는 활성 포트에만 다시 할당한 뒤 양방향 ping으로 확인한다.
다른 IPv4 주소가 있거나 케이블이 연결되지 않았으면 변경하지 않고 실패한다.
영구 Netplan 설정은 필요하지 않으며 `python3 manage_tp2.py network`로 모델을 띄우지 않고
네트워크 준비만 실행할 수도 있다. 워커 SSH 주소는 재부팅 후에도 접근 가능한
관리 LAN 주소를 사용해야 한다. Docker 직접 실행은 이 절차를 거치지 않는다.
Qwen 주소·인터페이스는 `QWEN_TP2_HEAD`, `QWEN_TP2_WORKER_RAIL`, `HEAD_NCCL_IF`, `WORKER_NCCL_IF`, `NCCL_SUBNET`을 사용한다.
