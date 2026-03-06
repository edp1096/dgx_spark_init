# Nanochat on DGX Spark

Karpathy's nanochat (GPT-2 grade) 학습 파이프라인.
데이터셋: ClimbMix-400B (karpathy/climbmix-400b-shuffle)

## 파일 구조

```
compose.download.yaml        # 모든 데이터 다운로드 + 토크나이저 (양쪽 노드 각각 1회)
compose.train.head.yaml      # pretrain → sft (rank 0, 2-node)
compose.train.worker.yaml    # pretrain → sft (rank 1, 2-node)
compose.train.single.yaml    # pretrain → sft (single node)
compose.eval.yaml            # base_eval → chat_eval (head 단독)
compose.shell.yaml           # interactive shell (chat_cli 등)
```

## 사전 준비

각 노드에 디렉토리 준비:
- `~/.cache/huggingface` — HuggingFace 캐시
- `~/.cache/nanochat` — 체크포인트, 데이터, 토크나이저, 평가 결과
- `~/workspace/nanochat` — nanochat 소스코드 (download 시 자동 clone + gb10 패치 + FA2 fallback 패치)

## Single (DGX Spark 1대)

### 1. 데이터 다운로드
`compose.download.yaml` 실행
- nanochat 소스 자동 clone + gb10/FA2 패치
- pretrain 데이터셋 (ClimbMix 240 shards)
- 토크나이저 학습/평가
- SFT 데이터 (identity_conversations.jsonl)
- base_eval 데이터 (eval_bundle, words_alpha.txt)
- chat_eval 데이터 (ARC, MMLU, GSM8K, HumanEval)

### 2. Pretrain + SFT
`compose.train.single.yaml` 실행
- pretrain 완료 후 자동으로 sft 시작

### 3. Evaluation
`compose.eval.yaml` 실행
- base_eval (CORE) → chat_eval (ChatCORE) 순차 실행

### 4. Interactive Shell (필요시)
`compose.shell.yaml` 실행
```bash
# 컨테이너 내부에서:
python -m scripts.chat_cli -i sft
```

## Cluster (DGX Spark 2대)

### 1. 데이터 다운로드 (양쪽 노드 각각)
`compose.download.yaml` 실행 (위와 동일)

### 2. Pretrain + SFT (양쪽 동시)
- Head: `compose.train.head.yaml` 실행
- Worker: `compose.train.worker.yaml` 실행
- pretrain 완료 후 자동으로 sft 시작

### 3. Evaluation (head만)
`compose.eval.yaml` 실행
- base_eval (CORE) → chat_eval (ChatCORE) 순차 실행

### 4. Interactive Shell (필요시)
`compose.shell.yaml` 실행
```bash
# 컨테이너 내부에서:
python -m scripts.chat_cli -i sft
```

## Volume 구조

| 호스트 경로 | 컨테이너 경로 | 용도 |
|---|---|---|
| `~/.cache/huggingface` | `/root/.cache/huggingface` | HuggingFace 캐시 |
| `~/.cache/nanochat` | `/root/.cache/nanochat` | 체크포인트, 데이터, 토크나이저, 평가 결과 |
| `~/workspace/nanochat` | `/root/nanochat` | nanochat 소스코드 (호스트에서 관리) |

## 체크포인트 위치 (~/.cache/nanochat/)

```
base_data_climbmix/       # ClimbMix pretrain 데이터셋
tokenizer/                # 토크나이저
base_checkpoints/d{N}/    # pretrain 모델 (N=depth)
chatsft_checkpoints/d{N}/ # SFT 모델
eval_bundle/              # CORE 평가 데이터
report/                   # 평가 결과
```

## Pretrain 시 메모리 / device-batch-size=64 대비 depth 적정값

* 주로 쓰이는 d20 / batch 32 에서 메모리 약 80GB, pretrain 33시간 가량 소요.
* d8 / batch 64 설정시 약 37분 소요.

| Args           | Prams | Ram    |
|----------------|:-----:|-------:|
| d8 / batch 64  | 125M  | ~57GB  |
| d16 / batch 64 | 537M  | ~106GB |

## 비고

- window pattern: SSSL (기본값, sliding window + full context 혼합)
- NGC PyTorch 26.02 컨테이너에 FA2 포함 → SSSL 네이티브 지원
- FA2 fallback 패치: flash_attention.py 교체 (FA3 → FA2 → SDPA 자동 전환)

## 참고

- Source: https://github.com/karpathy/nanochat
- Forum: https://forums.developer.nvidia.com/t/pretrain-nanochat-on-2-x-dgx-sparks/350564/1
- Gist (2-spark setup): https://gist.github.com/emaadmanzoor/d245c0c0ce90b25b4d50c0ffc448f876
- Gist (FA2 fallback): https://gist.github.com/edp1096/8670b744d88fddf89da0d0bc4ac56f95
