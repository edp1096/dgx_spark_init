# Llama 3.2 3B Fine-tuning (DGX Spark)

* `라마` -> `스탠포드 알파카` 파인튜닝
* 출처: https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/pytorch-fine-tune


## 학습 방식

| 방식 | 학습 대상 | 싱글 | 클러스터 | 출력 |
|------|-----------|------|----------|------|
| **Full SFT** | 전체 파라미터 | O | O (FSDP2) | 전체 모델 |
| **LoRA** | 저랭크 어댑터 | O | O (FSDP) | 어댑터 |
| **QLoRA** | 4bit 양자화 + LoRA | O | X (FSDP 비호환) | 어댑터 |

## 호스트 폴더 구조 (양쪽 노드 동일)

```
/home/edp1096/workspace/finetune/
├── scripts/
│   ├── Llama3_full_finetuning.py
│   ├── Llama3_LoRA_finetuning.py
│   ├── Llama3_qLoRA_finetuning.py
│   └── configs/
│       ├── accelerate_full.yaml
│       └── accelerate_lora.yaml
├── output/
└── logs/
```

## 싱글 노드

```bash
export HF_TOKEN=your_token_here

# Full SFT
docker compose -f single/compose.full.yaml up

# LoRA
docker compose -f single/compose.lora.yaml up

# QLoRA
docker compose -f single/compose.qlora.yaml up
```

## 클러스터 (2x DGX Spark)

QSFP 직결. 양쪽 거의 동시에 `docker compose up`.

| 노드 | IP (QSFP) | 역할 |
|------|-----------|------|
| Head (gx10-bb75) | 169.254.141.58 | rank 0 |
| Worker (gx10-f05a) | 169.254.44.172 | rank 1 |

```bash
# Full SFT
# Worker: docker compose -f cluster/compose.full.worker.yaml up
# Head:   docker compose -f cluster/compose.full.head.yaml up

# LoRA
# Worker: docker compose -f cluster/compose.lora.worker.yaml up
# Head:   docker compose -f cluster/compose.lora.head.yaml up
```

## 벤치마크 (Llama 3.2 3B, dataset_size=512, 1 epoch)

| 방식 | 싱글 | 클러스터 |
|------|------|----------|
| Full SFT | 230.9s / loss 1.076 | 57.0s / loss 1.119 |
| LoRA | 100.9s / loss 1.167 | 55.2s / loss 1.478 |
| QLoRA | 107.2s / loss 1.273 | - |

## 트러블슈팅

| 증상 | 해결 |
|------|------|
| NCCL timeout | QSFP 케이블/IP 확인, `NCCL_DEBUG=INFO`로 로그 확인 |
| 메모리 부족 | `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| rendezvous timeout | 양쪽 pip install 시간 차이. 거의 동시에 실행 |
