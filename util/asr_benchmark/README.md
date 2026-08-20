# ASR comparison benchmark

DGX Spark에서 다음 세 모델을 동일한 한국어 음성으로 비교합니다.

- `Qwen/Qwen3-ASR-1.7B-hf`
- `CohereLabs/cohere-transcribe-03-2026`
- `nvidia/nemotron-3.5-asr-streaming-0.6b`

## 2026-08-19 결과

FLEURS `ko_kr` test에서 길이 분위수로 고정 선택한 20개 음성(총 244.56초)을
한 건씩 순차 요청했습니다. 첫 요청은 워밍업으로 제외했고, CER는 NFKC 변환 후
영숫자만 남겨 계산했습니다.

| 모델 | CER | 평균 지연 | p95 지연 | 실시간 배속 | 콜드 기동+20초 요청 전체 피크 |
|---|---:|---:|---:|---:|---:|
| Qwen3-ASR 1.7B | **6.28%** | 1.010초 | 1.592초 | 12.1x | 8.59GiB |
| Cohere Transcribe 2B | 9.00% | 0.168초 | 0.225초 | 73.0x | 7.74GiB |
| Nemotron 3.5 ASR 0.6B | 10.56% | **0.116초** | **0.152초** | **105.8x** | **4.80GiB** |

전체 피크는 모든 ASR 엔진을 내린 상태의 `/proc/meminfo` `MemAvailable`을
기준으로, 캐시된 모델의 컨테이너 기동부터 20초 음성 한 건을 처리할 때까지
0.1초 간격으로 측정한 감소량입니다. 통합 메모리 환경이므로 Docker RSS보다 이
값을 용량 계획에 사용합니다.

표본이 20개뿐이고 FLEURS 일부 참조문에는 실제 발화와 달라 보이는 영문 병기나
문장 후반부가 포함되어 절대 CER보다 동일 표본의 상대 비교에 의미가 있습니다.
현재 1인용 앱의 기본값은 Qwen3-ASR 1.7B가 적합합니다. 12배속이면 충분히
빠르면서 정확도가 가장 좋았습니다. Nemotron은 향후 실시간 스트리밍 지연과
메모리가 정확도보다 중요한 모드의 후보로 유지합니다.

## 재현

```bash
python3 prepare_fleurs.py /path/to/fleurs-ko-test.parquet data/fleurs-ko

python3 bench_api.py data/fleurs-ko results/qwen.jsonl \
  --endpoint http://127.0.0.1:8694/v1/audio/transcriptions \
  --model Qwen/Qwen3-ASR-1.7B-hf
```

`bench_api.py`의 endpoint와 model을 바꾸면 나머지 OpenAI 호환 ASR API에도 같은
측정을 적용할 수 있습니다.
