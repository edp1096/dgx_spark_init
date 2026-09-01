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

## 2026-09-01 초경량 런타임 비교

SparkTalk용 Nemotron Q8 C++ 런타임과 초경량 후보를 동일한 FLEURS 표본으로
비교했다. 한국어는 기존과 동일한 20개, 영어는 길이 분위수로 선택한 10개다.
CER 정규화 방식도 위 비교와 같다.

| 모델·런타임 | 한국어 CER | 영어 CER | 한국어 배속 | 실행 메모리 | 비고 |
|---|---:|---:|---:|---:|---|
| Nemotron 3.5 ASR 0.6B Q8, NeMo-Speech.cpp CUDA | **10.67%** | 5.67% | **208.0x** | GPU 약 2.1GiB + RSS 617MiB | 32개 즉시 사용 locale, 자동 감지·문장부호·스트리밍 |
| SenseVoiceSmall Q8, llama.cpp CPU | 11.40% | **5.31%** | 44.1x 순수 추론 | 최대 RSS 약 **274MiB** | 한·영·일·중·광둥어, VAD 구간 SRT |
| OmniASR CTC 300M INT8, LiteRT CPU | 18.31% | 8.73% | 37.4x | 최대 RSS 약 1.10GiB | 1,600+ 언어, 커뮤니티 변환본, 고정 10초 입력 |

SenseVoice의 CLI 전체 지연은 매 요청마다 모델을 다시 적재하여 평균 0.424초지만,
로그가 보고한 순수 추론 평균은 0.277초였다. 상주 서버로 만들면 적재 비용은
제거할 수 있다. 한국어 정확도 차이는 0.73%p에 불과하고 메모리는 크게 줄지만,
지원 언어가 다섯 개뿐이므로 언어를 알 수 없는 미디어까지 하나의 모델로 처리하는
SparkTalk 기본값은 Nemotron을 유지한다. SenseVoice는 메모리가 빠듯해졌을 때
선택 가능한 경량 provider로 추가할 가치가 있다.

OmniASR 공식 fairseq2 런타임은 Linux ARM64용 `fairseq2n` 휠을 제공하지 않아,
DGX Spark에서는 커뮤니티 LiteRT INT8 변환본을 사용했다. 넓은 언어 범위와 달리
이번 한국어·영어 표본에서는 정확도가 낮고 SenseVoice보다 메모리도 커서 채택하지
않는다.

초경량 CLI 벤치:

```bash
python3 bench_cli.py /path/to/corpus results/sensevoice.jsonl -- \
  /path/to/llama-funasr-sensevoice \
  -m /path/to/sensevoice-small-q8.gguf \
  --vad /path/to/fsmn-vad.gguf -a '{audio}' --keep-tags
```

OmniASR LiteRT 벤치는 호스트에 패키지를 설치하지 않고 `uv` 캐시 환경에서 실행한다.

```bash
uv run --with ai-edge-litert --with sentencepiece --with numpy --with soundfile \
  python bench_omniasr_litert.py /path/to/corpus results/omniasr.jsonl \
  --model /path/to/omnilingual-ctc-300m.tflite \
  --tokenizer /path/to/tokenizer.model --threads 8
```
