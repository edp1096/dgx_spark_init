# Weights override

원본 BF16·원본 양자화·ablit BF16을 비교해, 변경된 가중치만 반영한 ablit 양자화 모델을 만든다.

```bash
python3 convert.py \
  --original /models/original-bf16 \
  --base /models/original-nvfp4 \
  --donor /models/ablit-bf16 \
  --profile ornith_15 \
  --activation-scales preserve \
  --output /models/ablit-nvfp4
```

프로필: `fp8`, `ornith_15`, `gemma4_26b`, `qwen38_huihui`. 검사만 하려면 `--check-only`.

Qwen GGUF 경로도 같은 명령을 쓴다. GGUF·RadixArk 입력은 기본 HF 캐시에서 찾는다.

```bash
python3 convert.py --profile qwen38_huihui \
  --source-bf16 /models/qwen38-original-bf16 \
  --activation-scales preserve --output /models/qwen38-ablit-nvfp4
```

Qwen은 기존 로컬 Docker 변환 환경을 사용한다. 파일 다운로드·모델 기동은 하지 않는다.
원본은 보존하며, 변환 후 구동 검증은 별도다.

LIL QAD의 혼합 MXFP8/NVFP4 구조에 Huihui GGUF 변경량을 반영하는 별도 후보 변환기는
[`model_adapters/qwen38_lil_huihui`](model_adapters/qwen38_lil_huihui/README.md)에 있다.
기존 RadixArk 프로필과 입력·재양자화 방식이 다르며, 실행 검증 전에는 운영 모델로 사용하지 않는다.
