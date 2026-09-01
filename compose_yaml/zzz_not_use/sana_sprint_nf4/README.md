# SANA-Sprint 0.6B NF4 API

고속 T2I 후보를 폐기하지 않고 재현 가능한 형태로 보관하는 비활성 런타임입니다.
Gradio 없이 OpenAI 호환 이미지 생성 API만 제공합니다.

구성:

- `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers`
- Sana transformer: BF16
- Gemma2-2B text encoder: bitsandbytes NF4 double quantization
- DC-AE: BF16, 512px tile / 64px overlap
- 1024×1024, batch 1 실측 피크: 약 4.80 GiB
- 워밍 후 2-step 생성: 약 1.16초

이 런타임은 T2I 전용입니다. 범용 T2I/I2I 엔진은 FLUX.2 Klein 4B NVFP4를
사용합니다.

## 실행

```bash
docker volume create media-hf-cache
docker compose build
docker compose up -d
curl http://127.0.0.1:8707/health
```

## 요청

```bash
curl http://127.0.0.1:8707/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "sana-sprint-0.6b-nf4",
    "prompt": "A cinematic portrait in Seoul at night",
    "size": "1024x1024",
    "steps": 2,
    "seed": 12345,
    "response_format": "b64_json"
  }'
```

bitsandbytes 0.50.2의 CUDA 13.2 라이브러리에는 GB10용 `sm_121` cubin과
전용 4-bit GEMM dispatch가 포함되어 있습니다. CUDA 13.3용 재컴파일은 필수 조건이
아닙니다.
