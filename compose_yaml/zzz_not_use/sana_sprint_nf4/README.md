# SANA-Sprint 0.6B NF4 API

속도·메모리 비교 결과를 재현하기 위해 보관하는 비활성 런타임입니다. Gradio 없이
OpenAI 호환 T2I와 SANA-Sprint용 실험적 인페인트·아웃페인트 API를 제공합니다.

구성:

- `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers`
- Sana transformer: BF16
- Gemma2-2B text encoder: bitsandbytes NF4 double quantization
- DC-AE: BF16, 512px tile / 64px overlap
- 1024×1024, batch 1 실측 피크: 약 4.25 GiB
- 워밍 후 2-step 생성: 약 1.16초
- 인페인트 2-step 실측: 약 2.19초
- 오른쪽 512px 아웃페인트 2-step 실측: 약 3.11초

공식 SANA 인페인트 구현은 few-step Sprint 스케줄러를 지원하지 않습니다. 이
런타임은 매 디노이즈 단계에서 마스크 바깥의 원본 latent를 복원하고 마지막에 원본
픽셀을 합성하는 별도 inference-only 방식을 사용합니다. 2-step 특성상 좁은
마스크에서는 기존 물체 흔적이 경계에 남을 수 있으므로 `overlap`과 `feather`를
조절해야 합니다.

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

## 인페인트와 아웃페인트

마스크의 흰색 부분을 새로 생성하고 검은색 부분은 보존합니다.

```bash
curl http://127.0.0.1:8707/v1/images/edits \
  -F image=@source.png -F mask=@mask.png \
  -F prompt='replace the cabin with a blue tent' \
  -F size=1024x1024 -F steps=2 -F seed=12345 \
  -o response.json

curl http://127.0.0.1:8707/v1/images/outpaint \
  -F image=@source.png \
  -F prompt='continue the alpine lake and mountains naturally' \
  -F right=512 -F overlap=96 -F feather=32 \
  -F steps=2 -F seed=12345 \
  -o response.json
```

응답은 OpenAI 이미지 API와 같은 `data[0].b64_json` 형식이며, 실측 시간과 CUDA
allocator 피크를 `metrics`에 함께 반환합니다. 동시에 여러 요청이 들어오면 한 장씩
직렬 처리해 통합메모리 피크를 제한합니다.

bitsandbytes 0.50.2의 CUDA 13.2 라이브러리에는 GB10용 `sm_121` cubin과
전용 4-bit GEMM dispatch가 포함되어 있습니다. CUDA 13.3용 재컴파일은 필수 조건이
아닙니다.
