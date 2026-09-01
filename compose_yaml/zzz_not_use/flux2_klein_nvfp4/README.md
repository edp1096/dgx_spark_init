# FLUX.2 Klein 4B NVFP4 API

공식 `black-forest-labs/FLUX.2-klein-4b-nvfp4` transformer와 uncensored Qwen
텍스트 인코더를 사용하는 이미지 생성 엔진입니다. 기본 텍스트 인코더는 공식 ComfyUI
혼합정밀 배치를 템플릿으로 재양자화한 uncensored NVFP4 파일입니다.
ComfyUI 런타임은 컨테이너 내부의
`127.0.0.1:8188`에만 바인딩되며 외부에는 OpenAI 호환 API `8691`만 노출됩니다.
Gradio와 별도 웹 UI는 사용하지 않습니다.
이미지 전용 런타임이므로 NGC CUDA 13.3용 PyTorch를 유지하고 ComfyUI의 오디오용
TorchAudio 의존성은 설치하지 않습니다.

## Uncensored NVFP4 텍스트 인코더 생성

원본과 공식 템플릿은 Hugging Face 캐시에 보존하고, 변환 결과는 외부 Docker 볼륨
`media-hf-cache`의 `local/flux2-klein-4b-uncensored-nvfp4/`에 기록합니다.

```bash
docker compose build api
docker compose run --rm \
  -e PYTHONPATH=/opt/ComfyUI \
  -w /opt/ComfyUI \
  --entrypoint python \
  api /opt/nvfp4-api/quantize_uncensored_text_encoder.py
```

변환은 원본 가중치 값을 공식 체크포인트의 레이어별 BF16·FP8·NVFP4 배치로 다시
양자화합니다. 공식 가중치 값은 결과에 복사하지 않습니다. DGX Spark 실측 결과는
7.5GB 원본이 3.58GiB로 줄었고, 1024픽셀 생성 후 GPU 상주는 10,660MiB에서
6,660MiB로 감소했습니다.

실행 런타임은 이 uncensored NVFP4 파일만 사용합니다. 원본 FP16과 공식 FP4 파일은
재변환 입력으로만 캐시에 남으며 실행 선택지로 노출하지 않습니다.

제공 API:

- `GET /health`
- `GET /v1/models`
- `POST /v1/images/generations`
- `POST /v1/images/edits` (PNG/JPEG/WebP 참조 이미지 최대 4개)

```bash
docker volume create media-hf-cache
HF_TOKEN=hf_xxx docker compose build
HF_TOKEN=hf_xxx docker compose up -d
curl http://127.0.0.1:8691/health
```
