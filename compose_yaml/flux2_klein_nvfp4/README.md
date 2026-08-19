# FLUX.2 Klein 4B NVFP4 API

공식 `black-forest-labs/FLUX.2-klein-4b-nvfp4` transformer와 uncensored Qwen
텍스트 인코더를 사용하는 이미지 생성 엔진입니다. ComfyUI 런타임은 컨테이너 내부의
`127.0.0.1:8188`에만 바인딩되며 외부에는 OpenAI 호환 API `8691`만 노출됩니다.
Gradio와 별도 웹 UI는 사용하지 않습니다.
이미지 전용 런타임이므로 NGC CUDA 13.3용 PyTorch를 유지하고 ComfyUI의 오디오용
TorchAudio 의존성은 설치하지 않습니다.

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
