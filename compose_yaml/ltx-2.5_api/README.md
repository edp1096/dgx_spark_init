# LTX 2.5 NVFP4 API

공식 Lightricks LTX-2.5 distilled NVFP4 파이프라인을 사용하는 헤드리스 영상 API입니다.
Gradio와 ComfyUI UI를 포함하지 않습니다.

```bash
docker volume create media-hf-cache
HF_TOKEN=... docker compose build
HF_TOKEN=... docker compose up -d
curl http://127.0.0.1:8695/health
```

모델은 첫 컨테이너 기동 때 `media-ltx-models` 볼륨에 내려받습니다. 파이프라인은 첫 생성
요청 때 메모리에 적재되며 이후 요청에서 재사용됩니다. 생성 동시성은 한 작업입니다.

```bash
curl -f -X POST http://127.0.0.1:8695/v1/videos/generations \
  -F 'prompt=A cinematic sunrise over a quiet mountain lake' \
  -F width=768 -F height=512 -F num_frames=121 -F fps=24 -F seed=42 \
  -o output.mp4
```

이미지-영상 생성은 같은 요청에 `-F image=@input.png -F image_strength=1.0`을 추가합니다.
