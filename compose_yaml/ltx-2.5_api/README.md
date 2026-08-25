# LTX 2.5 NVFP4 API

공식 Lightricks LTX-2.5 distilled NVFP4 파이프라인을 사용하는 헤드리스 영상 API입니다.
Gradio와 ComfyUI UI를 포함하지 않습니다.

```bash
docker volume create media-hf-cache
docker compose build
docker compose up -d
curl http://127.0.0.1:8695/health
```

모델은 첫 컨테이너 기동 때 `media-ltx-models` 볼륨에 내려받습니다. 파이프라인은 첫 생성
요청 때 메모리에 적재되며 이후 요청에서 재사용됩니다. 생성 동시성은 한 작업입니다.

백지 설치에서도 API는 모델 다운로드보다 먼저 기동합니다. Spark Media의
`설정 → 연결 → LTX 영상 모델 준비`에서 [LTX-2.5 모델 페이지](https://huggingface.co/Lightricks/LTX-2.5)
라이선스에 동의한 계정의 read 토큰을 Spark Media의 `설정 → 연결 → 다운로드 인증`에 한 번
저장하고 `모델 준비 시작`을 누릅니다. 토큰은 브라우저로 다시 반환되지 않으며 공용 모델
캐시의 권한 `0600` 파일에 저장되어 LoRA 다운로드와 함께 사용됩니다.
공식 Transformer·Gemma·VAE·업스케일러와 공개 Motion LoRA가 자동 다운로드되므로
그 밖의 API 키, SSH 작업, Compose 재시작 또는 수동 다운로드는 필요하지 않습니다.

Spark Media를 쓰지 않는 헤드리스 설치만 `.env.example`을 `.env`로 복사해
`HF_TOKEN`을 입력하는 CLI 대체 경로를 사용합니다. `.env`는 Git에서 제외됩니다.

Motion LoRA는 기본적으로 다운로드만 하고 적용하지 않습니다. Spark Media
`설정 → 생성 기본값 → 영상`에서 켜짐/꺼짐과 강도를 저장하면 다음 영상부터 즉시
반영됩니다. 권장 범위는 `0.35~0.70`이며 기본 제안값은 `0.50`입니다. LoRA 상태를
바꾼 첫 작업에서만 API가 파이프라인을 자동 재적재하며 컨테이너 재시작은 없습니다.

헤드리스 API에서 모든 요청의 기본값으로 켤 때만 `.env`를 사용할 수 있습니다.

```dotenv
LTX_LORA_PATH=/models/ltx-2.5/loras/ltx-2.3-ltx2-better-nsfw-motion.safetensors
LTX_LORA_STRENGTH=0.5
```

```bash
curl -f -X POST http://127.0.0.1:8695/v1/videos/generations \
  -F 'prompt=A cinematic sunrise over a quiet mountain lake' \
  -F width=768 -F height=512 -F num_frames=121 -F fps=24 -F seed=42 \
  -o output.mp4
```

이미지-영상 생성은 같은 요청에 `-F image=@input.png -F image_strength=1.0`을 추가합니다.
기존 단일 이미지 필드는 시작 프레임 호환용으로 유지됩니다. 다중 장면 조건은 이미지를
`images` 필드로 순서대로 보내고 같은 순서의 프레임 번호와 강도를 JSON 배열로 지정합니다.

```bash
curl -f -X POST http://127.0.0.1:8695/v1/videos/generations \
  -F 'prompt=A continuous cinematic camera move through three connected scenes' \
  -F width=768 -F height=512 -F num_frames=121 -F fps=24 -F seed=42 \
  -F 'images=@start.png' -F 'images=@middle.png' -F 'images=@end.png' \
  -F 'frame_indices=[0,60,120]' -F 'image_strengths=[1.0,0.8,1.0]' \
  -o output.mp4
```

시작과 마지막을 포함해 최대 10장까지 받을 수 있으며 동일 프레임에 이미지를 중복 배치할
수 없습니다.

## DGX Spark 통합 메모리 실측

2026-08-24에 Krea 2(`--gpu-only`), Qwen3-TTS, Qwen3-ASR, SeedVR2,
Media Access API, LoRA trainer와 llama.cpp Gemma 4 E2B를 함께 실행한 상태에서
`768x512`, 121프레임, 24fps 영상을 생성했다. 생성 전 시스템 사용량은
46.57 GiB, 생성 중 피크는 73.99 GiB였고 최소 가용 메모리는 47.64 GiB였다.
전체 CUDA 할당 피크는 58.07 GiB, LTX 프로세스의 CUDA 할당 피크는
23.30 GiB였으며 영상은 정상적으로 완료됐다.
