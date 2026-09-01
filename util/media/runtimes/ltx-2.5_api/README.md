# LTX 2.5 NVFP4 API

일반 생성은 공식 Lightricks LTX-2.5 distilled NVFP4, A2V는 공식 dev 체크포인트 FP8 cast와 distilled LoRA를 사용하는 헤드리스 영상 API입니다.
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
공식 Transformer·Gemma·VAE·업스케일러·A2V dev/LoRA와 공개 Motion LoRA가 자동 다운로드되므로
그 밖의 API 키, SSH 작업, Compose 재시작 또는 수동 다운로드는 필요하지 않습니다.

A2V 추가 자산은 약 51GB를 사용합니다. 음성 탭의 `A2V`로 보낸 음성은 영상 탭에 서버 내부
참조로 연결되며, 생성 요청 때 원본 오디오를 조건으로 사용하는 공식 `A2VidPipelineTwoStage`가 실행됩니다.

Spark Media를 쓰지 않는 헤드리스 설치만 `.env.example`을 `.env`로 복사해
`HF_TOKEN`을 입력하는 CLI 대체 경로를 사용합니다. `.env`는 Git에서 제외됩니다.

Motion LoRA는 기본적으로 다운로드만 하고 적용하지 않습니다. Spark Media
`설정 → 생성 기본값 → 영상`에서 켜짐/꺼짐과 강도를 저장하면 다음 영상부터 즉시
반영됩니다. 권장 범위는 `0.35~0.70`이며 기본 제안값은 `0.50`입니다. LoRA 상태를
바꾼 첫 작업에서만 API가 파이프라인을 자동 재적재하며 컨테이너 재시작은 없습니다.

## DGX Spark SM121 가속

이미지는 [Sana SM121 Sol-Attn 포크](https://github.com/edp1096/Sana/tree/feat/sol-attn-sm121)의
검증 커밋 `809638b437f49bdda969ebf568d12b8e91806c98`을 고정해 빌드합니다. 기본 `auto` 정책은
Stage 2 토큰이 32,000개 미만이면 기존 Dense attention을 유지하고, 그 이상이면 GB10용
CuTe DSL Sol-Attn을 사용합니다. LoRA가 없을 때는 Exact AdaLN도 함께 사용하고, Motion
LoRA가 켜지면 학습된 AdaLN을 보존하기 위해 Sol-Attn만 적용합니다. 초기화에 실패하면
Dense attention으로 자동 복귀합니다.

Spark Media의 `설정 → 생성 기본값 → 영상 → 고해상도 가속`에서 `자동` 또는 `끄기`를
선택할 수 있습니다. 헤드리스 요청에서는 `acceleration=auto|dense|sol`을 전달하며 실제
사용 경로는 응답 헤더 `X-LTX-Acceleration`과 `/health`의 `acceleration.last`에 기록됩니다.

```dotenv
LTX_SOL_MODE=auto
LTX_SOL_MIN_TOKENS=32000
LTX_SOL_EXACT_ADALN=1
```

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

2026-08-26 운영 이미지 후보에서 `1920x1088`, 121프레임을 실제 생성한 결과
`cute_sm121+exact-adaln` 경로로 136.8초에 완료됐다. Stage 2의 Sol 커널 호출은
141회였고 전체 통합 메모리 사용량은 다른 Spark Media 서비스가 실행 중인 상태에서
약 62 GiB였다. 시작 이미지와 Motion LoRA 0.5를 함께 적용한 동일 해상도 작업도
`cute_sm121` 경로로 149.3초에 완료됐다.

같은 날 공식 A2V dev FP8 경로를 `256x256`, 9프레임, 24fps로 실측했다. 첫 로드부터
완료까지 약 2분 15초, 전체 통합 메모리 관측 피크는 72.7 GiB였다. 결과에는 H.264 영상과
24kHz 스테레오 AAC가 함께 저장됐고, 완료 뒤 일반 distilled NVFP4 파이프라인으로 자동
전환한 9프레임 생성도 정상 완료됐다.
