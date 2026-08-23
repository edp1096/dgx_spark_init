# SeedVR2 image upscaler

SeedVR2 3B FP8을 독립 ComfyUI 런타임으로 실행하고 `POST /v1/images/upscale` API를 제공합니다.
생성 스튜디오에서는 모델명을 노출하지 않고 최근 이미지의 `고화질로 만들기`로 호출합니다.

```bash
docker volume create media-hf-cache
docker compose build
docker compose up -d
curl http://127.0.0.1:8698/health
```

첫 실행에서는 `seedvr2_ema_3b_fp8_e4m3fn.safetensors`와 `ema_vae_fp16.safetensors`를
`media-seedvr2-models` 볼륨에 자동 다운로드하므로 시간이 걸립니다. 출력은 원본 비율을
유지하는 2배 확대이며 어느 한 변도 4096px를 넘지 않도록 제한합니다.
