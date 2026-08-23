# Z-Image Turbo NVFP4 API

Standalone, headless ComfyUI service exposing an OpenAI-compatible image API.

```bash
docker compose up -d --build
curl http://127.0.0.1:8703/health
```

The default stack uses the official `Comfy-Org/z_image_turbo` NVFP4 diffusion
model, FP4 mixed Qwen3-4B text encoder, AE, and Alibaba PAI's full
`Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps` patch. Supplying a base64
`control_image` activates built-in Canny preprocessing. `control_strategy` can
be `full8` or `split4` (the default: four controlled steps followed by four
base steps). Models are unloaded after each request by default so the complete
media studio can remain online; set `ZIMAGE_UNLOAD_AFTER_GENERATION=false` to
retain the warm cache for repeated dedicated Z-Image sessions.
