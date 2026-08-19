# Media client (working title)

이미지·LTX 영상·CustomVoice 음성 생성·음성 인식 API를 사용하는 Go/Svelte 클라이언트입니다. 이 프로젝트에는
모델 런타임, Python 코드, Docker 제어 로직이 포함되지 않습니다.

API 서버는 별도 프로젝트로 운영합니다.

- 이미지 API: `compose_yaml/flux2_klein`
- 음성 API: `compose_yaml/qwen3_tts`
- 음성 인식 API: `compose_yaml/qwen3_asr`
- 영상 API: `compose_yaml/ltx-2.5_api`
- 프롬프트 향상 API: `compose_yaml/gemma4_litert`

## 클라이언트 빌드

```bash
make build
make dist
```

처음 실행하면 실행 디렉터리에 `media.yaml`과 `data/`가 생성됩니다. 기본 주소는
`http://0.0.0.0:8686`입니다. `media.yaml`에는 독립 API 서버의 endpoint만 지정합니다.

```yaml
engines:
  image:
    endpoint: http://127.0.0.1:8691
  speech:
    endpoint: http://127.0.0.1:8692
  recognition:
    endpoint: http://127.0.0.1:8694
  video:
    endpoint: http://127.0.0.1:8695
  prompt:
    endpoint: http://127.0.0.1:8696
```

클라이언트를 종료해도 API 서버는 영향을 받지 않으며, API 서버가 꺼져 있을 때는
화면에 `offline`으로 표시됩니다.

화면의 `설정` 탭에서 각 API endpoint와 모델·해상도·언어·업로드 제한 등의 기본값을
수정할 수 있습니다. 저장한 연결 정보와 기본값은 즉시 적용되고 `media.yaml`에도
기록됩니다. `listen`과 `data_dir` 변경만 Media 프로세스를 다시 시작해야 적용됩니다.

## API 서버 실행

```bash
docker volume create media-hf-cache

cd ../../compose_yaml/flux2_klein
docker compose up -d

cd ../qwen3_tts
docker compose up -d custom

cd ../qwen3_asr
docker compose up -d

cd ../ltx-2.5_api
docker compose up -d

cd ../gemma4_litert
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
uvx --from litert-lm litert-lm serve \
  --config ./config.json --host 0.0.0.0 --port 8696
```

API 서버들은 모델 캐시 볼륨만 공유하며 서로 독립적으로 시작·종료할 수 있습니다.
Qwen3-TTS는 프리셋 화자를 사용하는 CustomVoice만 운영합니다.
Qwen3-ASR은 Gradio와 vLLM 없이 Transformers native API로 운영합니다.
LTX 2.5 영상 API는 공식 distilled NVFP4 파이프라인을 사용하며 동시 생성은 한 작업으로 제한합니다.
Gemma 4 E2B LiteRT는 LTX 캡션 형식으로 한국어 원문을 번역·확장합니다. 현재 공개
`.litertlm` 번들은 이미지 입력을 인식하지 못하므로 I2V에서는 프롬프트 향상을
자동으로 건너뛰고 원문을 그대로 사용합니다. 비전 입력이 포함된 호환 번들을 사용할
때만 설정에서 `prompt_enhancement.vision_enabled`를 켭니다.
