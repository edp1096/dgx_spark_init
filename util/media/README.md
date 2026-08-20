# Media client (working title)

이미지·LTX 영상·CustomVoice 음성 생성·자막 API를 사용하는 Go/Svelte 클라이언트입니다. 이 프로젝트에는
모델 런타임, Python 코드, Docker 제어 로직이 포함되지 않습니다.

API 서버는 별도 프로젝트로 운영합니다.

- 이미지 API: `compose_yaml/flux2_klein`
- 음성 API: `compose_yaml/qwen3_tts`
- 음성 인식 API: `compose_yaml/qwen3_asr`
- 영상 API: `compose_yaml/ltx-2.5_api`
- 프롬프트 향상 API: `compose_yaml/gemma4_litert`
- 미디어 접근 API: `compose_yaml/media_access_api`

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
  media:
    endpoint: http://127.0.0.1:8697
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

cd ../media_access_api
docker compose build
docker compose up -d
```

API 서버들은 모델 캐시 볼륨만 공유하며 서로 독립적으로 시작·종료할 수 있습니다.
Qwen3-TTS는 프리셋 화자를 사용하는 CustomVoice만 운영합니다.
Qwen3-ASR은 Gradio와 vLLM 없이 공식 `qwen-asr` Transformers wrapper 및
Forced Aligner로 운영합니다.
LTX 2.5 영상 API는 공식 distilled NVFP4 파이프라인을 사용하며 동시 생성은 한 작업으로 제한합니다.
Gemma 4 E2B LiteRT는 LTX 캡션 형식으로 한국어 원문을 번역·확장합니다. 현재 공개
`.litertlm` 번들은 이미지 입력을 인식하지 못하므로 I2V에서는 프롬프트 향상을
자동으로 건너뛰고 원문을 그대로 사용합니다. 비전 입력이 포함된 호환 번들을 사용할
때만 설정에서 `prompt_enhancement.vision_enabled`를 켭니다.

자막 탭은 로컬 영상·음성 파일을 디스크로 스트리밍 업로드하거나 URL에서 미디어를
가져옵니다. Media Access API가 yt-dlp, FFmpeg, Playwright를 담당하고 영상과 음성 입력은
기본적으로 `${HOME}/.local/share/media-access-api/media`에 영구 보관하여 웹에서 Range 스트리밍합니다.
Go 앱은 미디어 자산 ID와 종류만 기록하고 Media Access API의 스트림을 프록시합니다. Qwen3-ASR은
최대 180초 WAV 구간만 받습니다. Forced Aligner가 반환한 어절 시각에 각 구간의
원본 오프셋을 더해 실제 영상 시간축의 자막 큐를 만듭니다. 결과는 SRT, VTT,
타임코드 TXT, 일반 TXT 중에서 복수 선택할
수 있으며 Gemma 4 E2B 번역문 또는 원문·번역문 병기도 선택할 수 있습니다. 영상
영상 결과에는 플레이어용 VTT가 자동 생성되고, 음성 전용 결과는 오디오 플레이어로
재생할 수 있습니다. 작업 삭제 시 원본 미디어와 자막도 함께 삭제됩니다.
URL 작업은 영상 다운로드 바이트·퍼센트·ETA, MP4 저장, 음성 분리, ASR 구간,
번역 배치, 자막 파일 생성 단계를 각각 화면에 표시합니다.

영상 링크는 `파트·영상 출처 조회`로 사이트가 제공하는 선택지를 먼저 불러올 수
있습니다. SupJav의 숫자 버튼은 파트로, 각 파트의 버튼은 영상 출처로 표시합니다.
TV/JPA/ST/DS 같은 약어뿐 아니라 `SERVER:` 뒤의 `Streamtape`, `Mixdrop`, `NinjaStream`
같은 임의의 풀네임도 동적으로 표시합니다. 출처의 `자동`은 StreamTape 계열을 우선하고 사용할 수
없으면 해당 파트의 다른 출처를 순서대로 시도합니다.

설정의 저장소 관리에는 Media Access API의 `prepare-*` 임시 폴더 개수와 용량이
표시됩니다. `찌꺼기 정리`는 실행 중인 작업 폴더를 제외한 중단 작업의 임시 파일을
즉시 삭제합니다. 시작 시 자동 정리는 설정한 보존 시간보다 오래된 임시 폴더만
대상으로 합니다. 자막 번역 기본값은 원문과 번역문이며, 번역 언어는 추천 목록에서
고르거나 Gemma에 지시할 언어 이름을 직접 입력할 수 있습니다.
