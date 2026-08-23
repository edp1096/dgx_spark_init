# Spark Media

이미지·LTX 영상·CustomVoice 음성 생성·자막 API를 사용하는 Go/Svelte 클라이언트입니다. 이 프로젝트에는
모델 런타임, Python 코드, Docker 제어 로직이 포함되지 않습니다.

API 서버는 별도 프로젝트로 운영합니다.

| 화면 기능 | API 위치 | 기본 주소 | 필요 여부 |
|---|---|---|---|
| 고품질 이미지 생성 | `compose_yaml/krea2_turbo_nvfp4` | `http://127.0.0.1:8691` | 이미지 생성 모드 |
| vLLM-Omni 공통 런타임 이미지 | `compose_yaml/vllm_omni` | 포트 없음 | Qwen3-TTS 이미지 최초 빌드 시 |
| 음성 생성 | `compose_yaml/qwen3_tts` | `http://127.0.0.1:8692` | 음성 탭 사용 시 |
| 음성 인식 | `compose_yaml/qwen3_asr` | `http://127.0.0.1:8694` | 자막 탭 사용 시 |
| 영상 생성 | `compose_yaml/ltx-2.5_api` | `http://127.0.0.1:8695` | 영상 탭 사용 시 |
| 프롬프트 향상·번역 | `compose_yaml/llama.cpp` | `http://127.0.0.1:8696` | 향상·자막 번역 사용 시 |
| 미디어 다운로드·분할 | `compose_yaml/media_access_api` | `http://127.0.0.1:8697` | URL·자막 처리 시 |
| 이미지 고화질화 | `compose_yaml/seedvr2_upscaler` | `http://127.0.0.1:8698` | 최근 이미지 후처리 시 |
| Krea 2 LoRA 학습 | `compose_yaml/krea2_lora_trainer` | `http://127.0.0.1:8704` | LoRA 제작소 사용 시 |

Media 앱 자체는 Go 바이너리이며 Docker 컨테이너가 아니다. 모든 API는 서로
독립적이므로 사용할 탭에 필요한 것만 올릴 수 있다. 자막 탭에서 로컬 음성만
전사하려면 Qwen3-ASR과 Media Access API가 필요하고, 번역까지 사용하려면 Gemma 4
E2B llama.cpp 서비스도 실행한다. 이미지·음성·영상 생성 API를 모두 동시에 띄울지는
DGX Spark 통합 메모리 사용량을 보고 결정한다. vLLM-Omni 항목은 상시 실행하는 API가
아니라 Qwen3-TTS가 사용하는 `dgx-vllm-omni:v0.26.0` 런타임 이미지를 만드는 빌더다.

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
  upscale:
    endpoint: http://127.0.0.1:8698
  trainer:
    endpoint: http://127.0.0.1:8704

image:
  default_mode: create
  backends:
    create:
      endpoint: http://127.0.0.1:8691
      model: krea2-turbo-nvfp4
```

클라이언트를 종료해도 API 서버는 영향을 받지 않습니다. 헤더에는 필요한 API 전체의
상태가 하나의 신호등으로 표시되며, 이를 누르면 API별 `online`·`offline` 상태를
확인할 수 있습니다.

이미지 탭의 `Krea 2 Turbo`는 ComfyUI 그래프를 노출하지 않고 기능 모듈을 켜고 끄는
제어판으로 구성됩니다. `인물·장면 유지`는 Krea Identity Edit, `자세·구도`는 Krea
Depth Control, `스타일 LoRA`는 선택한 Krea LoRA들을 개별 강도로 순차 중첩하며 세
모듈은 동시에 사용할 수 있습니다. 선택한 LoRA와 강도는 작업 기록에 저장되어 설정
복제로 그대로 복원됩니다. 두 장 Identity 편집에서는 주 참조에 장면, 추가 참조에 삽입할 인물을
넣습니다. `실험 편집·윤곽`은 NK2E v0.3의 국소 편집 또는 Canny 구조 참조를 사용하며,
현재는 다른 Krea 모듈과 조합하지 않습니다. `부분 수정·확장`은 Krea2 AnyPaint로 흰색
마스크 영역을 다시 그리거나 원본의 상·하·좌·우 캔버스를 확장합니다. 마스크가 없으면
아웃페인트만 수행하며, 프롬프트에는 변경 지시보다 완성될 전체 장면을 적는 편이
안정적입니다. 원본 위에서 브러시·지우개로 직접 수정 영역을 칠하거나, 반투명 인물·얼굴·
가로·사각 프리셋을 이동하고 크기를 조절해 마스크로 만들 수 있습니다. 외부에서 만든
흑백 마스크 파일도 계속 사용할 수 있습니다. 원본과 마스크는 작업 기록에 보관되어
`참조` 또는 `전체 작업` 복제로 재사용할 수 있습니다. 모든 기능 모듈의 입력 썸네일과
최근 생성 결과는 클릭하면 공통 확대 모달로 열립니다. 과거 Klein 작업 기록을 전체 복제하면 참조 이미지를
Krea Identity 입력으로 변환합니다.

이미지 작업이 대기·실행 중이면 최근 이미지 카드에 경과 시간, 예상 진행률, 남은 시간과
예상 완료 시각을 표시합니다. ETA는 동일한 모드·해상도·스텝의 최근 완료 시간 중앙값을
우선 사용하며, 실제 ComfyUI 스텝 이벤트가 아닌 추정치임을 화면에 명시합니다.

Identity에는 얼굴·특징에 주의를 집중하는 마스크와 변경을 허용할 영역만 지정하는
수정 한정 마스크를 별도로 사용할 수 있습니다. `내용·구도 참조`는 Qwen3-VL
conditioning으로 최대 4장의 내용을 참고하고, `스타일 이미지 참조`는 최대 2장에서
화풍·색감·질감을 가져옵니다. 전용 모델을 사용하는 기능은 화면에서 호환되지 않는
다른 모듈과의 동시 선택을 제한합니다. 모델 내부 조정에서는 프롬프트 준수 강화와
2-vector·3-vector 필터 완화 가중치를 선택하고 강도를 조절할 수 있습니다. 별도 `생성
설정`에서는 시드·스텝과 기본 `Euler/Simple` 또는 디테일 탐색용 `ER-SDE/Simple`
프리셋을 고릅니다. 실제 무작위 시드·샘플러·스케줄러·스텝은 작업 기록과 이미지 EXIF에
함께 저장됩니다.

이미지 프롬프트 향상을 켜면 첫 생성 클릭에서 Gemma 4 E2B가 Krea 2용 영어 프롬프트를
제안하고, 사용자가 내용을 확인한 다음 다시 생성할 수 있습니다. JSON처럼 구조화된 프롬프트는
임의로 변경하지 않습니다. 이미지 크기는 화면 비율과 0.75/1/2MP 등급만 고르는 간편 모드가
기본이며, 필요한 경우 폭과 높이를 직접 지정할 수 있습니다. 최근 이미지의 `고화질로 만들기`는
별도 SeedVR2 3B FP8 서비스로 2배 복원한 결과를 새 이미지 작업으로 보존합니다.

프롬프트 예제 모달에는 Krea 공식 예제를 우선 표시하고 각 예제의 출처를 함께
제공합니다. 프롬프트 조립기에서는 주제·외형·포즈·카메라·환경·조명·재질·스타일·
글자·유지 조건을 조합할 수 있으며, 포즈 120개와 메이크업·얼굴 연출 프리셋을
미리보기로 선택할 수 있습니다. 조립 결과는 모달 하단에 고정되며 현재 프롬프트를
교체하거나 뒤에 이어 붙일 수 있습니다.

`LoRA` 탭에서는 학습 데이터셋을 만들고 이미지를 여러 장 올린 뒤, 이미지별 캡션과
트리거 단어를 정리해 Krea 2 Turbo LoRA 학습을 시작할 수 있습니다. 학습 엔진은 Ostris
ai-toolkit CLI를 사용하지만 데이터셋·설정·진행률·중지는 Media 제어판에서 관리합니다.
완성된 LoRA는 공유 볼륨에 자동 등록되며 이미지 탭의 `사용자 LoRA` 모듈에서 최대 5개를
각기 다른 강도로 중첩할 수 있습니다.

이미지·영상·음성·자막의 최근 결과와 전체 생성 기록은 페이지 단위로 표시됩니다.
탭마다 표시 개수와 최신순·오래된순 정렬을 독립적으로 저장하며, 상단과 하단에서
페이지를 이동할 수 있습니다. 이미지·영상·자막은 갤러리와 목록 보기를 지원하고
결과를 모달에서 크게 열 수 있습니다. 이미지 모달에서는 원본 파일과 생성 설정이
담긴 EXIF 정보를 확인할 수 있습니다.

화면의 `설정` 탭은 `연결`, `기본값`, `이미지 정보`, `저장소`로 구분됩니다. 각 API
endpoint와 모델·해상도·언어·업로드 제한 등의 기본값을 수정하고, 새 이미지 EXIF에
기록할 제작자·저작권·설명을 관리할 수 있습니다. 저장한 연결 정보와 기본값은 즉시
적용되고 `media.yaml`에도 기록됩니다. `listen`과 `data_dir` 변경만 Media 프로세스를
다시 시작해야 적용됩니다.

## API 서버 실행

아래 명령은 모든 탭을 사용할 때의 전체 구성이다. 일부 탭만 사용한다면 위 표에
해당하는 서비스만 실행하면 된다.

```bash
docker volume create media-hf-cache
docker volume create media-krea-user-loras

cd ../../compose_yaml/krea2_turbo_nvfp4
docker compose up -d

cd ../krea2_lora_trainer
docker compose build
docker compose up -d

# Qwen3-TTS가 사용할 vLLM-Omni 런타임 이미지를 최초 한 번 빌드한다.
cd ../vllm_omni
docker compose --profile check build

cd ../qwen3_tts
docker compose up -d custom

cd ../qwen3_asr
docker compose up -d

cd ../ltx-2.5_api
docker compose up -d

cd ../seedvr2_upscaler
docker compose build
docker compose up -d

cd ../llama.cpp
./scripts/build_host.sh
./scripts/install_user_service.sh

cd ../../compose_yaml/media_access_api
docker compose build
docker compose up -d
```

API 서버들은 모델 캐시 볼륨만 공유하며 서로 독립적으로 시작·종료할 수 있습니다.
Qwen3-TTS는 vLLM-Omni 런타임에서 프리셋 화자를 사용하는 CustomVoice만 운영합니다.
Qwen3-ASR은 Gradio와 vLLM 없이 공식 `qwen-asr` Transformers wrapper 및
Forced Aligner로 운영합니다.
LTX 2.5 영상 API는 공식 distilled NVFP4 파이프라인을 사용하며 동시 생성은 한 작업으로 제한합니다.
Gemma 4 E2B QAT Q4_K + MTP llama.cpp는 LTX 캡션 형식으로 한국어 원문을
번역·확장합니다. I2V 시작 이미지는 LTX에 직접 전달하며, Media의 이미지 인식 기반
프롬프트 향상은 기본적으로 끄고 원문 동작 지시를 사용합니다.

자막 탭은 로컬 영상·음성 파일을 디스크로 스트리밍 업로드하거나 URL에서 미디어를
가져옵니다. Media Access API가 yt-dlp, FFmpeg, Playwright를 담당하고 영상과 음성 입력은
기본적으로 `${HOME}/.local/share/media-access-api/media`에 영구 보관하여 웹에서 Range 스트리밍합니다.
Go 앱은 미디어 자산 ID와 종류만 기록하고 Media Access API의 스트림을 프록시합니다. Qwen3-ASR은
최대 180초 WAV 구간만 받습니다. Forced Aligner가 반환한 어절 시각에 각 구간의
원본 오프셋을 더해 실제 영상 시간축의 자막 큐를 만듭니다. 결과는 SRT, VTT,
타임코드 TXT, 일반 TXT 중에서 복수 선택할
수 있으며 Gemma 4 E2B 번역문 또는 원문·번역문 병기도 선택할 수 있습니다. 영상
결과에는 플레이어용 VTT가 자동 생성되고, 음성 전용 결과는 오디오 플레이어로
재생할 수 있습니다. 작업 삭제 시 원본 미디어와 자막도 함께 삭제됩니다.
URL 작업은 영상 다운로드 바이트·퍼센트·ETA, MP4 저장, 음성 분리, ASR 구간,
번역 배치, 자막 파일 생성 단계를 각각 화면에 표시합니다. 진행 중인 자막 카드의
`중지`를 누르면 해당 작업만 `cancelled`로 전환하고 Media Access API에서 그 요청에
속한 yt-dlp 또는 FFmpeg 프로세스를 종료합니다. 중지된 작업은 삭제할 수 있으며,
남은 부분 파일은 설정의 `찌꺼기 정리` 대상입니다.

언어의 `Auto · 단일 언어`는 첫 구간에서 감지한 언어를 이후 구간에 고정하여 일반
영상의 인식 흔들림을 줄입니다. `Auto · 다중 언어`는 구간마다 자동 감지를 유지하고
감지된 언어를 모두 기록합니다. 한국어·일본어·영어가 섞인 노래처럼 실제 후렴이
반복되는 입력에서는 반복 문장을 ASR 환각으로 간주하지 않되, 잘못된 타임스탬프와
한 시점으로 뭉친 정렬 결과는 계속 거부합니다.

영상 링크는 `영상 내부 선택지 조회`로 사이트가 제공하는 선택지를 먼저 불러올 수
있습니다. SupJav의 숫자 버튼은 파트로, 각 파트의 버튼은 영상 출처로 표시합니다.
TV/JPA/ST/DS 같은 약어뿐 아니라 `SERVER:` 뒤의 `Streamtape`, `Mixdrop`, `NinjaStream`
같은 임의의 풀네임도 동적으로 표시합니다. 출처의 `자동`은 StreamTape 계열을 우선하고 사용할 수
없으면 해당 파트의 다른 출처를 순서대로 시도합니다.

설정의 저장소 관리에는 Media Access API의 `prepare-*` 임시 폴더 개수와 용량이
표시됩니다. `찌꺼기 정리`는 실행 중인 작업 폴더를 제외한 중단 작업의 임시 파일을
즉시 삭제합니다. 시작 시 자동 정리는 설정한 보존 시간보다 오래된 임시 폴더만
대상으로 합니다. 자막 번역 기본값은 원문과 번역문이며, 번역 언어는 추천 목록에서
고르거나 Gemma에 지시할 언어 이름을 직접 입력할 수 있습니다.
