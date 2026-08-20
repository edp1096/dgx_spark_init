# SparkTalk

Go 백엔드와 Svelte 프런트엔드로 만든 간단한 로컬 LLM 채팅 앱입니다. OpenAI 호환 API(SGLang·vLLM 등)에 연결하며 채팅 기록은 SQLite에 저장합니다.

## 개발 실행

```bash
# 터미널 1
go run ./cmd/chat

# 터미널 2 (Node.js 20.19+ 필요)
cd web
npm install
npm run dev
```

브라우저에서 `http://127.0.0.1:5173`에 접속합니다.

## 배포 빌드

```bash
make build   # Svelte + Go 컴파일 확인
make dist    # Linux/Windows 배포 바이너리 생성
```

`dist/`에는 Linux와 Windows의 amd64·arm64 바이너리가 생성됩니다.
DGX Spark에서는 `sparktalk-linux-arm64`, 일반적인 Windows PC에서는
`sparktalk-windows-amd64.exe`를 실행합니다. `make clean`은 생성된
바이너리와 프런트엔드 빌드 파일을 정리합니다. Windows PowerShell에서는
GNUWin32의 GNU Make를 사용할 수 있습니다.

Svelte UI가 Go 바이너리에 포함되므로 배포할 때 `dist/`의 운영체제별
바이너리 하나만 복사하면 됩니다.

## 필요한 API와 Docker 기동

SparkTalk 자체는 Go 바이너리이며 Docker 컨테이너가 아니다. 전체 기능을 사용하려면
다음 세 API를 별도로 실행한다.

| 역할 | 위치 | 기본 주소 | 필요 여부 |
|---|---|---|---|
| 채팅·이미지·영상 인식 모델 | `compose_yaml/sglang_qwen38` | `http://127.0.0.1:8000` | 필수 |
| 음성 전사 | `compose_yaml/qwen3_asr` | `http://127.0.0.1:8694` | 음성·영상 첨부 시 필요 |
| FFmpeg·yt-dlp 변환/취득 | `compose_yaml/sparktalk_media_api` | `http://127.0.0.1:8698` | 음성·영상·URL 첨부 시 필요 |

이미지만 인식하는 채팅에는 SGLang만 있으면 된다. 음성·영상 또는 URL 미디어를
사용한다면 Qwen3-ASR과 SparkTalk Media API도 함께 실행한다. 생성 스튜디오용
`media_access_api`는 SparkTalk의 필수 구성요소가 아니다.

저장소 루트에서 다음 순서로 실행한다. 현재 DGX Spark에서 검증한 채팅 모델은
Huihui-RadixArk Qwen3.8 27B NVFP4 로컬 모델과 DFlash2 구성이다.

```bash
# 1. 채팅/VL 모델. Open WebUI는 올리지 않고 SGLang만 실행한다.
cd compose_yaml/sglang_qwen38
docker compose \
  -f compose.yaml \
  -f compose.huihui-radixark-nvfp4-dflash2-local.yaml \
  up -d --build sglang

# 2. 음성 인식. 외부 모델 캐시 볼륨은 최초 한 번만 만들면 된다.
cd ../qwen3_asr
docker volume create media-hf-cache
docker compose build
docker compose up -d

# 3. FFmpeg·yt-dlp 미디어 API
cd ../sparktalk_media_api
make build
make up

# 4. SparkTalk 앱
cd ../../util/chat
make dist
cd dist
./sparktalk-linux-arm64
```

`compose.huihui-radixark-nvfp4-dflash2-local.yaml`은 해당 로컬 target 모델 경로가
준비된 DGX Spark용이다. 최초 실행에는 DFlash2 다운로드와 전용 SGLang 이미지 빌드,
CUDA graph 컴파일이 필요하다. 다른 OpenAI 호환 SGLang·vLLM 서버를 사용할 때는
컨테이너 1번 대신 그 서버를 실행하고 `sparktalk.yaml`의 `api.endpoint`와 모델명을
맞춘다. 비디오 인식에는 VL 입력을 지원하는 모델이 필요하다.

yt-dlp로 지원 사이트를 가져오지 못할 때는 고정된 이미지 전체를 다시 빌드하기
전에 선택 업데이트를 적용할 수 있다.

```bash
cd compose_yaml/sparktalk_media_api
make ytdlp-update
make ytdlp-version
```

각 서비스와 SparkTalk 연결 상태는 다음과 같이 확인한다.

```bash
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8694/health
curl -fsS http://127.0.0.1:8698/health
curl -fsS http://127.0.0.1:8585/api/health
```

SGLang은 재부팅 후 첫 기동에서 `torch.compile` 때문에 준비에 시간이 걸릴 수 있다.
`docker logs -f sglang-qwen38`에서 준비 상태를 확인한다. 종료할 때는 SparkTalk에
`Ctrl+C`를 입력하고 각 compose 디렉터리에서 `docker compose down` 또는
SparkTalk Media API의 `make down`을 실행한다. SGLang은 기동할 때 사용한 두
compose 파일을 동일하게 지정해서 내린다.

## 설정

실행 옵션은 사용하지 않습니다. 처음 실행하면 현재 디렉터리에 내장된
샘플을 바탕으로 `sparktalk.yaml`을 만들고 곧바로 서버를 시작합니다.

```bash
./dist/sparktalk-linux-arm64
```

API endpoint, listen address, SQLite 경로, 기본 모델과 reasoning effort는
`sparktalk.yaml` 또는 웹 화면 좌측 하단의 **설정**에서 관리합니다.
Endpoint·모델·reasoning 변경은 저장 즉시 적용되며 listen address와 DB
경로 변경은 앱 재시작 후 적용됩니다.

## 지능형 문맥 관리

브라우저와 SQLite에는 대화 원문을 그대로 보존하되, 모델에 보낼 때만 오래된
구간을 구조화된 체크포인트로 요약합니다. 모델 API의 문맥 길이를 자동으로
조회하고 출력 예약분과 안전 여유분을 제외한 입력 예산의 80%에 도달하면,
최근 대화는 원문으로 유지하고 이전 구간만 요약으로 교체합니다. 이미지와
파일은 다시 전송하지 않고 이름·형식·크기와 당시 대화에 남은 설명으로
대표합니다.

대화 화면 오른쪽의 문맥 표시를 누르면 현재 사용량, 원문 구간과 요약된
체크포인트를 확인하고 원문 위치로 이동할 수 있습니다. **지금 구간 정리**는
한도 도달 전에도 수동 체크포인트를 만들며, **요약 초기화**는 체크포인트만
삭제하고 대화 원문은 건드리지 않습니다. 질의를 수정하거나 이전 답변에서
다시 생성하면 해당 분기 이후의 체크포인트도 자동으로 무효화됩니다.

설정 화면에서 자동 관리 여부, 모델이 보고하지 않을 때 사용할 문맥 길이,
정리 시점, 출력 예약 토큰, 안전 여유분, 원문으로 유지할 최근 토큰과 이미지
예상 토큰을 조절할 수 있습니다. 기본값은 각각 자동 감지, 80%, 8192, 4096,
32768, 2048입니다. 모델이 문맥 초과를 반환하면 한 번 더 보수적으로 정리한
뒤 자동 재시도하고, 그래도 실패하면 기존 대화는 그대로 둔 채 오류를
표시합니다.

모델 요청은 `생성 중`, `완료`, `실패`, `사용자 중지` 상태로 관리합니다.
실패하거나 중지한 요청도 화면과 SQLite에는 이력으로 남지만, 다음 모델
요청에는 첨부 원본과 긴 오류 전문을 다시 보내지 않습니다. 대신 요청 내용,
첨부 파일명·형식·크기와 짧게 정리한 실패 원인만 텍스트 기록으로 전달하므로
“왜 실패했나” 같은 후속 질문은 가능하면서 손상된 미디어가 이후 대화를
계속 막는 일은 방지합니다. 사용자 요청과 정상 답변의 완료 처리는 하나의
DB 트랜잭션으로 반영됩니다.

## 웹 도구

별도 검색 서버 없이 내장 `web_search`와 `web_fetch`를 사용할 수 있습니다.
채팅 상단의 **웹 자동**을 켜면 모델이 필요한 경우 DuckDuckGo 검색 또는
공개 웹 페이지 읽기를 선택해서 실행합니다. 실행 기록은 답변의 **웹 도구**
영역에서 접고 펼칠 수 있습니다. 검색 결과는 참고 자료로만 모델에 전달되며,
로컬·사설망 주소는 차단됩니다. 사용 여부, 최대 호출 횟수, 검색 결과 수와
타임아웃은 설정 화면 또는 YAML의 `tools` 항목에서 조정합니다.

## 이미지 인식

입력창의 `＋` 버튼, 드래그 앤 드롭 또는 클립보드 붙여넣기로 PNG·JPEG·WebP
이미지를 한 메시지에 최대 6개까지 첨부할 수 있습니다. 이미지당 최대 크기는
15MB이며, 비전 입력을 지원하는 OpenAI 호환 모델이 필요합니다. 첨부 이미지는
SQLite 파일 옆의 `<database>.media` 디렉터리에 저장되고 질문 수정·답변 재시도·
버전 전환에서도 함께 보존됩니다.

## 음성·영상 인식

음성 파일은 원본을 대화 모델에 직접 보내지 않고 `sparktalk_media_api`에서 16kHz
mono PCM WAV로 변환한 뒤 Qwen3-ASR의 전사문을 전달합니다. 영상 파일은
Qwen VL이 볼 수 있는 영상 원본과 음성 트랙의 전사문을 함께 전달합니다.
음성 트랙이 없는 영상은 영상만 전달합니다.

기본 서비스 주소는 SparkTalk Media API `http://127.0.0.1:8698`, Qwen3-ASR
`http://127.0.0.1:8694`입니다. 설정 화면의 **음성 인식**에서 활성화 여부,
두 endpoint, ASR 모델, 인식 언어, 문맥·전문용어 힌트와 타임아웃을 관리할
수 있습니다. 서비스 상태도 같은 영역에 표시됩니다.

전사 결과는 `<database>.media`에 첨부 ID별 작은 캐시로 저장됩니다. 답변
재시도와 이후 대화에서 같은 파일을 다시 인식하지 않으며, 언어·힌트·모델
설정이 바뀌면 자동으로 다시 인식합니다. 미사용 미디어를 정리하면 해당 전사
캐시도 함께 삭제됩니다.

입력창의 `⌁` 버튼에 YouTube·Vimeo·Dailymotion 등 yt-dlp가 인식하는 URL을
넣으면 SparkTalk 백엔드가 SparkTalk Media API를 통해 단일 미디어를 취득하고 일반
파일 첨부와 동일하게 대화방에 보관합니다. 브라우저에서 바로 재생할 수 있으며,
음성·영상 분석도 기존 ASR/VL 경로를 그대로 사용합니다. URL 첨부도 파일당 64MB,
메시지당 총 96MB 제한을 적용합니다.

기본 listen address는 `0.0.0.0:8585`입니다. 같은 네트워크의 다른 PC에서는
`http://서버-IP:8585`로 접속합니다.
