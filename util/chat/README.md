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
다음 API를 별도로 실행한다.

| 역할 | 위치 | 기본 주소 | 필요 여부 |
|---|---|---|---|
| 채팅·이미지·영상 인식 모델 | `compose_yaml/sglang_qwen38` | `http://127.0.0.1:8000` | 필수 |
| 음성 전사 | `compose_yaml/qwen3_asr` | `http://127.0.0.1:8694` | 음성·영상 첨부 시 필요 |
| 답변 음성 | `compose_yaml/qwen3_tts` | `http://127.0.0.1:8692` | AI 답변 읽기 사용 시 필요 |
| SparkTalk Extra Media | `compose_yaml/sparktalk_extra` | `http://127.0.0.1:8698` | 음성·영상·URL 첨부 시 필요 |
| SparkTalk Extra SSH | `compose_yaml/sparktalk_extra` | `http://127.0.0.1:8699` | 채팅에서 승인형 SSH 실행 시 필요 |

이미지만 인식하는 채팅에는 SGLang만 있으면 된다. 음성·영상 또는 URL 미디어를
사용한다면 Qwen3-ASR과 SparkTalk Extra Media도 함께 실행한다. 답변 읽기를
사용한다면 Qwen3-TTS도 실행한다. 생성 스튜디오용
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

# 3. 미디어와 SSH 부가기능
cd ../sparktalk_extra
make build
make up

# 4. AI 답변 읽기(선택). 공통 vLLM-Omni 이미지는 최초 한 번 빌드한다.
cd ../vllm_omni
docker compose build
cd ../qwen3_tts
docker compose up -d custom

# 5. SparkTalk 앱
cd ../../util/chat
make dist
cd dist
./sparktalk-linux-arm64
```

`compose.huihui-radixark-nvfp4-dflash2-local.yaml`은 해당 로컬 target 모델 경로가
준비된 DGX Spark용이다. 최초 실행에는 DFlash2 다운로드와 전용 SGLang 이미지 빌드,
CUDA graph 컴파일이 필요하다. 다른 OpenAI 호환 SGLang·vLLM 서버를 사용할 때는
컨테이너 1번 대신 그 서버를 실행하고 `sparktalk.yaml`의 `model.endpoint`와 모델명을
맞춘다. 비디오 인식에는 VL 입력을 지원하는 모델이 필요하다.

yt-dlp로 지원 사이트를 가져오지 못할 때는 고정된 이미지 전체를 다시 빌드하기
전에 선택 업데이트를 적용할 수 있다.

```bash
cd compose_yaml/sparktalk_extra
make ytdlp-update
make ytdlp-version
```

각 서비스와 SparkTalk 연결 상태는 다음과 같이 확인한다.

```bash
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8694/health
curl -fsS http://127.0.0.1:8692/health
curl -fsS http://127.0.0.1:8698/health
curl -fsS http://127.0.0.1:8699/health
curl -fsS http://127.0.0.1:8585/api/health
```

SGLang은 재부팅 후 첫 기동에서 `torch.compile` 때문에 준비에 시간이 걸릴 수 있다.
`docker logs -f sglang-qwen38`에서 준비 상태를 확인한다. 직접 실행한 SparkTalk은
`Ctrl+C`로 종료하고, 아래 사용자 systemd 서비스를 설치했다면 `systemctl --user
stop sparktalk`을 사용한다. 각 compose 서비스는 해당 디렉터리에서
`docker compose down` 또는 SparkTalk Extra의 `make down`으로 내린다. SGLang은
기동할 때 사용한 두 compose 파일을 동일하게 지정해서 내린다.

## DGX Spark 자동기동

터미널이나 SSH 연결이 종료돼도 SparkTalk을 유지하려면 저장소의 사용자 systemd
서비스를 설치한다. 서비스 파일은 저장소가 `%h/workspace/dgx_spark_init`에 있다는
현재 DGX Spark 배치를 기준으로 한다. 경로가 다르면 설치 전에
`deploy/systemd/sparktalk.service`의 두 경로를 수정한다.

```bash
cd /home/edp1096/workspace/dgx_spark_init/util/chat
make dist
mkdir -p ~/.config/systemd/user
cp deploy/systemd/sparktalk.service ~/.config/systemd/user/sparktalk.service
systemctl --user daemon-reload
systemctl --user enable --now sparktalk.service

# 로그아웃·재부팅 후에도 사용자 서비스를 시작한다. 환경에 따라 관리자 권한이
# 필요할 수 있다.
loginctl enable-linger "$USER"
```

배포본을 다시 만든 뒤에는 실행 중 프로세스가 새 바이너리를 읽도록 서비스를
재시작한다.

```bash
systemctl --user restart sparktalk
systemctl --user status sparktalk
journalctl --user -u sparktalk -f
```

해제할 때는 다음 순서로 처리한다. SQLite·설정·미디어 파일은 삭제하지 않는다.

```bash
systemctl --user disable --now sparktalk
rm ~/.config/systemd/user/sparktalk.service
systemctl --user daemon-reload
```

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
외형 설정에서 **다크**, **라이트**, **시스템 설정 따름** 테마를 선택할 수
있습니다. 시스템 모드는 브라우저가 보고하는 OS 색상 설정 변경을 실시간으로
반영하며, 선택값은 `appearance.theme`에 저장됩니다.

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

## 승인형 SSH 도구

`SparkTalk Extra`의 SSH 서비스를 실행하면 웹 설정에서 서버 프로필을 등록하고
채팅에서 해당 별칭을 지정할 수 있습니다. 모델이 제안한 대상·명령·목적을 화면에
먼저 표시하며 사용자가 **이번만 실행** 또는 **이 대화에서 허용**을 선택해야 첫
SSH 요청이 시작됩니다. stdout/stderr는
편집하지 않은 원문으로 실시간 표시되고 종료 코드와 소요 시간도 함께 저장됩니다.
첫 연결이면 독립 승인 카드에 SHA256 호스트 지문과 **키 신뢰 후 이번만**,
**키 신뢰·대화 허용**이 표시되어 별도로 설정 화면을 왕복하지 않고 안전하게
`known_hosts` 등록까지 처리합니다.

승인 카드에는 **거부**, **이번만 실행**, **이 대화에서 허용**이 표시됩니다.
`이 대화에서 허용`은 현재 대화방과 해당 SSH 서버 조합에만 적용되며 다른 대화방에는
전파되지 않습니다. 허용 관계는 SQLite에 ID만 저장되어 SparkTalk을 재시작해도
유지됩니다. 채팅 상단의 **SSH 허용** 메뉴에서 서버별 또는 전체 해제가 가능하며,
서버 프로필 수정·삭제 또는 대화방 삭제 시에는 자동 해제됩니다. 영구 전역 무승인
설정은 제공하지 않습니다.

SSH 프로필은 SQLite에 저장하지만 개인키 본문과 비밀번호는 저장하지 않습니다.
전용 키는 저장소 밖의
`/home/edp1096/.local/share/sparktalk/extra/ssh/keys`에 두며 가져오기·호스트 키
확인 방법은 `compose_yaml/sparktalk_extra/README.md`를 참고합니다. 현재 구현은
개인키 인증과 비대화형 명령만 지원하며 기본값은 명령별 승인입니다.

여기서 `키 ID`는 비밀번호가 아니라 Extra에 import한 개인키 파일의 별명입니다.
서버 계정 비밀번호와 암호화된 개인키의 passphrase는 지원하지 않습니다. 키 생성,
대상 서버의 `authorized_keys` 등록, `make ssh-key-import`, 화면 입력 예시는 Extra
README의 **SSH 키** 절에 순서대로 작성되어 있습니다.

웹 설정에서 키 ID를 정하고 **Ed25519 생성**을 누르면 개인키는 Extra의 외부 키
폴더에 직접 생성되고 공개키만 화면에 표시됩니다. 기존 개인키 파일 가져오기는
HTTPS 또는 localhost 접속에서만 허용하며 원격 HTTP에서는 보안을 위해 차단합니다.

## 이미지 인식

입력창의 `＋` 버튼, 드래그 앤 드롭 또는 클립보드 붙여넣기로 PNG·JPEG·WebP
이미지를 한 메시지에 최대 6개까지 첨부할 수 있습니다. 이미지당 최대 크기는
15MB이며, 비전 입력을 지원하는 OpenAI 호환 모델이 필요합니다. 첨부 이미지는
SQLite 파일 옆의 `<database>.media` 디렉터리에 저장되고 질문 수정·답변 재시도·
버전 전환에서도 함께 보존됩니다.

## 음성·영상 인식

음성 파일은 원본을 대화 모델에 직접 보내지 않고 SparkTalk Extra Media에서 16kHz
mono PCM WAV로 변환한 뒤 Qwen3-ASR의 전사문을 전달합니다. 영상 파일은
Qwen VL이 볼 수 있는 영상 원본과 음성 트랙의 전사문을 함께 전달합니다.
음성 트랙이 없는 영상은 영상만 전달합니다.

기본 서비스 주소는 SparkTalk Extra Media `http://127.0.0.1:8698`, Qwen3-ASR
`http://127.0.0.1:8694`입니다. 설정 화면의 **음성 인식**에서 활성화 여부,
두 endpoint, ASR 모델, 인식 언어, 문맥·전문용어 힌트와 타임아웃을 관리할
수 있습니다. 서비스 상태도 같은 영역에 표시됩니다.

입력창의 마이크 버튼을 누르면 최대 5분까지 녹음하고, 다시 누르면 즉시
Qwen3-ASR로 전사해 입력창에 문장을 넣습니다. 이 받아쓰기 녹음은 첨부 파일이나
DB에 저장하지 않습니다. 브라우저 마이크 API는 안전한 출처에서만 제공되므로
HTTPS·localhost를 사용하거나 개발용 Chromium에서 접속 주소 전체(스킴·IP·포트)를
`unsafely-treat-insecure-origin-as-secure` 목록에 등록해야 합니다. 주소창의 사이트
권한에서도 마이크를 허용해야 합니다.

PC 상단과 모바일 우측 설정 패널의 **음성대기**를 켜면 마이크 스트림을 유지하고
주변 소음에 맞춰 발화 시작과 약 1초의 침묵을 자동 감지합니다. 발화별 녹음은
감지 시점보다 앞선 약 0.45초의 PCM 프리롤을 포함하므로 첫 음절 손실을 줄입니다.
각 발화는 독립된 WAV로 만들어 Qwen3-ASR로 순서대로 전사한 뒤 즉시 전송합니다. 이전 답변이 생성 중이면 전사문을
입력창에 보관했다가 답변이 끝난 직후 자동 전송합니다. 연속 모드는 매번 명시적으로
켜야 하며 페이지를 새로 열 때 자동으로 마이크를 활성화하지 않습니다.
설정의 추임새 필터를 켜면 문장부호만 있는 결과와 `아`, `어`, `음`, `흠`, `큼`
같은 단독 발화는 자동 전송하지 않습니다. 수동 마이크 입력과 첨부 파일 전사에는
이 필터를 적용하지 않습니다.

전사 결과는 `<database>.media`에 첨부 ID별 작은 캐시로 저장됩니다. 답변
재시도와 이후 대화에서 같은 파일을 다시 인식하지 않으며, 언어·힌트·모델
설정이 바뀌면 자동으로 다시 인식합니다. 미사용 미디어를 정리하면 해당 전사
캐시도 함께 삭제됩니다.

## 답변 음성

AI 답변 아래의 **읽기**를 누르면 SparkTalk Go 백엔드가 Qwen3-TTS CustomVoice
API로 화면에 보이는 답변 전체를 보내고, 24 kHz PCM을 도착하는 즉시
브라우저에서 연속 재생합니다. 음성대기 모드에서는 답변 전체가 끝날 때까지 기다리지
않고 생성 중 완성된 문장을 세 문장 또는 약 140자 단위로 묶어 같은 재생 큐에
넣습니다. 요청마다 억양이 초기화되는 현상을 줄이면서 조기 재생을 유지하기 위한
절충입니다. 설정의
**답변 음성**에서 endpoint, 모델, 언어, 화자, 연기 지시, 시드와 답변 완료 후
자동 재생 여부를 관리합니다. Markdown 표식과 내부 도구 호출문은 읽지 않습니다.
시드가 `-1`이면 브라우저가 답변마다 임의 seed를 하나 만들고 해당 답변의 모든
음성 묶음에 공유합니다. 고정값을 입력하면 모든 답변에 그 값을 사용합니다.

음성대기와 함께 사용할 때는 AI 음성 재생 중 PCM 발화 판정을 일시 중지하고,
재생이 끝나거나 사용자가 정지하면 자동으로 대기 상태로 돌아갑니다. 따라서
스피커로 나온 답변이 다시 ASR을 거쳐 질문으로 전송되는 순환을 방지합니다.

입력창의 `⌁` 버튼에 YouTube·Vimeo·Dailymotion 등 yt-dlp가 인식하는 URL을
넣으면 SparkTalk 백엔드가 SparkTalk Extra Media를 통해 단일 미디어를 취득하고 일반
파일 첨부와 동일하게 대화방에 보관합니다. 브라우저에서 바로 재생할 수 있으며,
음성·영상 분석도 기존 ASR/VL 경로를 그대로 사용합니다. URL 첨부도 파일당 64MB,
메시지당 총 96MB 제한을 적용합니다.

설정의 **도구 > URL 미디어 자동 가져오기**가 켜져 있으면, URL을 메시지에
쓰고 “이 영상을 요약해줘”처럼 분석을 요청할 때 모델이 `media_import` 도구를
선택할 수 있습니다. 모델이 임의 주소를 받지 못하도록 사용자가 해당 대화에
직접 입력한 정확한 URL만 허용합니다. 취득한 파일은 해당 사용자 메시지에
첨부·저장되고, 같은 응답 라운드의 Qwen3.8 VL/ASR 입력으로 자동 전달됩니다.

기본 listen address는 `0.0.0.0:8585`입니다. 같은 네트워크의 다른 PC에서는
`http://서버-IP:8585`로 접속합니다.
