# SparkTalk Extra

SparkTalk의 부가기능을 담당하는 자체 Go 서비스 모음이다. 하나의 프로젝트와
Makefile로 관리하지만, 신뢰할 수 없는 미디어 입력과 SSH 개인키가 같은 보안
영역을 공유하지 않도록 두 컨테이너로 실행한다.

| 서비스 | 이미지·컨테이너 | 기본 주소 | 역할 |
|---|---|---|---|
| Media | `sparktalk-extra-media` | `127.0.0.1:8690` | FFmpeg, yt-dlp, Deno를 이용한 미디어 취득·변환 |
| SSH | `sparktalk-extra-ssh` | `127.0.0.1:8699` | 등록 서버 연결 확인, 호스트 키 검증, 승인된 명령 실행 |

Cloudflare 통과나 브라우저 자동화가 필요한 미디어 사이트는 별도의
`media_access_api`가 담당한다.

## 실행

```bash
cd compose_yaml/sparktalk_extra
make test
make build
make up
make ps
```

```bash
make logs
make media-logs
make ssh-logs
make down
make clean
```

`make clean`과 `make purge`는 저장소 밖의 SSH 키·known_hosts를 삭제하지 않는다.
기본 SSH 데이터 위치는 다음과 같다.

```text
/home/edp1096/.local/share/sparktalk/extra/ssh/
├── keys/
└── state/
    └── known_hosts
```

다른 위치는 `SSH_DATA_DIR=/path make up`으로 지정한다.

## SSH 키

SSH는 **공개키 인증만 지원**한다. 서버 계정 비밀번호 인증과 암호화된 개인키의
passphrase 입력은 지원하지 않는다. 키 방식에서는 다음 두 파일이 한 쌍이다.

- 개인키: SparkTalk Extra만 보관하며 외부로 보내지 않는 파일
- 공개키: 접속할 서버의 `~/.ssh/authorized_keys`에 등록하는 파일

`키 ID`는 암호가 아니라 Extra가 사용할 개인키 파일의 별명이다. 영문·숫자와
마침표·밑줄·하이픈을 사용할 수 있으며, 예를 들어 `dgx-main`처럼 정한다.

### DGX Spark 자기 자신에 접속하는 예

가장 간단한 방법은 SparkTalk 웹의 **설정 → SparkTalk Extra → 인증 키**에서
키 ID `dgx-main`을 입력하고 **Ed25519 생성**을 누르는 것이다. 개인키는 Extra가
외부 키 폴더에 직접 생성하므로 브라우저로 전송되거나 컨테이너 이미지에 포함되지
않는다. 화면에 나타난 공개키는 **공개키 복사**로 복사한다.

복사한 한 줄을 접속 대상 계정의 `~/.ssh/authorized_keys`에 추가한다. DGX Spark
자기 자신이라면 터미널에서 다음처럼 등록한다.

```bash
mkdir -p /home/edp1096/.ssh
chmod 700 /home/edp1096/.ssh
printf '%s\n' '화면에서 복사한 ssh-ed25519 공개키 한 줄' \
  >> /home/edp1096/.ssh/authorized_keys
chmod 600 /home/edp1096/.ssh/authorized_keys
```

기존 개인키를 가져오려면 인증 키 영역에서 동일한 키 ID와 파일을 선택한다.
개인키가 평문 네트워크를 지나지 않도록 **HTTPS 또는 localhost로 접속한 웹에서만**
업로드가 활성화된다. 원격 HTTP 접속에서는 서버 내부 Ed25519 생성만 사용할 수 있다.

CLI 등록도 계속 지원한다. 현재 계정에 키가 없다면 passphrase 없는 SparkTalk
전용 키를 만든다.

```bash
mkdir -p /home/edp1096/.ssh
chmod 700 /home/edp1096/.ssh
ssh-keygen -t ed25519 \
  -f /home/edp1096/.ssh/sparktalk_ed25519 \
  -N ""
```

생성된 공개키를 DGX Spark의 로그인 허용 목록에 추가한다.

```bash
cat /home/edp1096/.ssh/sparktalk_ed25519.pub \
  >> /home/edp1096/.ssh/authorized_keys
chmod 600 /home/edp1096/.ssh/authorized_keys
```

개인키를 Extra의 저장소 밖 전용 디렉터리로 가져온다.

```bash
cd /home/edp1096/workspace/dgx_spark_init/compose_yaml/sparktalk_extra
make ssh-key-import \
  SSH_KEY=/home/edp1096/.ssh/sparktalk_ed25519 \
  SSH_KEY_ID=dgx-main
```

SparkTalk 웹의 **설정 → SparkTalk Extra → 서버 추가**에는 다음처럼 입력한다.

```text
표시 이름: DGX Spark
별칭: dgx-main
호스트: 192.168.100.61
포트: 22
사용자: edp1096
키 ID: dgx-main
제한시간(초): 60
```

다른 서버에 접속할 때는 화면에서 복사한 공개키(또는 CLI에서 만든
`sparktalk_ed25519.pub`)를 그 서버 계정의 `~/.ssh/authorized_keys`에 등록하고
화면의 호스트·사용자만 해당 서버에 맞춘다.
`ssh-copy-id`를 이용하면 최초 등록 과정에서만 원격 서버 비밀번호를 요구할 수
있지만, 이후 SparkTalk의 실제 접속에는 비밀번호가 사용되지 않는다.

호스트 주소·사용자·키 ID 같은 프로필은 SparkTalk의 SQLite에 저장한다. 저장 후
`시험`을 누르면 처음 연결할 때 SHA256 호스트 지문이 표시된다. 대상 서버에서
다음 명령으로 제공 중인 모든 호스트 공개키 지문을 확인하고, 화면의 SHA256 값과
일치하는 항목이 있을 때만 `이 키 신뢰`를 누른다. Ed25519만 고정해서 확인하면
서버가 ECDSA 키를 제시하는 경우 엉뚱한 지문과 비교하게 된다.

```bash
for key in /etc/ssh/ssh_host_*_key.pub; do
  ssh-keygen -lf "$key"
done
```

신뢰한 호스트 키는 Extra의 `known_hosts`에 기록된다. 이후 서버 키가 달라지면
자동으로 덮어쓰거나 신뢰하지 않고 연결을 차단한다.

설정에서 미리 `시험`하지 않고 채팅에서 첫 명령을 실행해도 된다. 이 경우 독립된
명령 승인 카드에 새 호스트의 SHA256 지문과 `키 신뢰 후 이번만`, `키 신뢰·대화
허용` 버튼이 함께 나타난다. 승인하면 Extra가 접속 직전에 같은 키인지 다시
확인한 뒤 `known_hosts`에 기록하고 명령을 실행한다. 이미 신뢰한 키가 나중에
바뀐 경우에는 이 버튼을 제공하지 않고 연결을 차단한다.

UI에서 생성하거나 import한 개인키도 저장소·YAML·SQLite·컨테이너 이미지에 넣지
않으며, 호스트의 다음 위치에 0600 권한으로 보관된다. SSH 컨테이너는 이 외부
폴더만 마운트해서 사용하므로 컨테이너 재빌드 후에도 유지된다.

```text
/home/edp1096/.local/share/sparktalk/extra/ssh/keys/dgx-main
```

SSH API는 비밀번호, 암호화된 개인키, 대화형 PTY를 지원하지 않는다. 명령은
SparkTalk 화면에서 기본적으로 개별 승인하며 stdout/stderr, 종료 코드와 소요
시간이 읽기 전용 도구 기록으로 남는다.

명령 승인에서 `이번만 실행`을 선택하면 다음 명령도 다시 묻는다. `이 대화에서
허용`을 선택하면 현재 대화방과 해당 서버에 한해 이후 명령을 자동 승인한다. 이
허용 관계는 SparkTalk SQLite에 대화 ID와 서버 프로필 ID만 저장되므로 앱을
재시작해도 유지된다. 채팅 상단의 **SSH 허용** 메뉴에서 서버별 또는 전체 해제가
가능하며, 서버 프로필 변경·삭제 또는 대화방 삭제 시 자동 해제된다. 다른 대화방과
다른 서버에는 적용되지 않으며 명령, 개인키, 비밀번호는 이 권한 레코드에 저장하지
않는다.

## Media API

컨테이너 내부 API는 8698을 유지하고, 호스트에서는 SeedVR2의 8698과 충돌하지
않도록 8690으로 공개한다.

```bash
curl http://127.0.0.1:8690/health

curl --data-binary @input.mp4 -H 'Content-Type: video/mp4' \
  http://127.0.0.1:8690/v1/probe

curl -F file=@input.mp4 \
  'http://127.0.0.1:8690/v1/audio/extract?sample_rate=16000&channels=1' \
  -o audio.wav

curl -H 'Content-Type: application/json' \
  -d '{"url":"https://www.youtube.com/watch?v=VIDEO_ID","max_download_mb":64,"max_height":720}' \
  http://127.0.0.1:8690/v1/source/download -o media.mp4
```

모델 영상 디코더와의 호환성을 위해 URL 영상은 용량 한도 안에서 H.264
30fps 이하 포맷을 우선한다. 적합한 스트림이 없으면
H.264/yuv420p/AAC MP4로 자동 정규화한다. AV1·VP9·고프레임률
영상에서 발생하는 SGLang/Decord 디코더 오류를 피하기 위한 처리다.

재생목록·라이브·사설망 URL은 받지 않으며 유료·DRM·로그인 전용 콘텐츠를
우회하지 않는다. 기존 `sparktalk_media_api_ytdlp-data` Docker 볼륨을 명시적으로
재사용하므로 프로젝트 이름을 바꿔도 yt-dlp override는 유지된다.

```bash
make ytdlp-version
make ytdlp-update
make ytdlp-update YTDLP_UPDATE_TO=stable
make ytdlp-reset
```

## SSH API

SparkTalk 백엔드 전용 API다. 기본적으로 loopback에만 공개된다.

```bash
curl http://127.0.0.1:8699/health

curl -H 'Content-Type: application/json' \
  -d '{"host":"192.168.100.61","port":22,"user":"edp1096","key_id":"dgx-main"}' \
  http://127.0.0.1:8699/v1/ssh/check
```

`/v1/ssh/exec`은 NDJSON으로 `start`, `stdout`, `stderr`, `exit` 이벤트를
스트리밍한다. 이 API를 외부 네트워크에 직접 공개하지 않는다.

## 주요 환경변수

| 변수 | 기본값 | 의미 |
|---|---:|---|
| `SPARKTALK_EXTRA_MEDIA_PORT` | `8690` | Media 호스트 포트 |
| `SPARKTALK_EXTRA_SSH_PORT` | `8699` | SSH 호스트 포트 |
| `SPARKTALK_EXTRA_SSH_DATA_DIR` | 사용자 데이터 폴더 | 전용 키와 known_hosts 위치 |
| `SPARKTALK_EXTRA_SSH_MAX_CONCURRENCY` | `2` | 동시 SSH 명령 수 |
| `SPARKTALK_EXTRA_SSH_MAX_OUTPUT_MB` | `4` | 명령당 원문 출력 상한 |
| `SPARKTALK_EXTRA_SSH_TIMEOUT_SECONDS` | `300` | 서버측 최대 명령 시간 |

서드파티 재배포 고지는 `THIRD_PARTY_NOTICES.md`를 참고한다.
