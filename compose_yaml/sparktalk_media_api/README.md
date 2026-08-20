# SparkTalk Media API

SparkTalk의 로컬 영상·음성을 변환하고 일반 웹 미디어 URL을 취득하는 자체 HTTP
API다. FFmpeg/FFprobe, yt-dlp와 Deno를 하나의 ARM64/AMD64 이미지로 제공한다.
Cloudflare 통과나 브라우저 자동화가 필요한 사이트는 `media_access_api`가 담당한다.

## 구성

- 폴더: `compose_yaml/sparktalk_media_api`
- 서비스/컨테이너: `sparktalk-media-api`
- 이미지: `sparktalk-media-api:latest`
- 기본 주소: `http://127.0.0.1:8698`
- 최대 URL 영상 해상도: 720p

URL 영상은 영상·음성 예상 용량을 합산해 제한 안에서 가장 높은 해상도를 고른다.
YouTube 자동 더빙이 있으면 `language_preference`와 기본 언어를 이용해 원본 음성을
선택한다. 다운로드 후에는 무손실 MP4 remux로 영상·음성 종료 시점을 맞추고
faststart 인덱스를 다시 작성해 Decord/Qwen VL의 EOF 오류를 방지한다.

## 실행

```bash
make build
make up
make logs
```

```bash
make down
make clean
```

## API

```bash
curl http://127.0.0.1:8698/health

curl --data-binary @input.mp4 -H 'Content-Type: video/mp4' \
  http://127.0.0.1:8698/v1/probe

curl -F file=@input.mp4 \
  'http://127.0.0.1:8698/v1/audio/extract?sample_rate=16000&channels=1' \
  -o audio.wav

curl -F file=@input.webm \
  http://127.0.0.1:8698/v1/video/normalize -o video.mp4

curl -H 'Content-Type: application/json' \
  -d '{"url":"https://www.youtube.com/watch?v=VIDEO_ID"}' \
  http://127.0.0.1:8698/v1/source/probe

curl -H 'Content-Type: application/json' \
  -d '{"url":"https://www.youtube.com/watch?v=VIDEO_ID","max_download_mb":64,"max_height":720}' \
  http://127.0.0.1:8698/v1/source/download -o media.mp4
```

재생목록·라이브·사설망 URL은 받지 않는다. 유료·DRM·로그인 전용 콘텐츠를
우회하지 않는다. `max_download_mb`, `max_duration_seconds`, `max_height`는 서버
설정 이하의 요청별 제한이다.

## yt-dlp 선택 업데이트

이미지에는 `versions.env`로 고정한 버전이 들어가며, 사이트 변경 때만 Docker
볼륨의 override를 갱신한다.

```bash
make ytdlp-version
make ytdlp-update
make ytdlp-update YTDLP_UPDATE_TO=stable
make ytdlp-update YTDLP_UPDATE_TO=stable@2026.07.04
make ytdlp-reset
```

`make clean`은 업데이트 볼륨을 보존하고 `make purge`는 볼륨도 제거한다.

## 설정

| 환경 변수 | 기본값 | 의미 |
|---|---:|---|
| `SPARKTALK_MEDIA_API_BIND_ADDR` | `127.0.0.1` | 호스트 공개 주소 |
| `SPARKTALK_MEDIA_API_PORT` | `8698` | 호스트 포트 |
| `SPARKTALK_MEDIA_API_MAX_UPLOAD_MB` | `512` | 요청당 업로드 상한 |
| `SPARKTALK_MEDIA_API_MAX_DOWNLOAD_MB` | `4096` | URL 다운로드 상한 |
| `SPARKTALK_MEDIA_API_MAX_DURATION_SECONDS` | `14400` | URL 미디어 길이 상한 |
| `SPARKTALK_MEDIA_API_MAX_VIDEO_HEIGHT` | `720` | URL 영상 해상도 상한 |
| `SPARKTALK_MEDIA_API_MAX_CONCURRENCY` | `2` | 동시 작업 수 |
| `SPARKTALK_MEDIA_API_TIMEOUT_SECONDS` | `1800` | 작업 제한 시간 |
| `SPARKTALK_MEDIA_API_TMPFS_SIZE` | `6g` | 요청 임시공간 상한 |
| `PUID` / `PGID` | `1000` | 컨테이너 실행 사용자 |

임의 FFmpeg 명령 실행 API는 제공하지 않는다. 서드파티 재배포 고지는
`THIRD_PARTY_NOTICES.md`를 참고한다.
