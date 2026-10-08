# 지원 서비스 운영

설정 → 시스템 → 지원 서비스에서 Media·Collector·Documents·SSH를 관리한다.
각 카드에 기능 사용 허용, 이미지 준비 여부, 컨테이너 실행 상태, HTTP 준비 상태를 따로 표시한다.
ASR 설정과 관계없이 네 서비스의 실제 상태를 확인한다.

## 시작·중지와 배치

- 모델 세트 시작·중지는 모델 서비스만 대상으로 한다. 지원 서비스는 카드에서 각각 시작·중지·재시작한다.
- `이미지 준비`는 이미지를 빌드하거나 워커로 전달하며 실행 중인 컨테이너를 교체하지 않는다. 새 이미지는 서비스 재시작 시 적용된다.
- 기능 사용 허용 토글은 도구 사용을 허용한다. 컨테이너를 시작하거나 중지하지 않는다.
- 현재 모델 세트에 서비스 배치가 있으면 그 호스트·포트·endpoint를 사용한다. 세트에 없는 서비스는 공통 정의를 사용한다.
- 배치 변경은 서비스를 중지한 뒤 설정을 저장하고 시작한다. 로컬/워커의 데이터는 각 호스트 데이터 디렉터리에 있으며 배치 변경으로 자동 이전되지 않는다.
- 외부 서버 모드에서는 endpoint와 기능 사용 허용을 설정하고 상태를 확인한다. Docker 제어는 관리형 모드에서 제공한다.

## 배포 자산

원본은 `compose_yaml/sparktalk_extra/`다. `services.json`이 서비스 ID·이미지·버전·포트·빌드 자산을 정의한다.
Media·Collector·SSH는 `0.1.0`, Documents는 `0.4.0` 이미지를 사용한다.
독립 실행과 앱 내장 실행은 같은 Compose 원본에서 healthcheck, CPU·메모리·PID 제한,
read-only/tmpfs 및 볼륨 정책을 가져온다. 호스트별 경로와 포트는 실행 환경에 맞춰 변환한다.

```sh
# util/talk에서 실행
make support-assets        # 원본을 앱 내장 자산으로 패키징
make support-assets-check  # 누락·불일치 검사
make dist                  # 일치 검사 후 웹과 배포 바이너리 빌드
```

내장 빌드용 Go 소스와 모듈 파일에는 `.asset` 접미사를 붙인다. Go embed의 중첩 모듈 제외 규칙을
피하기 위한 포장 방식이며, 빌드 디렉터리에 쓸 때 원래 이름으로 복원한다.
`docms/compose.yaml`은 기존 독립 문서 서비스 관리 명령을 위한 생성 파일이다.
공통 Compose와 독립 DocMS Compose를 동시에 실행하지 않는다.

Media의 yt-dlp 업데이트는 기존 영속 볼륨을 재사용한다. SSH의 키와 known_hosts도 기존 호스트 경로를 재사용한다.
컨테이너 재시작으로 이 데이터를 지우지 않는다.

## 설정·API 호환성

`extra.media_endpoint`를 Media 주소의 기준으로 사용한다. 기존 `asr.ffmpeg_endpoint` 설정은
로드·저장 시 호환되도록 동기화한다. 기존 기능 허용 필드도 유지한다.

`GET /api/support`는 서비스별 사용 허용, 설치, 실행, 준비 상태, 목표 이미지와 실행 이미지,
현재 작업을 반환한다. `GET /api/health`의 `extra`에는 네 서비스가 모두 포함된다.
제어는 `POST /api/runtime/components/{id}/{prepare|start|stop|restart}`를 사용한다.

SSH 클라이언트는 `internal/support/ssh`에 있다. Collector의 수집 연동, Media의 ASR 변환 연동,
Documents의 문서 도구 연동은 각 기능 모듈에 남는다.

## 검증 기록 (2026-09-08)

- Go 전체 테스트, 프런트엔드 단위 테스트 83개 및 관련 화면 E2E 15개 통과.
- 내장 자산 57개 일치 검사와 Linux·Windows amd64/arm64 바이너리 빌드 통과.
- 실행 환경에서 네 서비스 이미지 준비 전후 모든 컨테이너 ID 동일.
- 지원 서비스만 순서대로 재시작한 뒤 네 서비스 모두 Docker healthy·HTTP 준비 완료.
- 기존 모델 컨테이너 ID, SSH 저장 파일과 지원 서비스 영속 볼륨 경로 유지.
- Media probe·오디오 추출, Collector Chromium 수집, Documents 한글 PDF·HWP 생성 확인.
- 모델 세트 중지가 지원 서비스를 건드리지 않는 동작과 ASR·기능 허용과 독립적인 health 응답을 테스트로 검증.
