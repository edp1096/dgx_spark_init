# Media Access API

배포 이미지는 `ghcr.io/edp1096/media-access-api:v1.0.1`이다. 현재 이미지는
Linux ARM64용이며 `latest`, `v1`, 정확한 릴리스 태그와 소스 커밋 태그를 함께
제공한다.

AI 모델과 분리된 미디어 입력·전처리 API다. 파일 또는 URL을 받아 `ffmpeg`로
16kHz mono WAV 구간을 만들고 manifest와 함께 ZIP 스트림으로 반환한다. 구간은
설정된 최대 길이 안에서 FFmpeg가 찾은 무음의 중앙을 우선 경계로 사용한다.
영상 입력은 재인코딩 없는 MP4 remux를 우선 시도하고, 음성 전용 입력은 웹 재생이
안정적인 M4A(AAC)로 보관한다. 자산은 컨테이너의
`/data/media/<asset-id>`에 영구 보관한다. 기본 호스트 경로는
`${HOME}/.local/share/media-access-api`이며 `MEDIA_ACCESS_DATA_DIR`로 바꿀 수 있다.
컨테이너를 교체해도 bind mount된 미디어와 브라우저 세션은 유지된다.

URL 처리는 `yt-dlp + Deno/EJS`를 우선 사용한다. 일반 추출에 실패하면 Chromium을
일반 프로세스로 먼저 실행한 뒤 고정 로컬 CDP 포트에 Playwright가 후접속하여
MP4/HLS/DASH 요청을 찾는다. 이 방식은 Playwright 기본 런처의 자동화 플래그를
사용하지 않아 `navigator.webdriver`를 `false`로 유지한다. 동일 세션의 쿠키와
헤더로 다시 내려받으며 브라우저 인증 상태는 `/data/sessions`와
`/data/direct-chromium/profiles`에
사이트·브라우저별로 저장된다. `cf_clearance`를 비롯한 쿠키 값은 코드나 설정에
넣지 않으며 실제 브라우저가 받은 값만 보관한다.

직접 실행 Chromium이 실패할 때만 `browseforge-chromium`, Camoufox, Firefox를
후순위 호환 경로로 사용한다. 도메인마다 프로필을 분리하고
BrowseForge REST API로 루트·상세 페이지를 탐색한 뒤 DOM과 Performance API에서
본영상 후보를 얻는다. 쿠키는 같은 프로필에서 함께 내보내며 해석이 끝난 브라우저
세션은 닫아 메모리를 회수한다.

BrowseForge v2.1.12 Go 서버와 ARM64 전용 Chromium은 Media Access API 이미지에
포함된다. 컨테이너 안의 Xvfb에서 headed Chromium을 실행하고 REST API는
`127.0.0.1:19280`에만 바인딩한다. 외부 BrowseForge 컨테이너, API 토큰 마운트,
19280/6901 포트 공개는 필요하지 않다. 프로필과 토큰은
`/data/browseforge`에 보관되며 컨테이너 교체 후에도 유지된다. BrowseForge 내부의
Camoufox와 CloakBrowser 런타임은 비활성화되어 있다.

직접 실행 Chromium의 Cloudflare 확인은 별도 Xvfb `:98`에서 처리한다. 대부분의
Managed Challenge는 자동으로 끝나며, 대화형 체크박스가 남을 때만 페이지
스크린샷을 OCR하여 위치를 찾고 `xdotool`의 X11 입력으로 클릭한다. CDP는
`127.0.0.1:19281`에만 열리고 단일 브라우저 작업 잠금으로 보호된다.

직접 실행 Chromium이 실패하고 BrowseForge Chromium에도 확인 화면이 남으면 이미지 내부의 공식 Camoufox
작업기를 사용한다. Camoufox는 전용 Playwright 1.60 가상환경에서
`playwright-captcha`의 Shadow DOM 탐색을 실행하며, 외부 포트나 별도
컨테이너를 만들지 않는다. 도메인별 영속 프로필은 `/data/camoufox/profiles`에
보관한다. 확인을 마친 같은 Camoufox 세션에서 미디어 요청과 쿠키·실제 UA·Referer를
함께 수집하므로 다른 브라우저의 UA로 위장하거나 `cf_clearance`를 하드코딩하지 않는다.
대화형 체크박스가 나타나면 `playwright-captcha`는 iframe과 checkbox 좌표만 찾고,
실제 포인터 이동과 button down/up은 작업별 독립 Xvfb 화면에서 `xdotool`의 XTest
입력으로 발생시킨다. 포인터는 곡선으로 iframe 주변 여러 지점을 살핀 뒤 잠시
머물고, 체크박스를 약간 지나쳤다가 되돌아와 클릭한다. 좌표 탐지가 실패할 때만
브라우저 입력을 보조 경로로 사용한다.
브라우저 지문은 실제 실행 환경과 모순되지 않도록 Camoufox의 수집된 Linux preset으로
고정하며 UA·platform·oscpu·WebGL·폰트 묶음을 함께 선택한다. UA 문자열을 별도로
덮어쓰지 않는다. 현재 KR 출구 네트워크와 일치하도록 브라우저·컨테이너 시간대는
`Asia/Seoul`로 고정한다. WebGL 데이터와 호환되는 preset만 사용하며 도메인 해시로
preset과 Canvas·Audio·Font seed를 고정해 컨테이너 재시작 후에도 같은 도메인은 같은
장치 지문을 유지한다.
ARM64 Linux에서도 Firefox 123 이상은 웹 호환성과 지문 축소를 위해 UA,
`navigator.platform`, `navigator.oscpu`를 의도적으로 `Linux x86_64`로 반환한다.
따라서 실제 CPU 문자열을 `aarch64`로 수동 덮어쓰지 않고 Firefox의 표준 노출값을
유지한다.
Camoufox ARM64 브라우저는 v152 릴리스 아카이브와 SHA-256을 `versions.env`에
고정해 빌드 중 검증하며, 실행 중에는 브라우저나 기본 애드온을 내려받지 않는다.

SupJav처럼 Chromium TLS 연결을 리셋하면서 Firefox 계열만 Challenge까지 허용하는
사이트는 Playwright 없는 직접 Camoufox 경로를 먼저 사용한다. 전용 Xvfb 화면에서
Tesseract로 체크박스 문구를 찾고 X11 입력으로 검증한 다음, 같은 브라우저에서
`view-source:`를 열어 `xclip`으로 HTML을 회수한다. 이 소스에서 파트·서버 이름과
`data-link`를 읽으므로 인증 이후에도 CDP, WebDriver, Playwright를 연결하지 않는다.
영속 프로필은 `/data/direct-camoufox/profiles`에 보관한다.

사이트 진입 정책은 `site_adapters.py`의 독립 어댑터가 담당하고 범용 브라우저
엔진은 페이지 실행, 미디어 후보 수집, 세션 보관만 담당한다. `missav888.com`,
`.net`, `.org`는 이름이 비슷해도 별도 서비스로 취급하며 요청 호스트를 다른
도메인으로 바꾸지 않는다. 각 어댑터는 해당 호스트 안에서만 루트 선접속이나
`/enter` 같은 절차를 수행한다. SupJav도 별도 어댑터에서 루트 선접속을 처리한다.
SupJav의 복수 서버는 `ST`/`StreamTape`/`Stream Tape`를 먼저 시도하고, 없거나 유실되었으면
페이지에 표시된 다른 서버를 위에서부터 순서대로 시도한다. 라이브 위젯의 광고
HLS는 본영상 후보에서 제외한다.
`POST /v1/media/options`는 SupJav의 숫자 파트와 파트별 영상 출처를 반환한다.
`/v1/media/prepare`에 `media_part`와 `media_source`를 전달하면 페이지의 초기
플레이어와 무관하게 지정한 조합만 해석한다. 출처를 생략하면 선택한 파트 안에서
StreamTape 계열 이름을 우선하고 나머지를 화면 순서대로 시도한다. 출처는 고정 목록으로
제한하지 않는다. `SERVER:` 뒤의 `Streamtape`, `Mixdrop`, `NinjaStream` 같은 임의의
풀네임을 읽어 그대로 반환한다.
페이지에 섞인 광고 MP4보다 정상 HLS/DASH와 높은 해상도 응답을 우선한다.
HLS CDN이 일부 조각에 403/5xx를 반환하면 조각을 병렬 재시도하고, 최고 화질을
완성하지 못한 경우 720p와 480p 순으로 낮춰 완전한 파일을 받는다. 누락된 조각이
있는 부분 파일은 성공으로 처리하지 않는다.
Vimeo의 전용 `playlist.json`은 최고 해상도 비디오와 기본 오디오 조각을 제한된
병렬 요청으로 조립한 뒤 MP4로 remux한다. Dailymotion은 yt-dlp 기본 추출기를 쓴다.

`GET /v1/media/storage`는 `prepare-*` 임시 폴더의 전체·활성·정리 가능 용량을
반환한다. `DELETE /v1/media/storage/temp`는 현재 처리 중인 폴더를 제외하고 정리하며,
`older_than_hours`를 주면 마지막 변경 시각이 그보다 오래된 폴더만 삭제한다.

`POST /v1/media/thumbnails`는 multipart의 `video` 파일에서 타임라인 미리보기용
160×90 프레임 50개를 추출해 10×5 JPEG 스프라이트로 반환한다. 재생 UI는 이 한 장의
좌표만 바꾸므로 마우스 이동 중 원본 영상을 반복 탐색할 필요가 없다.

Deno는 이미지에 포함되어 있으며, YouTube의 JavaScript challenge solver는
`yt-dlp[default]`에 포함된 `yt-dlp-ejs`를 사용한다.
YouTube GVS가 요구하는 영상별 PO Token은 공식 yt-dlp 문서의 추천 구성에 따라
`bgutil-ytdlp-pot-provider` 플러그인과 이미지 내부 Deno HTTP 서버가 자동 발급한다.
서버는 `127.0.0.1:4416`에만 바인딩하며 외부 컨테이너나 공개 포트가 필요하지
않다. Media API, BrowseForge, PO Token 서버, Xvfb 중 하나가 종료되면 컨테이너도
종료되어 Docker의 재시작 정책으로 전체 프로세스를 일관되게 복구한다.

```bash
make rebuild
curl http://127.0.0.1:8697/health
```

## 버전 관리와 롤백

외부 런타임과 Python 패키지 버전은 `versions.env`에서 한 번에 관리한다.
BrowseForge, bgutil provider, Playwright 베이스 이미지는 버전 태그뿐 아니라 OCI
manifest digest도 고정하므로 같은 ARM64 이미지를 다시 받는다. 업데이트 확인은
파일을 변경하지 않으며, 승인한 버전만 명시적으로 적용한다.

```bash
make show-versions
make check-updates
make set-version COMPONENT=yt-dlp VERSION=2026.7.4
make set-version COMPONENT=bgutil VERSION=1.3.1
make set-version COMPONENT=browseforge VERSION=v2.1.12
make set-version COMPONENT=playwright VERSION=1.62.0
make set-version COMPONENT=camoufox VERSION=0.5.5
make set-version COMPONENT=playwright-captcha VERSION=0.1.5
make rebuild
make verify
```

이미지 업데이트 전후에는 작동이 확인된 이미지를 태그로 보관하고 즉시 되돌릴 수
있다. `snapshot`과 `rollback`은 데이터 볼륨을 변경하지 않는다.

```bash
make snapshot TAG=media-20260819
make rollback TAG=media-20260819
```

`make verify`는 단일 컨테이너 구성, 내부 서비스 상태, 라이선스 파일, Deno JS
solver 및 실제 YouTube mweb PO Token 발급을 내려받기 없는 simulate 요청으로
검증한다. FFmpeg는 Ubuntu 저장소 패키지이므로 완전한 패키지 단위 동결 대상은
아니며, 보안 업데이트를 반영한 깨끗한 빌드 뒤 같은 검증을 수행한다.

파일 처리 예시:

```bash
curl -o prepared.zip \
  -F file=@movie.mkv \
  -F segment_seconds=180 \
  http://127.0.0.1:8697/v1/media/prepare
```

manifest의 `asset.id`가 존재하면 영상 또는 음성 스트리밍과 삭제가 가능하다.
`asset.media_type`은 `video` 또는 `audio`이고, Range 응답을 사용하므로 긴 미디어도
웹 플레이어에서 임의 위치로 탐색할 수 있다.
`/v1/media/prepare`에 `request_id`를 함께 보내면
`/v1/media/progress/<request_id>`에서 다운로드 바이트·총량·퍼센트·ETA와 저장 및
음성 분리 단계를 조회할 수 있다. 이 경우 `prepare-<request_id>` 폴더에 원본과
단계별 체크포인트를 기록한다. 앱이나 컨테이너가 중단된 뒤 같은 `request_id`로
다시 요청하면 검증된 원본 다운로드를 건너뛰고 영상 보관 또는 음성 분할부터
재개한다. 중단 시 남은 불완전한 WAV 조각은 재사용하지 않고 원본에서 다시 만든다.
작업 관리자가 조회를 마치면 progress 항목을 DELETE로 정리한다.

진행 중인 준비 작업 하나만 취소하려면 같은 `request_id`로 다음 API를 호출한다.
해당 요청에 등록된 yt-dlp 또는 FFmpeg 프로세스만 종료하며 다른 작업과 영구 보관된
미디어에는 영향을 주지 않는다. 중간 파일이 든 `prepare-<request_id>` 폴더는
비활성 임시 폴더로 남아 저장소 정리 API에서 회수할 수 있다.

```bash
curl -X DELETE \
  http://127.0.0.1:8697/v1/media/prepare/REQUEST_ID
```

```bash
curl -H 'Range: bytes=0-1048575' \
  http://127.0.0.1:8697/v1/media/assets/ASSET_ID
curl -X DELETE http://127.0.0.1:8697/v1/media/assets/ASSET_ID
```

URL 처리 예시:

```bash
curl -X POST -F url=https://supjav.com/206680.html \
  http://127.0.0.1:8697/v1/media/options

curl -o prepared.zip \
  -F url=https://example.com/video-page \
  -F segment_seconds=180 \
  http://127.0.0.1:8697/v1/media/prepare
```

파트와 출처를 지정하는 경우:

```bash
curl -o prepared.zip \
  -F url=https://supjav.com/206680.html \
  -F media_part=2 \
  -F media_source=ST \
  -F segment_seconds=180 \
  http://127.0.0.1:8697/v1/media/prepare
```

Cloudflare 확인 화면은 직접 실행 Chromium과 X11 클릭 방식으로 먼저 처리한다.
Cloudflare가 브라우저나 네트워크 위험도를 거부하면 CAPTCHA 대행 API로
우회하지 않고 명확한 오류를 반환한다.
