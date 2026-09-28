# 최초 설치 보완 결과 — 2026-09-29

아래는 최초 점검 기록이며, 발견된 구현 누락은 다음과 같이 보완했다.

- 전체 build asset을 내장해 FLUX Python 모듈 누락을 해결했다. 등록된 모든 구성요소의 빌드 입력을 검사한다.
- QAD 공개 Huihui 모델, Ornith·Gemma26 본체/draft, ASR·화자 GGUF, TTS 모델/코덱/토크나이저 준비를 연결했다. 기존 정상 가중치는 재사용한다.
- FLUX 로컬 베이스 의존성을 제거하고 최초 텍스트 인코더 변환·본체·VAE·LoRA·rembg 준비를 연결했다. 신규 세트 콜드 스타트에서는 큰 LLM 적재 전에 FLUX 준비를 수행한다.
- Qwen TP2와 DS41에 이미지 빌드·다운로드·워커 전달 경로를 추가했다. DS41은 양쪽 rank의 expert packing을 수행한다.
- DS4FVE ablit은 빈 캐시에서 원본을 먼저 받는다. 토큰 미등록 시 range 요청 문자열 오류를 수정했다.
- 클러스터 4종을 동일 패키징 manifest에서 관리·검사한다. ASR·Ornith·Gemma26 빌드 자산도 정규 소스와 동기화 검사한다.
- Extra SSH와 일반 모델 다운로드·TTS 변환은 실행 호스트 UID/GID를 사용한다. 배포 바이너리는 정식 Makefile의 CGO_ENABLED=0 경로로 빌드한다.
- 초기 다운로드/빌드 대기 시간을 확장하고 모델 준비 화면에 모든 모델 서비스를 노출했다. 호스트 드라이버·Docker·권한·SSH 준비는 별도 전제이며 앱이 임의 설치하지 않는다.

검증 범위: 전체 Go 테스트, Python 초기 다운로드/누락·손상 검사, mock CLI를 통한 두 클러스터의 빈 이미지·가중치 준비 흐름, 패키징/COPY 검사, 프런트엔드 빌드. 실제 전체 대형 모델의 무캐시 다운로드·GPU 추론 전수 재검증과 구분한다. 추가로 실제 화자 구분 GGUF 107,012,128바이트를 빈 폴더에 다운로드해 SHA-256을 확인했고, 네트워크 차단 후 재사용도 통과했다. Linux ARM64 정적 실행 파일만 마운트한 빈 환경에서 설정·DB 생성, UI·JS·CSS·health HTTP 200을 확인했다.

---

# Talk 최초 설치 전수 점검 — 2026-09-29

결론: 실행 파일만 복사한 앱 기동은 가능하지만, 현재 모든 AI 세트를 빈 머신에서 준비·실행할 수 있는 배포는 아니다. 기존 로컬 이미지·가중치 때문에 드러나지 않았던 의존성이 있다.

## 범위와 실제 수행

- 내장 catalog: 모델 세트 8개, 구성요소 16개 전부 조사. 미사용 기본 구성요소 DreamLite도 포함.
- 실행 파일만 마운트한 격리 컨테이너: 빈 작업 디렉터리와 HOME, Docker 소켓/모델 캐시/네트워크 없음. 설정·SQLite 생성, UI HTML·JS·CSS, health API 200 확인. 이는 전체 GPU 모델의 클린 설치 시험이 아니다.
- 12개 단일 서비스: Go에 실제 내장된 자산을 임시 폴더에 풀고 `docker compose config --quiet` 전부 통과. 모델 기동 성공을 뜻하지 않는다.
- 클러스터 패키지 4개 모두 해제·검토. 내장 shell 54개 `bash -n`, Python 201개 AST 파싱 통과.
- 실제 Go EmbedFiles 기준 Dockerfile COPY 입력 39개 검사: FLUX의 `phased/` 누락 발견.
- `package-support --check` 70개 파일 통과. `package-recipes --check`는 GLM 16개·DS4FVE 240개 파일만 검사한다. DS41·Qwen TP2 아카이브는 존재하지만 이 동기화 검사 manifest에는 없다.
- `go test ./...` 통과. 모델 통합 시험은 opt-in이므로 이것으로 빈 캐시 모델 기동 성공을 주장하지 않는다.
- Huihui QAD의 빈 경로 검증을 실행해 로컬 변환본 요구 오류 재현. DS41·Qwen TP2 `runtime.sh setup`을 실행해 exit 2 재현.
- 운영 모델·호스트 설정·이미지·가중치를 삭제하거나 다시 구축하지 않았다. 대규모 다운로드, 외부 레지스트리 가용성, 실제 포맷 OS에서의 모든 모델 기동은 수행하지 않았다.

## 구성요소 전체

| 구성요소 | 이미지 준비 | 가중치 준비 | 빈 머신 판정 |
|---|---|---|---|
| Qwen QAD TP1 | 내장 Dockerfile 자동 빌드 | official/huginn은 snapshot_download; huihui_lil은 로컬 검증만 | 공식·huginn 경로는 존재하나 cold build 미검증. 현재 사용 중인 Huihui QAD는 차단 |
| Qwen TP2 | 로컬 이미지 존재 확인만 | 기존 로컬 Huihui-RadixArk 경로 존재 확인만 | setup/model 자동 준비 없음, exit 2 |
| Ornith 35B | pull_policy never, build 자산 매핑 없음 | 로컬 /hf/edp1096 경로, offline 모드 | 이미지·가중치 둘 다 사전 준비 필요 |
| Gemma 26B | pull_policy never, build 자산 매핑 없음 | 본체와 assistant 로컬 경로, offline 모드 | 이미지·본체·draft 사전 준비 필요 |
| Gemma 31B | 내장 Dockerfile.dflash로 빌드 | 본체·DFlash Hub ID/revision 지정, 실행 시 로딩/다운로드 | 준비 경로 존재, 전체 cold build/추론 미검증 |
| GLM 5.3 | 전체 준비에서 내장 Dockerfile 빌드, worker 전달 | 공식·Huihui와 DFlash 다운로드 후 rsync | 전체 준비 경로 존재, 두 호스트 전제. model-only는 서빙 이미지가 먼저 필요 |
| DeepSeek V4 Flash Vision | 공개 이미지 pull 및 worker pull | 공식 다운로드·동기화; ablit은 원본 shard를 이용한 선택 패치 | 공식 준비 경로 존재. 빈 캐시에서 ablit부터 선택하면 원본 shard 부족으로 실패 |
| DeepSeek V4.1 | 준비된 b12x8 이미지 요구 | pinned checkpoint 및 rank별 packed expert 요구 | setup/model/image 자동 준비 미지원, exit 2 |
| FLUX | 내장 paint 빌드는 로컬 dgx-flux2-klein-nvfp4:4b에 의존 | 베이스 이미지의 준비 스크립트에 의존 | 로컬 베이스 자동 준비 없음 + phased COPY 자산 누락 |
| DreamLite | NGC PyTorch 기반 내장 Dockerfile | 고정 Hub revision from_pretrained | 경로 존재, cold build/다운로드 미검증; 기본 세트에는 미포함 |
| Nemotron ASR | 내장 빌드 매핑 없음. 이미지가 없으면 먼저 빌드하라는 오류 | 로컬 ASR·diarization GGUF 사용 | 이미지 및 모델 준비 누락 |
| Magpie TTS | 내장 Dockerfile/패치로 runtime 빌드 | 로컬 Magpie GGUF·codec·tokenizer 디렉터리 요구 | 이미지 빌드 경로는 있으나 가중치 준비 연동 없음 |
| Extra Media | 내장 빌드: Go, ffmpeg, yt-dlp, Deno | 별도 신경망 모델 불필요 | 경로 존재, 소스 동기화/Compose 통과. 완전 무캐시 빌드는 미검증 |
| Extra Collector | 내장 빌드: Go, Chromium | 별도 신경망 모델 불필요 | 경로 존재, 소스 동기화/Compose 통과. 완전 무캐시 빌드는 미검증 |
| Extra SSH | 내장 빌드 | 사용자 키·known_hosts 등록 필요 | 경로 존재. UID/GID 기본 1000 의존 주의 |
| Extra Documents | Rust/Go/Node 다단계 내장 빌드 | 코드·라이브러리 포함 | 경로 존재, 소스 동기화/Compose 통과. 완전 무캐시 빌드는 미검증 |

## 8개 모델 세트의 종합 판정

- flash-next: official/huginn 모델 준비 경로는 있으나 공통 FLUX·ASR이 막힌다. huihui_lil 선택 시 본체 준비도 막힌다.
- flash-next-tp2: 본체 준비 미지원 + FLUX·ASR·TTS 가중치 준비 문제.
- ornith35 / gemma26: 본체 이미지·가중치 및 공통 부가 모델 준비 문제.
- gemma: 본체 준비 경로 존재; FLUX·ASR·TTS 준비 문제로 세트 전체는 초기 설치 통과 아님.
- glm53-worker-extra: 이미지·가중치·워커 Extra 준비 경로 존재. SSH·Docker·GPU 런타임·네트워크 전제가 있으며 전체 cold 설치는 미검증.
- ds4fve: 공식 본체 준비 경로 존재. ablit 원본 의존성 및 worker ASR/TTS 준비 문제.
- ds41: 본체 준비 미지원 및 worker ASR/TTS 준비 문제.

## 확인된 구현상 누락

1. `internal/orchestrator/catalog.go`의 `//go:embed assets/*`는 디렉터리를 재귀 포함할 때 `_`로 시작하는 파일을 제외한다. `assets/flux2-paint/phased/__init__.py`가 실제 EmbedFiles에 없다. 따라서 `COPY phased/`는 내장 자산만 펼친 빌드에서 실패한다. 현재 로컬 수동 빌드 이미지가 이를 가렸다.
2. `qwen_qad_prepare.go`는 huihui_lil에 snapshot_download를 수행하지 않고 transfer-manifest/runtime-qualification을 가진 로컬 변환본만 검사한다. Hub에 모델이 있어도 이 경로에서는 가져오지 않는다.
3. `build_assets.go`에 ASR·Ornith·Gemma26 빌드 매핑이 없다. ASR은 `ensureLocalServiceImage`에서 선빌드를 요구하며 나머지 둘은 `pull_policy: never`로 막힌다.
4. TTS Dockerfile에 converter stage는 있지만 Talk가 이를 실행하거나 필요한 가중치를 다운로드하는 경로가 없다.
5. DS41/Qwen TP2 archive의 runtime.sh는 자동 준비 요청을 명시적으로 거부한다.
6. DS4FVE ablit downloader는 원본 shard가 이미 있어야 동작한다. 토큰이 없으면 일부 range 요청에서 `'Bearer '+token`도 TypeError 가능성이 있다.
7. 패키징 검증은 등록된 모든 recipe를 포괄하지 않는다. 기존 `TestEmbeddedCatalogIsComplete`도 flash-next와 gemma 두 세트만 순회한다.
8. README는 GLM EXL3, 과거 이미지 태그 및 준비 설명이 남아 실제 catalog/동작과 일치하지 않는다.

## 호스트와 배포 전제

- Go/Node 개발 환경은 배포 앱 실행에 필요 없다. UI·기본 설정·스킬·한자 낭독 사전·브라우저 확장 ZIP은 내장된다. 브라우저 확장 설치와 사이트 권한 설정은 별도 사용자 작업이다.
- Docker Engine/Compose, GPU 드라이버/Container Toolkit, Docker 실행 권한을 설치해 주는 호스트 부트스트랩은 없다.
- TP2에는 두 머신의 SSH 인증·호스트 신뢰, Python3·rsync·네트워크/RDMA 준비가 필요하다.
- 현재 Linux ARM64 dist 파일은 libc 동적 링크다. Makefile의 정식 dist는 CGO_ENABLED=0이므로 수동 빌드·복사와 정식 산출물 경로도 통일할 필요가 있다. DGX Spark Ubuntu에서 실행된 사실과 모든 OS에서 자립한다는 주장은 구분한다.
- Extra SSH Compose의 UID/GID 기본값은 1000이며 runtimePathEnvironment는 실행 호스트 UID/GID를 전달하지 않는다. UID가 다른 초기 사용자에서는 bind mount 권한 문제가 생길 수 있다.

위 표와 목록은 수정 전 최초 점검 기록이다. 현재 수정 및 검증 범위는 문서 맨 위의 보완 결과를 기준으로 한다.
