# Extra 서비스 구조 점검

## 적용 결과

후속 구현에서 네 서비스를 공통 지원 서비스 화면·상태 API·독립 시작/중지 정책으로 정리했다.
모델 세트는 배치 선택에만 지원 서비스를 참조하며, 모델 시작·중지는 지원 서비스를 제어하지 않는다.
원본 Compose에서 내장 빌드 자산을 생성하고 배포 빌드에서 일치를 검사한다.
운영 방법과 호환성은 [지원 서비스 운영](SUPPORT-SERVICES.md)을 따른다.

아래는 변경 전 점검 기록이다. 당시 실행 상태와 지적 사항을 보존한다.

## 판단

교통정리가 필요하다. 현재 네 서비스는 역할과 의존성이 달라 별도 실행 단위로 두는 것이 타당하다.
우선 정리할 대상은 등록·설정·기동/종료·상태 확인·배포 방식이다. 이번 점검에서는 실행 구성이나 서비스를 변경하지 않았다.

| 서비스 | 역할 | 호스트 기본 포트 | 점검 시 실행 상태 |
|---|---|---:|---|
| Extra Media | URL 영상·음성 가져오기, 변환·추출 | 8690 (컨테이너 내부 8698) | 정지 |
| Extra Collector | 브라우저를 이용한 웹 수집·다운로드 | 8695 | 정지 |
| Extra Documents | HWP/HWPX·DOCX·PPTX·XLSX·PDF 생성 | 8696 | 실행 |
| Extra SSH | 인증 키·호스트 관리, 승인된 원격 명령 실행 | 8699 | 정지 |

기능 설정은 Documents·SSH·Collector·Media import 모두 켜져 있었다. ASR은 꺼져 있었다.
설정 토글은 기능 사용 허용이고 컨테이너 시작 명령이 아니므로, 이 자체가 오류는 아니다.
다만 사용자에게 사용 설정과 실제 가용 상태를 일관되게 보여주지 못하고 있다.

## 주요 발견

### 1. 문서 서비스만 모델 세트와 통합 상태에서 별도 취급 — 우선순위 높음

`extra-documents` 정의와 health URL은 카탈로그에 있지만, 내장 모델 세트에는 Media·SSH·Collector만 포함된다.
현재 사용자 카탈로그의 세트들도 동일했다. 문서 기능은 세트에 없어도 활성화 상태를 유지한다.
반면 SSH·Collector는 세트에 없으면 기능을 끈다. 의도된 공용 서비스 정책인지 예외인지 코드만으로 일관된 규칙을 찾기 어렵다.

`/api/health`의 `extra` 응답에는 SSH·Collector만 있으며 Documents는 빠져 있다.
문서 생성 자체는 정상 동작하지만, 통합 상태 화면에서 다른 서비스와 같은 기준으로 다룰 수 없다.

근거: [카탈로그](internal/orchestrator/assets/catalog.json), [세트 적용](internal/config/config.go), [통합 상태 API](internal/server/config_handlers.go).

### 2. Media가 ASR 설정·상태에 묶여 있음 — 우선순위 높음

Media endpoint가 `asr.ffmpeg_endpoint`에 저장된다. Media import 기능은 `tools.media_import_enabled`에서 별도로 켜진다.
ASR을 끄면 ASR 클라이언트는 FFmpeg/Media 상태까지 `disabled`로 반환하며 실제 상태를 조회하지 않는다.
따라서 음성 전사를 사용하지 않으면서 URL 미디어 기능을 사용하는 경우, 미디어 상태가 잘못 해석될 수 있다.
현재도 Media import가 켜져 있지만 통합 API의 FFmpeg 상태는 `disabled`였다.

근거: [ASR health](internal/asr/client.go), [모델 세트 endpoint 적용](internal/config/config.go), [도구 등록](internal/server/tool_registry.go).

### 3. 공용 서비스의 시작·종료 범위가 분명하지 않음 — 우선순위 높음

세트 시작은 해당 세트의 구성원을 기동한다. 다른 모델을 교체할 때는 기존 LLM만 중지하는 동작이 이미 있다.
반면 세트 중지는 그 세트의 모든 구성원을 역순으로 중지한다. 세트에 들어간 Extra는 함께 멈추고,
어느 세트에도 들어 있지 않은 Documents는 남는다. 공유 서비스의 실제 사용 여부를 판단하는 정책은 별도로 없다.

권장: 모델 중지와 지원 서비스 전체 중지를 명확히 구분한다. 공용 서비스는 모델 교체 때 재사용하고,
명시적인 서비스 중지 또는 분명한 소유·사용 정책에 따라 종료한다. 기존 모델 세트의 로컬/워커 배치 선택은 보존한다.

근거: [StartBundle / runBundleStart / StopBundle](internal/orchestrator/controller.go), [공유 정의와 세트별 배치](internal/orchestrator/bindings.go).

### 4. 설치·배포 경로와 Compose 옵션이 다름 — 우선순위 중간

Documents는 앱에 빌드 자산을 포함하며 이미지가 없으면 빌드할 수 있다. 나머지 Extra 3종은 앱 실행 장비에
이미지를 먼저 만들어 놓고, 필요하면 워커로 전송하는 방식이다. 신규 장비에서의 준비 경험이 다르다.
Documents 이미지는 `0.4.0`으로 고정되지만, 나머지 3종의 앱 Compose 참조는 `latest`다.

독립 실행 Compose의 3종에는 Docker healthcheck가 있지만 앱 내장 Compose에는 빠져 있다.
SSH의 read_only/tmpfs, Media의 tmpfs 및 일부 제한 설정도 두 실행 경로에서 다르다.
앱 오케스트레이터의 HTTP 준비 확인은 별도로 있으므로 이를 ‘건강 확인이 전혀 없다’고 해석해서는 안 된다.

권장: 버전 고정, 준비·업데이트·상태 확인의 인터페이스를 통일하고, 독립/앱 실행의 옵션 차이를 제거하거나 명시한다.

근거: [이미지 준비](internal/orchestrator/remote.go), [내장 빌드 목록](internal/orchestrator/build_assets.go),
[독립 Compose](../../compose_yaml/sparktalk_extra/compose.yaml), [내장 Compose 디렉터리](internal/orchestrator/assets).

### 5. 코드·이름·패키징의 중심이 나뉘어 있음 — 우선순위 중간

`internal/extra`는 이름과 달리 SSH 클라이언트다. Collector 클라이언트는 `internal/knowledge`,
Media 연결은 `internal/asr`, Documents의 HTTP 호출은 `internal/server/document_tool.go`에 있다.
독립 문서 서비스 이름은 `docms`, 내장 이름은 `extra-documents`다.
이름만 보고 서비스 설정과 클라이언트·배포 파일을 찾기 어렵다.

문서 구현은 독립 `docms`와 내장 자산에 복사본이 있다. 현재 대응되는 **23개 파일은 모두 동일**했다.
이미 코드 불일치가 발생했다고 판단하지는 않는다. 복사가 늘어날 때를 대비해 원본 위치와 패키징·동기화 검사를 고정하는 것이 좋다.

근거: [SSH 클라이언트](internal/support/ssh/client.go), [Collector 클라이언트](internal/knowledge/collector.go),
[Media/ASR 클라이언트](internal/asr/client.go), [문서 호출](internal/server/document_tool.go),
[독립 서비스](../../compose_yaml/sparktalk_extra), [내장 문서 자산](internal/orchestrator/assets/extra-documents).

## 권장 정리 방향

1. **공통 지원 서비스 목록과 상태 계약부터 정리한다.** 네 서비스 모두 사용 허용·설치 여부·실행 여부·준비/오류 상태를 구분한다. Documents를 통합 health에 넣고 Media health는 ASR에서 분리한다.
2. **서비스의 운영 범위를 정한다.** 모델 세트의 참조/로컬·워커 배치는 유지하되, 공유 서비스의 시작·종료 규칙을 동일하게 적용한다. Documents만 예외로 남기거나 무조건 모든 세트에 넣는 식의 임시 대응은 피한다.
3. **설정 UI를 공통 ‘지원 서비스’ 관리 영역으로 모은다.** 서비스 카드에서 호스트, endpoint, 버전, 준비·시작·중지·상태를 확인한다. 기능별 문서·웹·음성·SSH 화면은 기능 설정과 해당 서비스 상태를 연결해 보여준다.
4. **서비스별 클라이언트와 배포 자산의 원본을 명확히 한다.** 미디어/음성 인식, 웹 수집/지식 저장, 문서 생성/문서 읽기의 경계는 유지한다. 각 기능마다 새 Extra 컨테이너를 추가하는 방식으로 확장하지 않는다.
5. **버전·패키징·검증 규칙을 통일한다.** 내장 자산 생성 및 독립 구성과의 비교, 공통 health contract와 lifecycle 테스트를 둔다.

현재 공통 컴포넌트 정의와 세트별 배치 override는 이미 구현되어 있어 재사용할 수 있다.
전체를 새로 만드는 작업보다, 이 구조에 공통 서비스 상태와 수명 정책을 보완하는 단계적 정리가 적합하다.

## 적용 시 확인할 시나리오

- ASR을 꺼도 Media import의 실제 서비스 상태가 표시됨
- 문서 기능을 켰지만 서버가 없는 경우 준비/시작 방법과 상태가 분명히 표시됨
- LLM을 교체해도 공용 지원 서비스가 불필요하게 재시작되지 않음
- 명시적인 세트/서비스 중지 범위가 UI 설명과 일치함
- 로컬에서 워커로 배치를 바꿀 때 endpoint·볼륨·키 저장 위치가 일관됨
- 독립 실행과 앱 실행의 healthcheck·버전·자원 제한 정책이 일치함

이 문서는 현재 구조와 실제 실행 상태를 읽어서 작성한 점검 결과다. 서비스 통합·이동·설정 변경은 실행하지 않았다.
