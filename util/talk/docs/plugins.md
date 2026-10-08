# Talk 플러그인

라이브러리 → 플러그인에서 ZIP 패키지를 설치·설정·활성화한다.
Talk 재빌드 없이 외부 실행파일을 붙일 수 있다. 업무 플러그인은 아직 포함하지 않는다.
내장 Go 확장용 `Builtins()`와 외부 패키지는 설정·권한·작업 API를 공유한다.

## 외부 패키지

ZIP 최상단에 `plugin.json`과 정적 실행파일 `plugin` 두 파일을 넣는다.
추가 파일·심볼릭 링크·중복 이름·다른 경로·동적 ELF 실행파일은 거부한다.
패키지/실행파일은 각각 최대 128MiB, 매니페스트는 256KiB다.
`plugin.json` 예제는 `pluginsdk/example/plugin.json`에 있다.
`platform`은 `linux/arm64` 또는 `linux/amd64`이며 서버 아키텍처와 일치해야 한다.

- 설치: 격리 환경에서 ID·버전·통신 규격과 Start를 확인한 후 비활성 상태로 등록한다.
- 활성화: 설정과 요청 권한을 확인하고 활성화한다. 설치 검사는 업무 실행을 허용하지 않는다.
- 업데이트: 먼저 비활성화하고 같은 ID의 새 버전 패키지를 설치한다. 권한은 다시 허용한다.
- 데이터 버전 변경: SDK의 Migrate에 사본을 전달한다. 패키지·설정·데이터를 하나의 DB 트랜잭션으로 교체한다.
- 실패: 기존 패키지·설정·데이터를 유지한다. 외부 서비스 호출은 설치·마이그레이션 중 금지한다.
- 복구: 이전 버전과 **업데이트 직전 데이터**로 되돌린다. 업데이트 이후 데이터 변경은 없어지므로 화면에서 확인한다.
- 제거: 데이터 보존 또는 데이터·실행 기록 삭제를 선택한다. 실행 중인 패키지는 제거하지 않는다.

설치 파일은 DB 경로에 `.plugins`를 붙인 디렉터리에 저장한다. 현재 버전과 직전 복구 버전을 보존한다.
파일을 먼저 기록·동기화하고 DB를 교체한다. 정전으로 남은 미참조 디렉터리는 실행되지 않는다.
실행 전 SHA-256을 확인하며 해시는 무결성 확인용이다. 제작자 서명/신원 인증이나 마켓플레이스는 제공하지 않는다.

## SDK로 제작

`pluginsdk`는 표준 라이브러리만 사용하는 공개 Go 패키지다. 개발 시 Talk 소스를 SDK 참조용으로 사용한다.
다른 디렉터리의 프로젝트에서 다음과 같이 참조할 수 있다. 설치받는 사용자는 Go나 Talk 소스가 필요 없다.

```sh
go mod init example.com/my-talk-plugin
go mod edit -require=sparktalk@v0.0.0
go mod edit -replace=sparktalk=/절대경로/dgx_spark_init/util/talk
# main.go는 pluginsdk/example/main.go를 참고한다.
CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -o plugin .
zip example.zip plugin.json plugin
```

SDK의 `Plugin`에 ID, Version, Start, Handle, Stop, Migrate를 구현한다.
Start에서 시작한 지속 작업은 `Client.Host(ctx, "ready", {}, &result)`가 성공한 다음 실행한다.
Start의 Context는 플러그인 수명에 해당한다. Stop은 정리용이며 취소에 응답해야 한다.
표준출력은 통신 전용이다. 진단 로그는 stderr로 보내며 서버는 앞 8KiB만 보관한다.
내장 예제를 빌드할 때는 Talk 디렉터리에서 `CGO_ENABLED=0 go build -o plugin ./pluginsdk/example`을 실행한다.

## 통신과 격리

통신은 표준입출력의 줄 단위 JSON-RPC 2.0이다. 프로토콜 버전은 1, 요청 ID는 문자열,
메시지는 최대 2MiB, 수신 중인 요청은 최대 16개다. 초과하거나 잘못된 메시지는 연결을 종료한다.
`hello`, `start`, `handle`, `stop`, `migrate`와 `$/cancel`을 사용한다.
호스트 호출은 `host.ready/get/put/delete/call/submit/runs/cancel`이며 각 호출에서 권한을 검사한다.
`call` 입력은 `{service,request}`, 저장소는 `{key,value}`, 작업 제출은 `{operation,key,request}`다.
Talk는 자기 서비스 접속 키나 호스트 환경변수를 실행파일에 전달하지 않는다.

외부 실행은 **Linux arm64/amd64, Landlock ABI 3 이상, seccomp**가 필요하다.
시작 전에 제한을 적용하고 실행파일로 exec한다. 제한 설정 실패 시 실행을 거부한다.
플러그인은 자기 패키지 파일만 읽고 실행할 수 있다. 직접 파일 쓰기·소켓 생성·자식 프로세스 생성은 차단한다.
저장·네트워크·모델·도구 사용은 허용한 Host API를 통해 수행한다.
가상 주소 공간은 2GiB, 파일 디스크립터는 64개로 제한하며 코어 덤프를 막는다.
Go의 GOMEMLIMIT 128MiB는 GC 목표이며 별도의 물리 메모리 보장량이 아니다.

정상 비활성화 시 Stop에 최대 500ms를 주고 남은 프로세스를 종료한다.
작업 취소·시간 초과에는 실행 프로세스를 종료하여 뒤에서 계속 실행되는 것을 막는다.
충돌·프로토콜 위반은 실패 상태로 표시하며 자동 재실행하지 않는다. 다시 비활성화·활성화하여 재시작한다.
권한 있는 호스트 도구가 이미 수행한 외부 동작까지 되돌리는 것은 아니다.


## 내장 Go 확장과 공통 생명주기

`internal/plugins.Definition`을 만들고 `Builtins()`에 등록한다.
새 플러그인은 기본 비활성이다. 라이브러리 → 플러그인에서 설정과 요청 권한을 저장한 후 활성화한다.

- `Manifest`: ID, 표시 이름, `x.y.z` 버전, `api_version=1`, 데이터 버전, 요청 권한, 작업, 화면 선언.
- ID·작업 이름: 영문 소문자로 시작하는 최대 24자의 소문자·숫자·밑줄.
  중복 ID, 생성되는 도구 이름의 충돌, 호환되지 않는 API 버전은 등록 오류다.
- `ValidateConfig`: JSON 객체 설정 검증. 설정·권한은 비활성 상태에서만 바꾼다.
- `Start(ctx, host)`: 초기화. 10초 시작 제한의 취소 신호를 준수해야 한다.
  지속 작업자는 `host.Ready()` 또는 `ctx.Done()`을 기다린 뒤 동작한다.
  `Ready`는 활성 상태의 DB 저장까지 성공한 뒤에만 닫힌다.
- `Handle(ctx, host, operation, request)`: 선언된 작업 실행. 입력은 JSON 객체다.
  입력 필드의 상세 검증은 구현이 담당한다. 반환값은 유효한 JSON이어야 한다.
- `Stop(ctx)`: 소유한 작업자·자원 정리. 비활성화는 실행 취소와 완료 대기 후 Stop을 호출한다.
- 종료 시 활성화 설정은 보존하고, 다음 기동에서 Start를 다시 호출한다.
  이전 프로세스에서 끝나지 않은 실행 기록은 `interrupted`로 바꾸며 자동 재실행하지 않는다.

상태는 `disabled → starting → active → stopping → disabled`이며,
시작·종료 오류는 `failed`와 오류 내용으로 표시한다. 실행 함수의 오류는 해당 실행에 기록한다.
취소에 응답하지 않아 종료 대기가 만료되면 `stopping` 상태를 유지하며 재활성화를 차단한다.
작업 종료 후 비활성화를 다시 요청해 정리를 마칠 수 있다.

## Host 권한과 연동

플러그인에는 DB·HTTP 서버 객체 대신 ID에 묶인 Host를 전달한다.
요청 권한은 모두 사용자가 허용해야 활성화할 수 있으며, 버전 변경으로 제거된 권한은 저장된 허용 목록에서도 제거한다.

| 권한 | 제공 기능 |
| --- | --- |
| `storage` | 자기 플러그인의 JSON 키·값 읽기/쓰기/삭제 |
| `tools` | 선언한 작업을 대화 도구로 공개 |
| `jobs` | 자기 작업의 백그라운드 제출·목록·취소 |
| `model.complete` | Talk에 설정된 모델 클라이언트로 텍스트 생성. 도구는 제공하지 않음 |
| `tool.call` | 기존 대화 ID에서 현재 사용 가능한 Talk 도구 호출. 기존 도구의 권한 검사를 유지 |

모델 호출은 Talk의 생성 추적에 등록되므로 비상 큐 취소 대상이다.
`tool.call`은 다른 플러그인 도구의 재귀 호출을 거부한다.
추가 사용자 승인이 필요한 도구는 백그라운드에서 승인받은 것으로 간주하지 않고 오류를 반환한다.
모델 자동 기동·교체는 하지 않는다. 서비스가 내려가 있으면 호출이 실패한다.
Host 서비스는 코어의 `pluginServices()`에서 추가하며, 새 권한은 다시 허용해야 한다.

이 권한 검사는 **Host API 경계**다. 같은 프로세스의 Go 코드에서 OS 접근을 차단하는 보안 샌드박스는 아니다.
이 단락은 내장 Go 확장에 해당한다. 외부 패키지는 위의 프로세스 격리 규격을 사용한다.
플러그인은 Context 취소를 준수하고 Stop에서 자체 goroutine을 정리해야 한다.
코어는 호출한 콜백의 panic을 회수하지만 플러그인이 직접 만든 goroutine의 panic이나 무한 루프를 강제 격리하지 못한다.

## 작업과 실행 기록

작업에는 이름, 설명, 입력 JSON Schema(`type: object`), 도구 공개 여부,
백그라운드 실행 허용 여부, 1~3,600초의 제한 시간을 선언한다.

- 플러그인당 한 작업만 실행한다. 다른 작업이 실행 중이면 `busy`를 반환하며 무제한 큐를 만들지 않는다.
- 백그라운드 작업은 HTTP 연결 종료와 무관하게 계속 실행한다.
- 대화에 공개된 백그라운드 작업은 실행 ID를 즉시 반환한다. 대화 ID·도구 호출 ID로 재시도 키를 만든다.
- 명시적인 재시도 키는 같은 플러그인의 같은 작업·입력·대화에만 재사용할 수 있다.
  실행 중·완료·중단 기록이 있으면 재실행하지 않고 기존 기록을 반환한다. 입력 JSON은 저장된 바이트와 비교한다.
- 동기·백그라운드 실행 모두 시작 전 기록하고, 결과·실패·취소·시간 초과를 저장한다.
  결과 저장 실패는 플러그인 오류에 표시하며 외부 동작을 자동 재시도하지 않는다.
- 최근 실행 목록은 플러그인별 100건을 반환한다. 기록은 자동 삭제하지 않아 재시도 키가 유지된다.
- 비활성화 후 이전 Host와 이전 대화 도구 목록을 통한 실행은 거부한다.

실행 입력·결과는 각각 최대 256KiB다. 설정도 최대 256KiB의 JSON 객체다.
전용 저장 공간은 키당 최대 256KiB, 최대 128개 키·합계 1MiB다.
모델 접속 자격증명은 기존 Talk 설정에 두고 플러그인 설정에 복제하지 않는다.
동일 DB를 여러 Talk 프로세스가 동시에 소유하는 구성은 지원하지 않는다.

## 저장 형식 변경

`data_version`이 다르면 활성화를 차단한다. 비활성화 후 데이터 업데이트를 요청하면
`Migrate(ctx, from, to, values)`에 분리된 키·값 사본을 전달한다.
성공했을 때만 전체 키·값과 데이터 버전을 한 SQL 트랜잭션으로 반영한다.
실패·panic·취소·용량 초과 시 기존 데이터를 유지한다. 다운그레이드는 거부한다.
콜백 안에서 Host/DB 접근이나 외부 동작을 수행하지 않는다.

## UI와 HTTP

`Panels`는 제목·설명·선언된 작업 이름만 제공한다. 코어 화면이 설정, 권한,
작업 입력, 실행 버튼과 결과를 렌더링한다. 임의 HTML/JavaScript를 삽입하지 않는다.

- `POST /api/plugins/install`: `application/zip` 패키지 본문
- `POST /api/plugins/{id}/rollback`: `{}`
- `POST /api/plugins/{id}/remove`: `{ "purge": false }`
- `GET /api/plugins`: 등록 목록과 설정·상태
- `PUT /api/plugins/{id}/configure`: `{ "config": {}, "grants": [] }`
- `POST /api/plugins/{id}/enable|disable|migrate`: `{}`
- `POST /api/plugins/{id}/call|submit`: `{ "operation": "name", "key": "optional", "request": { "session_id": "optional", "input": {} } }`
- `GET /api/plugins/{id}/runs`: 최근 실행
- `POST /api/plugins/{id}/cancel`: `{ "run_id": "..." }`

패키지 업로드를 제외한 변경 요청은 `application/json`을 사용한다. 알 수 없는 필드, 본문 크기 초과,
브라우저의 다른 Origin 요청을 거부한다. 기존 Talk 접근 권한 범위에서 사용하는 관리 API다.

## 검증

```sh
go test -race ./pluginsdk ./internal/plugins ./internal/db ./internal/server
cd web && npm run build && npx playwright test e2e/plugins.spec.js
```

권한 철회, 저장 공간 분리, panic, 취소·시간 초과, 실행 중/완료 후 중복 방지,
DB 재개방, 재시작 복구, 마이그레이션 롤백, 도구 등록 해제, 모델 서비스 연결,
관리 화면의 상태 갱신을 테스트한다. 테스트에서 실제 모델이나 업무 도구를 실행하지 않는다.

외부 실행 테스트는 별도 Go 모듈에서 SDK를 참조해 패키지를 빌드한다. 실제 Landlock/seccomp 환경에서
파일·네트워크·자식 프로세스 차단, 환경변수 제거, 강제 취소, 충돌, 과도한 출력, 설치/업데이트/복구/제거,
마이그레이션·DB 커밋 실패, 체크섬 변경을 검증한다. 웹 테스트에서도 실제 패키지를 업로드·실행·제거한다.
