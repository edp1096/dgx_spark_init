# 편집 가능한 AI 세트

설정 → 시스템 → AI 세트 편집에서 기본 세트를 복제하거나 이름·구성 서비스·모델
프로필을 수정한다. 세트별로 각 서비스의 실행 호스트, API 주소, 상태 확인 URL, 공개 포트를
편집할 수 있다. 서비스 정의는 공유하지만 배치 변경은 편집 중인 세트에만 적용된다. 화면 하단의 **저장**을 누르기 전에는 실제 설정에 반영되지 않는다.
저장만으로 컨테이너를 시작하지 않는다. 시작·중지는 운영 패널에서 실행한다.

전체 정의는 `sparktalk.yaml`의 `runtime.catalog`에 저장한다. 기본 카탈로그는
최초 설정에만 사용하며 이후 저장한 주소를 localhost로 덮어쓰지 않는다.
`hosts`, `components`, `bundles`로 구성된 JSON 또는 YAML을 가져올 수 있다.
가져오기는 전체 카탈로그를 대체하므로 기존 세트도 유지하려면 먼저 JSON으로
내보내고 그 파일에 새 항목을 추가한다. 기본 세트를 삭제할 때는 기본 AI 세트도
남아 있는 세트로 바꾼다. 사용 중인 세트/서비스는 중지한 뒤 편집한다.

## 공통 서비스와 세트별 배치

`components`에는 Extra Media·Collector·SSH를 각각 한 번만 정의한다.
`bundles[].components`는 그 ID를 참조하고, `bundles[].bindings`에 세트별 차이만 저장한다.
예를 들어 같은 `extra-collector`를 Qwen 세트는 로컬, GLM 세트는 워커에 배치한다.

```yaml
bundles:
  - id: glm53-worker-extra
    name: GLM 5.3 Flash EXL3
    model_type: glm5.3
    model_id: glm-5.3-flash
    context_tokens: 524288
    components: [glm53, extra-media, extra-ssh, extra-collector]
    bindings:
      extra-collector:
        host: worker
        endpoint: http://192.168.100.60:8695
        health_url: http://192.168.100.60:8695/health
        bind_address: 192.168.100.60
        port: 8695
```

위 조각은 배치 형식 예시이며 전체 가져오기 파일은 옆 JSON 예제를 사용한다.
생략한 값은 공통 정의의 기본값을 사용한다. `0` 또는 빈 문자열은 명시적인 값으로
처리한다. 화면의 **기본 배치로 되돌리기**는 해당 세트의 서비스 배치만 초기화한다.
제어 방식, 컨테이너 이름, 메모리 예상치, 시작 제한시간 등도 세트별로 지정할 수 있다.
공통 역할·Compose 레시피 편집은 동일 정의를 쓰는 모든 세트에 반영된다.

이전 버전이 만든 `worker-extra-media`, `worker-extra-ssh`, `worker-extra-collector`는
설정을 읽거나 가져올 때 공통 ID와 세트별 배치로 자동 변환한다. 수정해 둔 주소·포트는
보존하며 다음 설정 저장 시 새 형식으로 기록한다. 별도 커스텀 레시피로 바꾼 정의는
임의로 합치지 않는다. 실행 중인 컨테이너·이미지·SSH 키를 이동하거나 삭제하지 않는다.

## 서비스 정의

- `host`: `hosts`에 등록한 실행 호스트 ID. 주소가 비어 있으면 앱 실행 호스트다.
- `controller`: `compose`, `glm53-cluster`, `external`(연결 전용).
- `endpoint`: SparkTalk 백엔드가 접속할 API base URL. 모델 API에는 `/v1`을 붙이지 않는다.
- `health_url`: 실제 상태 확인 URL. API 연결 시험도 이 주소를 사용한다.
- `bind_address`, `port`: Compose가 공개할 호스트 주소·포트. 빈 값/0이면 레시피
  기본값이다. API 주소는 프록시·SSH 터널일 수 있으므로 이 값과 별도로 관리한다.
  host network를 쓰는 FLUX는 `port`만 적용하고 바인딩은 해당 런타임을 따른다.
- `compose_asset`: 내장된 `compose.*.yaml` 레시피 이름.
- `container`: 실행 호스트 내 컨테이너 이름. 여러 세트가 같은 서비스를 쓸 때는
  같은 컴포넌트 ID를 참조한다. 주소가 다를 때는 서비스를 복제하지 않고 세트의
  `bindings`를 수정한다. 실행 대상은 세트 배치를 적용한 호스트·컨테이너로 식별한다.
- `memory_gib`: 해당 실행 호스트의 예상 추가 사용량. 실측에 맞게 조정한다.

ASR, TTS, 이미지 생성, Media, Collector, SSH도 모델과 함께 세트에서 주소를
결정한다. 포함하지 않은 ASR·TTS·이미지·SSH·Collector 기능은 관리형 모드에서
비활성화한다. 포함한 기능의 사용 여부는 각 기능 설정을 유지한다.
연결 전용 서비스는 API 상태만 확인하며 Docker 시작·중지와 메모리 계산에서 제외한다.

Compose 설정은 실행 호스트의 `data_dir/runtime/서비스ID/compose.yaml`에
실제 경로로 남긴다. 중지된 기존 컨테이너를 새 설정으로 시작하면 컨테이너를
재생성한다. 이미지와 볼륨을 삭제하지 않지만 컨테이너 쓰기 계층은 교체되므로
보존할 데이터는 볼륨에 둔다. 이미지·모델 설치는 미리 각 호스트에서 준비한다.

## GLM + 워커 Extra

기본 카탈로그에 `glm53-worker-extra`가 포함되어 있다.
[glm53-worker-extra.json](glm53-worker-extra.json)은 같은 구성을 단독으로 가져오는 예제다.

- GLM Head: SparkTalk 실행 호스트, API `127.0.0.1:8000`
- GLM Worker 및 Extra 3종: `192.168.100.60`
- 워커 Extra: `8690`(Media), `8695`(Collector), `8699`(SSH)
- 기본 메모리 예상치 Head 110GiB, Worker 108GiB는 **미검증 계획값**이다.
  운영 패널의 호스트별 여유와 실제 동시 작업 최대값을 확인한 뒤 조정한다.
- 워커 최소 확보 메모리 기본값은 2GiB다. GLM 로딩 및 브라우저·FFmpeg 작업에
  필요한 여유를 보장하는 값은 아니며, 실제 워크로드로 확인해야 한다.

GLM·DeepSeek는 앱에 포함된 실행 패키지를 앱 데이터 폴더에서 실행한다.
`manage_path`는 과거 설정을 읽기 위한 호환 필드이며 실행에는 사용하지 않는다.
워커 주소·API 포트는 서비스 정의, 통신망 IP·인터페이스·HCA와 모델 종류는
`runtime_options`로 지정한다. SSH 연결은 앱 실행 계정의 설정을 사용한다.
개발 저장소나 외부 `.env`를 준비할 필요가 없다.

중지는 Head와 Worker 양쪽에 직접 시도한다. 한쪽이 꺼져 있어도 나머지 노드를
중지하고 실패한 호스트를 표시한다. 다시 켠 뒤 중지 작업을 재시도할 수 있다.

원격 제어는 **SparkTalk 실행 계정의 OpenSSH**를 사용한다. 공개키 인증과
known_hosts 등록이 선행되어야 하며, 자동으로 호스트 키를 신뢰하지 않는다.
호스트의 `identity_file`은 앱 호스트의 키 파일 경로다. GLM Head가 별도 원격
호스트라면 Head → Worker 인증도 별도로 준비한다.

Extra SSH가 보관하는 도구 실행용 키는 위 관리용 SSH 인증과 별개이며, 워커의
`data_dir/extra/ssh/keys` 및 `state`에 준비한다. 워커 LAN API 예제를 사용할 때는
헤드에서만 접근할 수 있도록 네트워크 접근을 제한한다. 기존 loopback 바인딩을
유지하려면 SSH 터널을 별도로 구성하고 `endpoint`, `health_url`을 터널 주소로 바꾼다.
