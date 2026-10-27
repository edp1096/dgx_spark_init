# SparkTalk

로컬 AI 채팅 앱입니다. OpenAI 호환 모델에 연결하며 대화 기록은 SQLite에 저장합니다.
연결한 모델과 서비스에 따라 이미지·영상 생성, 음성 인식·답변 읽기, 문서 검색, 웹검색, SSH 작업을 사용할 수 있습니다.

## 실행

배포 바이너리 하나를 쓰기 가능한 폴더에 복사해 실행합니다.
DGX Spark는 `sparktalk-linux-arm64`, Windows PC는 `sparktalk-windows-amd64.exe`를 사용합니다.
배포본 실행에는 Go와 Node.js가 필요하지 않습니다.

```bash
chmod +x sparktalk-linux-arm64
./sparktalk-linux-arm64
```

브라우저에서 `http://서버주소:8585`에 접속합니다.
처음 실행하면 현재 폴더에 `sparktalk.yaml`과 `sparktalk.db`를 만듭니다.
첨부 파일과 지식 원문은 `sparktalk.db.media/`·`sparktalk.db.knowledge/`에 저장합니다. 백업할 때 함께 보관하세요.

DGX Spark에서 모델을 직접 실행하려면 NVIDIA 드라이버·Container Toolkit과 Docker Engine·Compose가 필요합니다.
실행 계정에 Docker 권한이 있어야 합니다. 두 Spark를 쓰는 구성은 양쪽의 SSH 연결과 Python 3·rsync도 준비합니다.

For a two-host reinstall, use the [backup and recovery guide](deploy/recovery/README.md).
Private keys and state backups remain separate from the executable and source.

## 모델과 설정

다른 세트로 전환하면 탑재된 ASR·TTS·이미지·임베딩과 허용된 공유 Extra 기능의 기본 스위치를 켭니다. 없는 기능은 꺼지며, 같은 세트에서 직접 바꾼 스위치는 저장·재시작 시 유지됩니다.

`모델 세트 중지`는 모델과 Talk가 관리하는 공유 Extra 서비스도 함께 종료합니다. 세트가 중지된 동안 Shorts 상태 조회로 작업 서비스가 자동 재기동되지 않습니다.

FP4 Local과 TP2 세트(Flash Next·GLM·DeepSeek)는 Fish 음성 합성 및 ACE-Step 음악 생성을 제외합니다. `disabled_support` 정책은 공유 서비스 노출, 도구 등록, 직접 생성 요청과 세트 기동에 함께 적용되며, 기존 저장 설정에도 반영됩니다. FP4의 1M 문맥·KV·프롬프트 캐시 설정은 유지합니다.

1. **설정 → 시스템 → 모델 준비**에서 사용할 모델을 선택하고 **전체 준비**를 실행합니다.
2. 준비가 끝나면 상단 모델 메뉴에서 세트를 선택해 시작합니다.
3. Extra Media·Collector·Documents·SSH는 **설정 → 시스템 → 지원 서비스**에서 준비합니다.

모델 준비는 필요한 이미지 빌드·가중치 다운로드·변환을 수행하고, 준비된 파일은 재사용합니다.
접근 제한 모델은 Hugging Face 이용 조건에 동의한 뒤 모델 준비 화면에 토큰을 등록합니다.
외부 모델 서버를 연결하거나 세트·실행 호스트를 바꾸려면 [AI 세트 설정](examples/README.md)을 참고하세요.

`Qwen 3.8 Flash-Next FP4-Local`은 로컬에서 제작한 Huihui v2 체크포인트를 사용합니다.
[Compose 구성과 준비 조건](../../compose_yaml/qwen38_fn_fp4_local_tsfold/README.md)을 참고하세요.

The Qwen FP4-Local and EXL3 3bit TensorFold deployments use Talk-owned embedded Compose files and build contexts. The recipes under `compose_yaml` are independent standalone deployments; Talk neither copies them during packaging nor reads them at runtime.

- **라이브러리**: 기억·문서 지식·스킬·작업 절차를 관리합니다.
- **도구 메뉴**: 웹검색·음성대기·대화별 SSH 허용을 설정합니다.
- **설정 → 음성 → 답변 음성**: 언어·화자·자동 읽기·속도를 설정합니다. 속도는 기본 `1.0`이며 `1.2~1.3`으로 올릴 수 있습니다.

마이크는 HTTPS 또는 localhost에서 사용할 수 있습니다. HTTP 접속 설정은 앱의 **마이크 사용 방법**을 참고하세요.

## EPUB 문서 번역

`.epub` 파일을 첨부하고 전체 번역을 요청할 수 있습니다. 첨부 미리보기·지식 가져오기·전체 문서 번역은 같은 EPUB 파서를 사용하며, EPUB 2/3의 OPF 읽기 순서에 따라 XHTML/HTML 본문과 각주·부록을 읽습니다. 지원하지 않는 본문 형식은 manifest에 지정된 XHTML/SVG 대체 본문을 사용합니다.

제목·저자·언어, 문단·표·링크·일반 이미지를 유지합니다. SVG 본문과 삽화는 텍스트와 포함된 이미지를 번역 원문으로 읽으며, 벡터 도형과 원래 배치는 유지하지 않습니다. 암호화된 본문은 오류로 알리며, 글꼴만 난독화된 책은 본문을 읽을 수 있습니다. 이미지에만 들어 있는 글자는 OCR하지 않습니다.

원본 32MiB·압축 해제 총량 128MiB·본문 400만 자 제한이 적용됩니다. 결과는 기존 TXT/Markdown/HTML/ZIP 및 문서 변환 기능으로 받으며, EPUB 파일로 내보내기는 지원하지 않습니다.

## 외부 툴 공통 API

설정 → 시스템 → 시작·연결의 **외부 툴 연결**에서 공통 API 주소를 복사합니다.
Base URL은 `http://서버주소:8585/v1`, 고정 모델 ID는 `sparktalk`입니다.
모델 세트를 전환하면 새 요청은 새 활성 모델로 전달됩니다. 진행 중인 요청은 기존 대상에 유지되며, 모델 서버를 중지하면 해당 요청이 종료될 수 있습니다.

`GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/responses`를 제공합니다.
생성 요청의 도구 정의·멀티모달 입력·스트리밍과 서버 오류는 그대로 전달하며, Responses API를 포함한 기능 지원 여부는 모델 서버에 따릅니다. Talk의 기억·내장 도구 실행 루프는 사용하지 않습니다.
공통 API에 별도 키는 필요하지 않으며, 클라이언트가 키 입력을 요구하면 임의 값을 사용합니다. 업스트림 인증에는 Talk에 저장된 모델 API 키를 사용합니다.
`GET /api/model-api`는 공통 경로와 현재 연결 대상 모델 ID를 반환합니다.

## 개발·빌드

`util/talk`에서 실행합니다. Go 1.25+와 Node.js 20.19.x 또는 22.12+가 필요합니다.

```bash
make build   # 프런트엔드·백엔드 빌드
make dist    # dist/에 Linux·Windows, amd64·arm64 배포본 생성
make test    # 테스트
```

개발 서버는 두 터미널에서 실행합니다.

```bash
go run ./cmd/chat
```

```bash
cd web
npm install
npm run dev
```

개발 화면은 `http://127.0.0.1:5173`입니다.
DGX Spark 자동기동용 [systemd 서비스](deploy/systemd/sparktalk.service)는 실제 설치 경로에 맞춰 사용합니다.

GLM 5.3 Flash FP4-Local uses the independently packaged `recipe_sources/glm53-fp4` runtime: two Sparks, vLLM/B12X, FP8 KV, 1M context, MTP 5 and Huihui abliterated weights by default. MTP 1 is selectable for Korean decode. Talk never reads `compose_yaml` at runtime. The standalone recipe and comparison evidence are in [glm53f_fp4_local_vllm](../../compose_yaml/glm53f_fp4_local_vllm/README.md).

Source: the active recipe's pinned model/runtime references and local qualification records.
