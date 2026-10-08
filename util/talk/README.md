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

## 모델과 설정

1. **설정 → 시스템 → 모델 준비**에서 사용할 모델을 선택하고 **전체 준비**를 실행합니다.
2. 준비가 끝나면 상단 모델 메뉴에서 세트를 선택해 시작합니다.
3. Extra Media·Collector·Documents·SSH는 **설정 → 시스템 → 지원 서비스**에서 준비합니다.

모델 준비는 필요한 이미지 빌드·가중치 다운로드·변환을 수행하고, 준비된 파일은 재사용합니다.
접근 제한 모델은 Hugging Face 이용 조건에 동의한 뒤 모델 준비 화면에 토큰을 등록합니다.
외부 모델 서버를 연결하거나 세트·실행 호스트를 바꾸려면 [AI 세트 설정](examples/README.md)을 참고하세요.

- **라이브러리**: 기억·문서 지식·스킬·작업 절차를 관리합니다.
- **도구 메뉴**: 웹검색·음성대기·대화별 SSH 허용을 설정합니다.
- **설정 → 음성 → 답변 음성**: 언어·화자·자동 읽기·속도를 설정합니다. 속도는 기본 `1.0`이며 `1.2~1.3`으로 올릴 수 있습니다.

마이크는 HTTPS 또는 localhost에서 사용할 수 있습니다. HTTP 접속 설정은 앱의 **마이크 사용 방법**을 참고하세요.

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
