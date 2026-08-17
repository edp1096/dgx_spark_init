# SparkTalk

Go 백엔드와 Svelte 프런트엔드로 만든 간단한 로컬 LLM 채팅 앱입니다. OpenAI 호환 API(SGLang·vLLM 등)에 연결하며 채팅 기록은 SQLite에 저장합니다.

## 개발 실행

```bash
# 터미널 1
go run ./cmd/chat

# 터미널 2 (Node.js 20.19+ 필요)
cd web
npm install
npm run dev
```

브라우저에서 `http://127.0.0.1:5173`에 접속합니다.

## 배포 빌드

```bash
make build   # Svelte + Go 컴파일 확인
make dist    # Linux/Windows 배포 바이너리 생성
```

`dist/`에는 Linux와 Windows의 amd64·arm64 바이너리가 생성됩니다.
DGX Spark에서는 `sparktalk-linux-arm64`, 일반적인 Windows PC에서는
`sparktalk-windows-amd64.exe`를 실행합니다. `make clean`은 생성된
바이너리와 프런트엔드 빌드 파일을 정리합니다. Windows PowerShell에서는
GNUWin32의 GNU Make를 사용할 수 있습니다.

Svelte UI가 Go 바이너리에 포함되므로 배포할 때 `dist/`의 운영체제별
바이너리 하나만 복사하면 됩니다.

## 설정

실행 옵션은 사용하지 않습니다. 처음 실행하면 현재 디렉터리에 내장된
샘플을 바탕으로 `sparktalk.yaml`을 만들고 곧바로 서버를 시작합니다.

```bash
./dist/sparktalk-linux-arm64
```

API endpoint, listen address, SQLite 경로, 기본 모델과 reasoning effort는
`sparktalk.yaml` 또는 웹 화면 좌측 하단의 **설정**에서 관리합니다.
Endpoint·모델·reasoning 변경은 저장 즉시 적용되며 listen address와 DB
경로 변경은 앱 재시작 후 적용됩니다.

## 웹 도구

별도 검색 서버 없이 내장 `web_search`와 `web_fetch`를 사용할 수 있습니다.
채팅 상단의 **웹 자동**을 켜면 모델이 필요한 경우 DuckDuckGo 검색 또는
공개 웹 페이지 읽기를 선택해서 실행합니다. 실행 기록은 답변의 **웹 도구**
영역에서 접고 펼칠 수 있습니다. 검색 결과는 참고 자료로만 모델에 전달되며,
로컬·사설망 주소는 차단됩니다. 사용 여부, 최대 호출 횟수, 검색 결과 수와
타임아웃은 설정 화면 또는 YAML의 `tools` 항목에서 조정합니다.

## 이미지 인식

입력창의 `＋` 버튼, 드래그 앤 드롭 또는 클립보드 붙여넣기로 PNG·JPEG·WebP
이미지를 한 메시지에 최대 6개까지 첨부할 수 있습니다. 이미지당 최대 크기는
15MB이며, 비전 입력을 지원하는 OpenAI 호환 모델이 필요합니다. 첨부 이미지는
SQLite 파일 옆의 `<database>.media` 디렉터리에 저장되고 질문 수정·답변 재시도·
버전 전환에서도 함께 보존됩니다.

기본 listen address는 `0.0.0.0:8585`입니다. 같은 네트워크의 다른 PC에서는
`http://서버-IP:8585`로 접속합니다.
