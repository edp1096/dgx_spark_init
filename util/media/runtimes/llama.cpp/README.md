# llama.cpp for DGX Spark

DGX Spark(GB10, CUDA `sm_121a`)용 llama.cpp를 빌드하고, 컴파일된 런타임을
호스트에 보관해 OpenAI 호환 API로 실행한다. 현재 SparkTalk과 SparkMedia의
공용 텍스트·멀티모달 엔진이며 기존 LiteRT-LM 서비스의 `8696` 포트를 대체한다.

현재 기본 구성:

- llama.cpp: `c060ca974c773c7c3d17fd1b66dc9d312bc292c0` (Gemma 4 12B assistant 검증본)
- API: `http://127.0.0.1:8696/v1`
- 모델 ID: `huihui-gemma4-12b`
- 모델: Huihui Gemma 4 12B abliterated i1 `Q4_K_M`
- 컨텍스트: `65,536`
- MTP speculative decoding: 사용
- 멀티모달 `mmproj`: 사용
- GPU 레이어 전체 적재 및 flash attention: 사용

공식 Q8_0 assistant의 MTP 제안 수는 실측상 속도와 출력 안정성의 균형이 가장
좋았던 3을 사용한다(`LLAMA_MTP_TOKENS=3`). `llama-server.12b.env.example`은
전환 당시 검증 설정을 보존한 사본이다.

## 모델 파일

기본 모델 경로는 다음과 같다.

```text
/home/edp1096/.cache/gguf/huihui-gemma4-12b-i1/
├── Huihui-gemma-4-12B-it-abliterated.i1-Q4_K_M.gguf
├── gemma-4-12B-it-assistant-Q8_0.gguf
└── mmproj-model-q8_0.gguf
```

경로나 파일명이 다르면 `llama-server.env.example`을 `llama-server.env`로 복사한
뒤 값을 변경한다. `llama-server.env`는 저장소에 커밋하지 않는다.

## 호스트 런타임 빌드

Docker는 재현 가능한 컴파일 환경으로만 사용하고 결과 바이너리와 공유 라이브러리는
호스트 `artifacts/<ref>/`에 내보낸다. `artifacts/current`가 현재 런타임을 가리키며
산출물은 Git에 포함하지 않는다.

```bash
./scripts/build_host.sh
```

기존 버전의 산출물이 이미 있으면 덮어쓰지 않고 중단한다. 새 llama.cpp 버전을
빌드할 때는 다음처럼 ref를 지정한다.

```bash
LLAMA_CPP_REF=<tag-or-commit> ./scripts/build_host.sh
```

호스트 산출물과 Docker 이미지를 함께 만들려면 `./build.sh`를 사용한다. 레지스트리
push가 필요할 때만 `LLAMA_PUSH=true`와 `LLAMA_IMAGE_REPO`를 지정한다.

직접 실행:

```bash
./scripts/run_host.sh
```

추가 llama-server 인자는 그대로 뒤에 붙일 수 있다.

```bash
./scripts/run_host.sh --verbose
```

직접 실행과 systemd, Docker Compose는 모두 같은 `8696` 포트를 사용하므로 한 번에
하나만 실행한다.

운영 설정을 초기화할 때는 예제를 복사한 뒤 서비스를 재시작한다.

```bash
cp llama-server.env.example llama-server.env
systemctl --user restart llama-cpp-spark.service
```

## 사용자 systemd 서비스

```bash
./scripts/install_user_service.sh
systemctl --user status llama-cpp-spark.service
journalctl --user -u llama-cpp-spark.service -f
```

일반 운영 명령:

```bash
systemctl --user restart llama-cpp-spark.service
systemctl --user stop llama-cpp-spark.service
systemctl --user start llama-cpp-spark.service
```

서비스는 로그인 여부와 무관하게 부팅 시 실행되도록 user systemd에 등록되며,
SparkTalk과 SparkMedia보다 먼저 시작한다.

제거할 때는 다음을 실행한다. 모델과 host artifact는 보존한다.

```bash
./scripts/uninstall_user_service.sh
```

## Docker Compose 대체 실행

호스트 서비스와 같은 모델·옵션을 사용한다. 같은 8696 포트를 사용하므로 동시에
실행하지 않는다.

```bash
docker compose up -d --build
docker compose logs -f llama-server
docker compose down
```

현재 운영 환경은 Docker 컨테이너가 아니라 호스트 user systemd 서비스를 사용한다.

## 연동

OpenAI 호환 클라이언트 설정:

```text
Base URL: http://127.0.0.1:8696/v1
Model:    huihui-gemma4-12b
API key:  필요 없음
```

- SparkTalk: `http://127.0.0.1:8696`, 모델 `huihui-gemma4-12b`
- SparkMedia: 프롬프트 향상 등 OpenAI 호환 호출에 같은 엔진 사용
- LiteRT-LM: 서비스 중지 및 `compose_yaml/zzz_not_use/gemma4_litert`로 이동

## 선택 사항: mmproj 멀티모달

이미지 입력이 필요하면 텍스트 모델과 호환되는 `mmproj` GGUF 파일을 모델 디렉터리에
넣고 활성화한다. 현재 기본 구성은 아래 파일을 자동으로 사용한다.

```text
/home/edp1096/.cache/gguf/huihui-gemma4-12b-i1/mmproj-model-q8_0.gguf
```

호스트 직접 실행 또는 systemd에서는 `llama-server.env`에 다음 값을 지정한다.

```bash
LLAMA_MMPROJ_ENABLED=true
LLAMA_MMPROJ_FILE=mmproj-model-q8_0.gguf
```

설정을 바꾼 뒤 서비스를 재시작한다.

```bash
systemctl --user restart llama-cpp-spark.service
journalctl --user -u llama-cpp-spark.service -n 100 --no-pager
```

Docker Compose도 같은 환경변수를 사용한다.

```bash
LLAMA_MMPROJ_ENABLED=true \
LLAMA_MMPROJ_FILE=mmproj-model-q8_0.gguf \
docker compose up -d --build
```

멀티모달이 필요 없으면 `LLAMA_MMPROJ_ENABLED=false`로 설정한다. 호환되지 않는
mmproj를 사용하면 이미지 인식 품질 저하나 로딩 실패가 발생할 수 있으므로 모델과
함께 배포된 파일을 사용한다.

JPEG 이미지 입력 확인 예제:

```bash
base64 -w0 /path/to/image.jpg | jq -Rs '{
    model: "huihui-gemma4-12b",
    messages: [{
      role: "user",
      content: [
        {type: "text", text: "이 이미지를 한 문장으로 설명해줘."},
        {type: "image_url", image_url: {url: ("data:image/jpeg;base64," + .)}}
      ]
    }]
  }' | curl -fsS http://127.0.0.1:8696/v1/chat/completions \
    -H 'Content-Type: application/json' \
    --data-binary @-
```

## 확인

```bash
curl -fsS http://127.0.0.1:8696/v1/models
curl -fsS http://127.0.0.1:8696/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"huihui-gemma4-12b","messages":[{"role":"user","content":"한 문장으로 인사해줘."}],"temperature":0}'
```

관련 서비스 상태를 함께 확인하려면 다음을 사용한다.

```bash
systemctl --user is-active llama-cpp-spark.service sparktalk.service media-app.service
curl -fsS http://127.0.0.1:8585/api/health
```

멀티모달 입력은 OpenAI 형식의 `image_url` 콘텐츠를 사용한다. JPEG 입력은 확인됐으며,
이미지 설명이 필요한 SparkMedia 작업은 전용 Qwen3-VL 경로를 우선 사용한다.
