# llama.cpp for DGX Spark

DGX Spark(GB10, CUDA `sm_121a`)용 llama.cpp를 빌드하고, 컴파일된 런타임을
호스트에 보관해 OpenAI 호환 API로 실행한다. 현재 SparkTalk과 SparkMedia의
공용 텍스트·멀티모달 엔진이며 기존 LiteRT-LM 서비스의 `8696` 포트를 대체한다.

현재 기본 구성:

- llama.cpp: `v0.2.0`
- API: `http://127.0.0.1:8696/v1`
- 모델 ID: `huihui-gemma4-e2b`
- 모델: Gemma 4 E2B QAT `Q4_K`
- 컨텍스트: `65,536`
- MTP speculative decoding: 사용
- 멀티모달 `mmproj`: 사용
- GPU 레이어 전체 적재 및 flash attention: 사용

## 모델 파일

기본 모델 경로는 다음과 같다.

```text
/home/edp1096/.cache/gguf/huihui-gemma4-e2b-qat/
├── Huihui-gemma-4-E2B-it-qat-q4_0-unquantized-abliterated-Q4_K.gguf
├── mtp-ggml-model-bf16.gguf
└── mmproj-model-bf16.gguf
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
Model:    huihui-gemma4-e2b
API key:  필요 없음
```

- SparkTalk: `http://127.0.0.1:8696`, 모델 `huihui-gemma4-e2b`
- SparkMedia: 프롬프트 향상 등 OpenAI 호환 호출에 같은 엔진 사용
- LiteRT-LM: 서비스 중지 및 `compose_yaml/zzz_not_use/gemma4_litert`로 이동

## 확인

```bash
curl -fsS http://127.0.0.1:8696/v1/models
curl -fsS http://127.0.0.1:8696/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"huihui-gemma4-e2b","messages":[{"role":"user","content":"한 문장으로 인사해줘."}],"temperature":0}'
```

관련 서비스 상태를 함께 확인하려면 다음을 사용한다.

```bash
systemctl --user is-active llama-cpp-spark.service sparktalk.service media-app.service
curl -fsS http://127.0.0.1:8585/api/health
```

멀티모달 입력은 OpenAI 형식의 `image_url` 콘텐츠를 사용한다. JPEG 입력은 확인됐으며,
이미지 설명이 필요한 SparkMedia 작업은 전용 Qwen3-VL 경로를 우선 사용한다.
