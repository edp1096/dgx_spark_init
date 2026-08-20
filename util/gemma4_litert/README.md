# Huihui Gemma 4 E2B LiteRT-LM

`huihui-ai/Huihui-gemma-4-E2B-it-abliterated`를 Linux ARM64에서 직접
LiteRT-LM으로 변환하고 DGX Spark GPU로 서비스한다. 완성 번들은 텍스트뿐 아니라
Gemma 4 비전 encoder/adapter를 포함하므로 이미지 prompt enhancement에도 사용할 수
있다.

최종 모델/저장소 이름은
`Huihui-gemma-4-E2B-it-abliterated-litert-lm`이다.

## 구성

- `scripts/build_converter_arm64.sh`: Linux aarch64용 `litert-converter`
  wheel을 LiteRT 소스에서 네이티브 빌드
- `scripts/setup_conversion_env.sh`: 변환용 Python 3.12 환경과 고정된 upstream
  소스 구성
- `scripts/download_model.sh`: 고정 리비전의 Huihui 원본 모델 다운로드
- `scripts/convert_multimodal.sh`: INT8 텍스트/비전 모델 변환 및 `.litertlm` 패키징
- `scripts/verify_bundle.sh`: 번들을 다시 풀어 필수 멀티모달 section 검증
- `scripts/setup_runtime_env.sh`: 변환 도구와 분리된 경량 LiteRT-LM 실행 환경 구성
- `scripts/run_server.sh`: 패치된 LiteRT-LM OpenAI 호환 서버 실행
- `systemd/media-prompt-enhancer.service`: 8696 포트의 사용자 서비스

소스 체크아웃, Bazel 캐시, 변환용 Python 환경, 원본/완성 모델은 각각
`builder/`, `artifacts/`, `models/`, `output/`에 두며 Git에는 포함하지 않는다.
이들은 모두 재생성 가능한 빌드 산출물이며 `make clean-build`로 제거한다.

실행 환경은 저장소 밖
`${GEMMA4_LITERT_RUNTIME_ROOT:-$HOME/.local/share/gemma4-litert}/venv`에 둔다.
등록 모델과 GPU 캐시는 LiteRT-LM 기본 데이터 경로인 `$HOME/.litert-lm`에
보관하므로 저장소를 정리하거나 다시 받아도 유지된다.

## 컨테이너 빌드·변환

권장 경로는 컴파일러와 대규모 Python 변환 환경을 ARM64 컨테이너 안에 격리하는
방식이다. 호스트에는 Docker만 필요하며 Go, Bazel, Clang, PyTorch, Transformers를
설치하지 않는다. wheel·다운로드 모델·완성 번들과 재사용 캐시는 기본적으로 저장소
밖의 `$HOME/.local/share/gemma4-litert-build`에 bind mount로 보존한다.

```bash
make container-build    # 도구 이미지 빌드 + custom converter wheel 컴파일
make container-convert  # wheel 준비 + 원본 모델 다운로드 + .litertlm 변환
make container-verify   # 호스트에 생성된 번들 검증
make container-all      # 변환 후 검증
make container-paths    # 호스트 결과 경로 표시
make container-ps       # 현재 실행 중인 변환 컨테이너 확인
make container-logs     # 현재 실행의 Docker 로그 실시간 확인
```

각 작업의 전체 출력은 `$HOME/.local/share/gemma4-litert-build/logs`에도 저장한다.
따라서 `--rm` 컨테이너가 끝난 뒤에도 `container-build.log`,
`container-convert.log`, `container-verify.log`를 확인할 수 있다. 실행 중에는 위의
`make container-logs` 또는 `docker logs -f <container-name>`을 사용한다.

`make container-build`가 생성한 wheel은 추가 복사 작업 없이 다음 경로에 남는다.

```text
$HOME/.local/share/gemma4-litert-build/artifacts/wheels/
```

완성 모델은 다음 경로에 생긴다.

```text
$HOME/.local/share/gemma4-litert-build/output/Huihui-gemma-4-E2B-it-abliterated-litert-lm/model.litertlm
```

다른 디스크를 사용하려면 `CONTAINER_DATA_DIR=/path make container-all`처럼 지정한다.
Hugging Face 인증이 필요하면 호스트의 `HF_TOKEN`만 컨테이너에 전달한다. 토큰과 모델은
이미지 레이어에 들어가지 않는다. 컨테이너는 작업마다 `--rm`으로 제거되며 8696
LiteRT-LM 런타임과 systemd 서비스에는 영향을 주지 않는다.
컨테이너 베이스는 Ubuntu 24.04 ARM64 digest로, Bazelisk는 Go 1.22와 실제 의존성이
호환되는 v1.25.0으로 고정한다.

컨테이너 빌드 데이터 전체를 지우려면 다음을 실행한다. 전용 marker가 있는 정확한
빌드 루트만 삭제하며 호스트 런타임과 `$HOME/.litert-lm` 등록 모델은 보존한다.

```bash
make clean-container-build
```

## 호스트 네이티브 빌드·변환

컨테이너를 사용할 수 없을 때만 호스트 네이티브 경로를 사용한다. Ubuntu 24.04
ARM64, Python 3.12, Clang, Go가 필요하다. 전체 절차는 다음 한 명령으로 재현할 수
있다.

```bash
make all
```

단계별 실행도 가능하다.

```bash
make converter
make env
make download
make convert
make verify
```

`make converter` 결과는
`artifacts/wheels/litert_converter-0.4.0-cp312-cp312-manylinux_2_27_aarch64.whl`에
생긴다. Linux aarch64 wheel이 PyPI에 없어 LiteRT 소스에서 직접 빌드하며 x86
에뮬레이션은 사용하지 않는다.

변환 설정은 다음과 같다.

- task: `image_text_to_text`
- text/vision quantization: `dynamic_wi8_afp32`
- prefill: 128, 256, 512
- KV cache: 4,096 tokens
- image soft tokens: 280 (원본 Gemma 4 processor 설정과 동일)
- audio encoder: 미포함

## 호환성 패치

고정된 2026-08-18 upstream 소스에는 두 가지 호환성 문제가 있어 작은 패치를
함께 관리한다.

1. LiteRT Torch가 최신 Transformers 5의 heterogeneous config에서 전역
   `head_dim`을 읽는다. 실제 레이어 설정을 읽도록 수정했다.
2. LiteRT-LM OpenAI 서버가 일반 요청에도 constrained decoding을 항상 켠다.
   Hugging Face tokenizer에서는 실패하므로 `response_format`이 지정된 경우에만
   켜도록 수정했다.

패치는 `patches/`에 있으며 `make env`가 자동 적용한다.

## 실행

경량 실행 환경을 준비한다. 변환용 환경의 Torch/JAX/Transformers는 서버에
설치되지 않는다.

```bash
make runtime
```

완성 번들을 등록한다.

호스트 네이티브 변환 결과:

```bash
${GEMMA4_LITERT_RUNTIME_ROOT:-$HOME/.local/share/gemma4-litert}/venv/bin/litert-lm import \
  output/Huihui-gemma-4-E2B-it-abliterated-litert-lm/model.litertlm \
  huihui-gemma4-e2b
```

컨테이너 변환 결과:

```bash
${GEMMA4_LITERT_RUNTIME_ROOT:-$HOME/.local/share/gemma4-litert}/venv/bin/litert-lm import \
  ${CONTAINER_DATA_DIR:-$HOME/.local/share/gemma4-litert-build}/output/Huihui-gemma-4-E2B-it-abliterated-litert-lm/model.litertlm \
  huihui-gemma4-e2b
```

서비스를 설치하고 시작한다.

```bash
make install-service
```

서비스를 중지하고 자동 시작 등록과 사용자 unit 파일을 제거하려면 다음을 실행한다.

```bash
make uninstall-service
```

서비스 해제는 `${GEMMA4_LITERT_RUNTIME_ROOT:-$HOME/.local/share/gemma4-litert}`의
LiteRT-LM 실행 환경과 `$HOME/.litert-lm`의 등록 모델·GPU 캐시를 삭제하지 않는다.
따라서 `make install-service`로 같은 모델을 다시 등록할 필요 없이 서비스를 복구할
수 있다.

빌드가 끝난 뒤 원본 모델, 소스 checkout, 변환 중간물과 로그를 제거하려면 다음을
실행한다. 이미 `$HOME/.litert-lm`에 등록한 모델과 실행 환경은 삭제하지 않는다.

```bash
make clean-build
```

8696 서비스를 해제하고 이 프로젝트가 등록한 `huihui-gemma4-e2b` 모델과 전용
LiteRT-LM 실행 환경까지 제거하려면 다음을 실행한다. 다른 LiteRT-LM 등록 모델과
공용 데이터 디렉터리는 보존한다.

```bash
make clean-runtime
```

변환 산출물과 런타임·등록 모델을 모두 정리하려면 다음을 실행한다.

```bash
make clean-all
```

OpenAI 호환 API는 `http://127.0.0.1:8696/v1/chat/completions`, 모델 ID는
`huihui-gemma4-e2b`이다. 외부 호스트에서는 Spark의 IP와 같은 포트 8696을
사용한다.

Hugging Face write 토큰을 설정하면 최종 저장소로 업로드할 수 있다.

```bash
HF_TOKEN=hf_write_token make upload
```

기본 저장소는
`edp1096/Huihui-gemma-4-E2B-it-abliterated-litert-lm`이며 `HF_REPO_ID`로
바꿀 수 있다.

## DGX Spark 실측

2026-08-19, 다른 이미지/TTS/ASR 엔진이 실행 중인 상태에서 측정했다.

| 항목 | 결과 |
| --- | ---: |
| ARM64 converter 빌드 | 성공, 75MB wheel |
| 전체 변환 시간 | 4분 48초 |
| 변환 최대 RSS | 60.3GiB |
| 변환 중 swap I/O | 0 |
| 완성 `.litertlm` | 4.9GiB |
| Text prefill/decode INT8 | 2.15GiB |
| External embedder INT8 | 387MiB |
| Per-layer embedder INT8 | 2.19GiB |
| Vision encoder INT8 | 163.4MiB |
| Vision adapter INT8 | 1.15MiB |
| 최초 text GPU 실행(컴파일 포함) | 8.66초 |
| 캐시된 image GPU 실행 | 4.44초 |

검증 이미지에서 모델은 뒤에서 본 사이버펑크 복장의 여성과 주황/호박색 간판을
정확히 식별했다. CLI와 OpenAI 호환 API의 텍스트·이미지 요청을 모두 확인했다.
