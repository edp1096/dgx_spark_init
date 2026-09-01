# Nemotron ASR for SparkTalk

SparkTalk의 마이크 발화와 첨부 미디어 음성을 함께 처리하는 ASR이다. 생성 스튜디오의
장시간 자막은 별도 `compose_yaml/qwen3_asr`(Qwen3-ASR 1.7B + ForcedAligner)를
계속 사용한다.

- 런타임: NVIDIA NeMo-Speech.cpp (CUDA, OpenAI 호환 전사 API)
- 모델: Nemotron 3.5 ASR Streaming 0.6B 공식 Q8_0 GGUF
- API: `http://127.0.0.1:8693`
- 기본 언어: `ko-KR`; 요청별로 `auto` 또는 지원 locale 지정 가능
- 모델과 빌드 산출물은 Git에 포함하지 않는다.

## 모델 다운로드와 빌드

```bash
./scripts/download_model.sh
docker compose build
docker compose up -d
docker compose logs -f api
```

호스트에 컴파일 산출물을 보관하려면 다음을 실행한다. CUDA 의존성을 정확히
유지하기 위해 운영은 Compose 런타임 이미지를 권장한다.

```bash
./scripts/build_host.sh
```

빌드 병렬도는 메모리 급증을 피하려고 기본 2로 제한한다. 필요할 때만
`NEMO_BUILD_JOBS`로 변경한다. 소스와 모델 버전은 Dockerfile 및 Compose에 고정되어
있다.

## 확인과 벤치마크

```bash
curl -fsS http://127.0.0.1:8693/ready | jq
curl -fsS http://127.0.0.1:8693/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=nemotron-3.5-asr-streaming-0.6b \
  -F language=ko-KR

./scripts/benchmark.sh sample.wav ko-KR
./scripts/benchmark.sh sample.wav auto
```

벤치 결과는 `results/`에 기록되며 Git에서 제외된다. Q8 정확도와 메모리를 확인한
후 다른 GGUF가 제공되면 `NEMO_MODEL_FILE`만 바꿔 같은 런타임으로 순차 비교한다.

공식 Q8_0과 같은 런타임에서 F16을 비교하려면 변환도 컨테이너 안에서 수행한다.
호스트에는 모델 결과만 남으며 Python, Torch, NeMo 패키지를 설치하지 않는다.

```bash
./scripts/convert_model.sh f16
NEMO_MODEL_FILE=nemotron-3.5-asr-streaming-0.6b.f16.gguf docker compose up -d --force-recreate
```

SparkTalk 설정값:

```yaml
asr:
  endpoint: http://127.0.0.1:8693
  model: nemotron-3.5-asr-streaming-0.6b
  voice_language: ko-KR
  media_language: auto
```

마이크 발화 언어는 모델의 locale 코드를 사용한다. 예: `auto`, `ko-KR`,
`ja-JP`, `en-US`, `zh-CN`. 한국어 발화가 대부분이면 `ko-KR` 고정이 자동
감지보다 안정적이다. 첨부 미디어는 언어를 예측할 수 없으므로 같은 Nemotron에
`auto`를 전달한다. 두 경로의 언어는 설정 화면에서 각각 변경할 수 있다.
