# Magpie TTS for SparkTalk

SparkTalk 답변 읽기에 사용하는 경량 다국어 TTS 서비스다.

- 모델: NVIDIA Magpie TTS Multilingual 357M v2607 + NeMo NanoCodec
- 런타임: NeMo-Speech.cpp CUDA HTTP 서버
- API: `http://127.0.0.1:8692`
- 기본 언어와 음성: SparkTalk 자동 판별, `Sofia`(여성)
- 지원 언어: 아랍어, 중국어, 영어, 프랑스어, 독일어, 힌디어, 이탈리아어,
  일본어, 한국어, 포르투갈어, 스페인어, 베트남어
- 지원 음성: `Aria`(여성), `Sofia`(여성), `Jason`(남성), `John`(남성), `Leo`(남성)
- 출력: 22,050 Hz, mono, signed 16-bit PCM

v2607의 factor-2 프레임과 한국어 프로필이 아직 upstream main에 없으므로
[PR 17](https://github.com/NVIDIA/NeMo-Speech.cpp/pull/17)의 커밋
`7752dabb3453ec7c5fd751e5e2052650aaa20497`을 고정한다. 브랜치 이름이
움직여도 빌드 결과가 바뀌지 않으며, 커밋이 예상값과 다르면 빌드를 중단한다.
이 커밋의 변환기가 v2607 체크포인트의 화자 인덱스 순서를 잘못 기록하므로 로컬
패치로 원본 `speakers.json` 순서인 `Aria, Jason, John, Leo, Sofia`를 적용한다.

## 최초 준비와 실행

모델은 Git 저장소 밖의 `/home/edp1096/.cache/nemo-speech/magpie-v2607`에 둔다.
다운로드, tokenizer 추출, v2607 GGUF 변환은 다음 명령으로 수행한다. 변환용
Python 패키지는 컨테이너 안에만 설치된다.

```bash
cd compose_yaml/magpie_tts
./scripts/setup_models.sh
docker compose build api
docker compose up -d api
docker compose logs -f api
```

포트와 모델 위치는 필요할 때 환경변수로 바꿀 수 있다.

```bash
MAGPIE_MODEL_DIR=/data/models/magpie-v2607 MAGPIE_PORT=8692 docker compose up -d api
```

## 확인

```bash
curl -fsS http://127.0.0.1:8692/ready
curl -fsS http://127.0.0.1:8692/v1/models | jq
curl -fsS http://127.0.0.1:8692/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"magpietts","voice":"Sofia","input":"안녕하세요.","language":"ko-KR","response_format":"pcm"}' \
  -o hello.pcm
```

SparkTalk 설정 기본값도 이 서비스에 맞춰져 있다. Magpie는 Qwen3-TTS의
연기 지시와 seed를 지원하지 않는다. 언어와 고정 음성만 요청별로 전달한다.
`auto`는 SparkTalk이 지원 문자 체계와 라틴계 언어를 판별해 실제 언어 코드를
전달하는 앱 설정이며, Magpie API 자체에 `auto`를 보내지는 않는다. 일본어와
중국어 tokenizer도 이미지에 포함해 v2607의 12개 언어를 모두 사용할 수 있다.
