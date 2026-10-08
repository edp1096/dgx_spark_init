# Qwen3-TTS의 서빙과 수명 관리

모든 내장 세트는 Qwen3-TTS 0.6B CustomVoice 본체 Q8_0 + 오디오 codec Q8_0을
사용한다. 기본 화자는 Sohee이고 출력은24kHz mono s16le PCM이다.
Magpie의 Talk 레시피·변환·서빙 연동은 제거했다. 구 Compose 및 과거 실측은
`compose_yaml/zzz_not_use/magpie_tts/`에 보존한다.

서빙은 ServeurpersoCom/qwentts.cpp commit
`51512f129a7419567f4b8abfb06801451789b8f1`의 네이티브 `tts-server`이다.
CUDA13/GB10 빌드이며 `GGML_BACKEND=CUDA0`으로 GPU를 지정한다. GPU 초기화
실패를 CPU 전용 서빙으로 감추지 않는다. 텍스트·HTTP 처리는 CPU도 사용한다.
본체/codec은 HF revision `b7ee2e8c7459c3bea99da23e3d178125a7d1713c`의
검증한 SHA256을 사용한다. 처음 준비할 때 두 GGUF를 직접 받는다.

## 요청과 종료

`POST /v1/audio/speech`의 PCM 경로는 생성 중 음성을 전달한다. Talk는
로케일을 Qwen 언어 이름으로 변환하고 seed42를 사용한다. 한국어 문장 안의
영문 브랜드명과 약어는 잘게 나누지 않는다. 긴 답변은 문장·단어 경계를 우선해
최대384자로 나누므로 네이티브 문맥 한도를 넘는 단일 요청을 만들지 않는다.
문장별 언어 판별·한자 독음
선택·괄호 생략 설정과 사용자의 자동 재생 선택은 유지한다.

서버는8개 HTTP worker 중 최대4개의 음성 요청을 받는다. 합성은1개씩 수행하고
나머지는 대기한다. 입장과 quiesce는 같은 mutex로 보호한다. PCM 입장 lease는
응답 전송 종료와 합성 thread join까지 유지하므로, 합성 후에도 전송 중인
요청을 유휴로 오인하지 않는다. 잘못된 JSON과 합성 실패도 lease를 반환한다.

- `/ready`, `/health`: 모델을 로드한 서버의 상태.
- `/v1/runtime/memory`: 실행·대기·quiescing·마지막 완료 후 유휴 시간.
- `/v1/runtime/quiesce`: 작업이 있으면409, 없으면 새 입장을 막고200.
- `/v1/runtime/resume`: 입장을 다시 허용.

상태가 불명확하면 자동 회수하지 않는다. Talk의 작업 큐는 요청을 기다리는
사용자도 보호하며 여러 문장의 PCM 소비가 끝날 때까지 lease를 유지한다.

## 배치와 예산

EXL3는 LLM 다음에 TTS를 시작하고 `keep_resident: true`로 유지한다.
QAD는 실제 요청에서 시작하고 연속 요청은 재사용한다. 마지막 요청 후2분
유휴 또는 메모리 압박 시 유휴 프로세스를 종료한다. 다른 세트의 기존 배치는
유지하며, GLM에는 워커 TTS를 LLM 뒤에 시작하는 배치를 추가했다.
기존 설정의 Magpie ID와 멤버십·바인딩을 Qwen으로 이전한다. 사용자의
TTS 활성화·자동 재생 선택·실행 호스트·endpoint·더 큰 예산을 보존한다.
호환되지 않는 Magpie 화자는 Sohee, 22.05kHz 설정은24kHz로 이전한다.

입장 예산은 전체4GiB, 초기화3GiB, 작업 여유1.5GiB다. 컨테이너 RAM과
swap 포함 한도는 모두12GiB이며 추가 swap은 허용하지 않는다. 통합 메모리는
GPU와 RSS가 겹치므로 둘을 더하지 않는다. 시스템 실제 가용량 검사도 유지한다.
실측 예산이 모든 입력의 절대 최대 메모리를 보증하지는 않는다.

## 교체 전 엔진 비교

같은13문장에서 Magpie 대비 Qwen Q8/Q8의 GPU 피크는1.47→2.42GiB,
RSS는0.99→1.83GiB였다. 시스템 가용량 감소는 약2.39→3.43GiB이며
호스트 활동·파일 캐시를 포함한다. 중앙 RTF는0.079→0.163이었다.
이 수치는 합성 엔진 시험이며 HTTP 서버의 부가 메모리는 별도로 관측한다.
발음은 두 ASR와 파형으로 확인했고 인간 청취/MOS 평가는 하지 않았다.
원자료: `~/.cache/model-download-jobs/tts-gguf-comparison-20261008/`.

서빙·Talk 교체의 실제 검증 기록은 `qwen3-tts-replacement.md`에 기록한다.

## 말하기 속도

`tts.speak_rate`는 기본1.0, 허용 범위0.5~2.0이다. 설정의 음성 > 답변 음성에서
화자 아래에 배치하며 자동·수동 읽기 모두 저장값을 사용한다. 기본값이 없는
기존 설정은1.0으로 정규화한다. `/api/tts/speech`의 `X-Audio-Speak-Rate`
헤더는 같은 요청의 설정 snapshot을 전달한다.

고정한 native Qwen 서버는 OpenAI `speed` 값을 파싱하지만 사용하지 않으므로
그 값만 보내는 방식은 쓰지 않는다. 브라우저의 PCM player가 SoundTouchJS
core2.1.1의 WSOLA로 속도를 조절한다. 음높이는 유지하고 PCM24kHz와 GPU
합성 경로는 유지한다.1.0은 처리 없이 원본 샘플을 그대로 재생한다.
네트워크·문장 경계마다 처리기를 초기화하지 않으며, 끝에는 lookahead를
flush하고 실제 입력 길이/속도로 출력 길이를 제한한다. 처리 지연과 청취
품질은 기기·음성에 따라 달라질 수 있으며 이 기능은 합성 시간을 단축하지 않는다.

검증:0.5·1.2·1.3·2.0의 출력 길이,220Hz 본문·660Hz 마지막 구간의 음높이,
짧은 발화·불규칙 PCM byte 경계·문장 큐·실제 저장 후 새로고침을 확인한다.
