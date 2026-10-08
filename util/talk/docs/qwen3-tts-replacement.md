# Qwen3-TTS Q8 교체 — 2026-10-08

화면의 표시 이름은 `Qwen3-TTS 0.6B`이며 양자화 정보에는 실제 Q8을 유지한다.

모든 내장9개 세트의 TTS를 Qwen3-TTS 0.6B CustomVoice 본체 Q8_0 + 배포
오디오 codec Q8_0로 교체했다. GLM에도 워커 TTS를 추가했다. 기존 설정과
카탈로그 가져오기의 Magpie 멤버십·바인딩은 Qwen으로 이전하며, 실행 위치와
더 큰 사용자 예산·활성화·자동 재생 선택은 보존한다. 이전 화자는 Sohee,
22.05kHz 설정은24kHz로 이전한다.

Talk의 Magpie 서빙·모델 변환·내장 레시피를 제거했다. 구 standalone Compose와
패치·과거 Talk 검증 기록은
[보관 디렉터리](../../../compose_yaml/zzz_not_use/magpie_tts/README.md)로 이동했다.
이전 이름은 설정 이전과 폐기 컨테이너 정리를 위한 호환 처리에만 남는다.
새 런타임은 [Qwen GGUF Compose](../../../compose_yaml/qwen3_tts_gguf/compose.yaml)와
[수명 관리 문서](tts-lifecycle.md)를 참조한다.

## GPU 서빙

- `sparktalk-qwen3-tts:0.6b-q8-lifecycle1`, 네이티브 C++ `tts-server`.
- CUDA13.1.1/GB10 sm121, `GGML_BACKEND=CUDA0`, 기본 동시 합성1개.
- HTTP `/v1/audio/speech`, PCM24kHz mono s16le 스트리밍.
- 본체 commit `51512f129a7419567f4b8abfb06801451789b8f1`;
  GGUF revision `b7ee2e8c7459c3bea99da23e3d178125a7d1713c`.
- 본체 SHA256 `4eb38675c736ed6ac72012846ac8d6ef80e5af8bc05726870f0b3a6569588519`.
- codec SHA256 `1883beeed99348fc35e23dd225e9082f93f6f8c109330a33d935baa8acdbfd94`.

모델은 실행 호스트의 `~/.cache/qwen3-tts`에 둔다. 모델 준비가 고정한 두 GGUF를
직접 다운로드·검증한다. Docker 이미지와 Git에는 모델 파일을 포함하지 않는다.
CUDA backend와 NVML의 GPU 할당을 실제 확인했다. CPU는 텍스트/HTTP 등에도
사용하며 CPU 전용 합성으로 대체한 시험이 아니다.

## 실제 검증

격리 포트18852에서7개 공통 문장을 합성했다. 한국어·영문 혼합·숫자 지정 읽기·
일본어·중국어·긴 문장 모두 이전 Q8/Q8 엔진 시험과 PCM이 byte 단위로 같았다.
등록 화자9개를 실제 API로 확인했다. 두 개 긴 요청의 대기 상태와 quiesce409,
유휴 quiesce200→음성503→resume200, 잘못된 JSON400, 연결 취소 후 유휴 복귀,
같은 PID 재사용을 확인했다. 동시 대기 시험의72초 출력은900프레임으로 제한한
종료 보호 시험이며 자연스러운 문장 완결성 시험으로 해석하지 않는다.

운영 Talk의 실제 `/api/tts/speech`에서도 혼합2문장과 지정 숫자 읽기의 PCM이
이전 엔진 시험과 완전히 같았다. 긴 답변은2개 요청으로 나누어113.44초 PCM을
19.16초에 생성했다. 마지막20초의 CPU SenseVoice 전사에서
`마지막 확인입니다. 모든 검사가 끝났습니다.`까지 복원했다.
긴 답변은 최대384자로 나누고 문장/단어 경계를 우선한다. 브랜드명과 소수점은
문장 경계로 오인하지 않는다. 명시 언어와 자동 언어에서 모두 본문 보존을 검사했다.

| 측정 | 격리 HTTP 서버 | 운영 Talk 읽기 경로 |
|---|---:|---:|
| GPU 피크 GiB | 2.441 | 2.439 |
| 프로세스 RSS 피크 GiB | 1.839 | 1.840 |
| 최소 시스템 가용 GiB | 15.252 | 17.419 |

통합 메모리이므로 GPU와 RSS를 더하지 않는다.50ms 샘플링 관측이며 더 짧은
피크를 놓칠 수 있다. 입장 예산은4GiB로 잡았다. 원자료는
`~/.cache/model-download-jobs/tts-gguf-comparison-20261008/qwen-serving-validation/` 및
`qwen-production-validation/`에 저장했다. 기존 발음 오류까지 해결됐다는 뜻이나
인간 청취/MOS 평가 결과는 아니다.

## 운영 적용 범위

로컬 Magpie 컨테이너는 실행·대기0의 유휴 상태에서 quiesce 후 종료·삭제했다.
운영 Talk를 새 바이너리로 갱신하고 Talk의 기동 API로 새 Qwen 컨테이너를 시작했다.
현재 EXL3 선택, TTS 활성화, 자동 재생false를 유지했다. EXL3·QWIM/MMH3·ASR·
Media·SSH·Collector는 컨테이너 ID와 PID가 유지됐다. 브라우저의 실제 설정 화면에서
Qwen 엔진·Sohee·24kHz·일본어 화자 옵션을 확인했다.

워커192.168.100.60은 SSH `No route to host`로 접근할 수 없었다. 워커 세트의
설정·다운로드·빌드·기동 레시피는 교체했지만 원격 서비스의 실제 교체/합성은
검증하지 않았다. 워커가 연결되면 Qwen 시작 경로가 준비 성공 후 이전 Talk
Magpie 컨테이너를 정리한다. 수명 API가 있는 이전 이미지는 quiesce를 먼저
요청하고 실행 중이면 종료를 거절한다. 기존 이미지의 graceful stop과 외부 이미지
보호·실행 중 종료 거절·폐기 컨테이너 부재를 회귀 시험으로 확인했다.

검사: Go 전체 테스트, 관련 race 검사, 초기 설치7개, 웹118개, 웹 빌드,
설정 탭 브라우저 검사, Linux/Windows arm64/amd64 빌드, 내장 자산·레시피 동기화,
Compose 구문, GPU 서빙·운영 API 검증을 통과했다. Git commit/push는 수행하지 않았다.

## Speak rate 추가

화자 아래에 `속도(Speak rate)`를 추가했다. 기본1.0, 범위0.5~2.0, UI 조절
단위0.1이며 설정 파일에 `tts.speak_rate`로 저장한다. Go 설정·유효성 검사·
응답 헤더와 브라우저 PCM player를 연결하고 음높이를 유지하는 WSOLA를
사용한다. 자동·수동 읽기는 같은 player를 사용한다. native 서버가 무시하는
`speed` 요청 값에 의존하지 않는다.

실제 Qwen의12.32초 음원은1.2배에서10.267초,1.3배에서9.477초로 줄었다.
113.44초의 긴 음원은94.533초/87.262초가 됐으며, 마지막20초의 CPU SenseVoice
전사에서 두 속도 모두 `마지막 확인입니다. 모든 검사가 끝났습니다.`를 복원했다.
1.0의 PCM은 byte 단위로 원본과 동일했다. 이 전사는 인간 청취 평가가 아니다.

Go 전체·웹130개·설정 저장/새로고침·실제 Web Audio1.3배 재생 검사를 통과했다.
실제 AudioBuffer의 출력 길이와220Hz 시험 신호의 음높이를 확인했다.
배포4종을 다시 빌드하고 운영 Talk를 갱신했다. 기존7개 서비스의 컨테이너
ID/PID와 EXL3 선택을 유지했다. Qwen의 PID2750233은 그대로이며 갱신 후
유휴 GPU 할당은2499MiB, RSS는1891360KiB였다. 속도 처리용 GPU 프로세스는
추가하지 않았다. 이 값은 합성 피크나 브라우저 전체 heap 측정값이 아니며
GPU와 RSS를 합산하지 않는다. 원자료는 같은 비교 작업 폴더의
`speak-rate-validation/`에 저장했다.

## 세트 재기동 오류 수정

QWIM/H3만 상주하고 EXL3·ASR·TTS가 중지된 상태에서, 기동 검사가 이미지의
향후 생성 작업 예산까지 추가해103.3GiB를 요구했다. 이미 반영된 이미지의
GPU·host 상주분을 빼고 아직 필요한 초기 적재분만 계산하도록 수정했다.
생성 요청의 작업 예산 검사는 그대로 유지한다. `stopWorkloads`도 상주 멤버를
보존하여 계산과 실제 동작을 일치시켰다. `start_after_llm` 서비스는 기존처럼
LLM 기동 전에 따로 중지하고 준비 후 다시 시작한다. QAD의 요청 시 실행·
유휴 회수 정책과 실제 메모리 부족 차단은 유지한다.

첫 재기동에서는 삭제된 Magpie의 Docker 응답 `error: no such object`를
대문자 전용 검사로 인식하지 못해 TTS 시작이 실패했다. 부재 검사에서
object/container와 대소문자를 모두 처리하도록 고쳤다. 이미 폐기된 컨테이너는
성공으로 처리하며, 권한 오류·외부 이미지·진행 중 작업은 계속 오류로 보호한다.
실제 Docker 소문자 오류와 같은 문자열을 회귀 시험에 추가했다.

최종 운영 검증은 QWIM/H3가 상주한 상태에서 EXL3만 중지 후 세트를 다시
시작했다. LLM 직전 가용100.1GiB/추가 예상92.1GiB로 통과했고, 이미지
컨테이너 ID와 PID3336367을 유지했다. EXL3·QWIM/H3·ASR·TTS가 모두
온라인이 됐다. 실제 문맥/KV1048576·Q8과 LLM 응답, Talk를 통한24kHz
PCM157440bytes·저장된 Speak rate1.3을 확인했다.

전체 검증의 시스템 가용량 최솟값은23.247GiB, 즉시 여유 최솟값은2.025GiB,
마지막 실제 응답 후 가용량은23.286GiB였다. 시스템100ms/GPU·RSS1초
간격으로 별도 기록했으며 GPU와 RSS를 더하지 않았다. Go 전체·관련 race·
QAD/상주 정책 회귀 검사와 배포4종 빌드를 통과했다. 원자료는 같은 비교
작업 폴더의 `exl3-start-memory-fix/`에 보존했다. 실제 QAD 모델을 새로 올리는
시험은 하지 않았으며 현재 선택은 EXL3로 유지했다.
