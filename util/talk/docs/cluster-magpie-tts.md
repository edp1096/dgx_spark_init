# GLM·DS4·DS41의 Magpie TTS

세 모델 세트에 기존 Magpie TTS(`magpietts`)를 추가했다. 워커의
`http://192.168.100.60:8692`를 사용하며, 세트 선택 시 Talk의 TTS 주소가
해당 배치로 전환된다. 기존 음성·언어·활성화 설정은 유지한다.

| 세트 | TTS 위치 | 시작 순서 |
|---|---|---|
| GLM 5.3 Flash EXL3 | 워커 | TTS → LLM |
| DeepSeek V4 Flash Vision Exp | 워커 | ASR·TTS → LLM |
| DeepSeek V4.1 Flash | 워커 | ASR·Extra → LLM → TTS |

DS41의 TTS binding은 `start_after_llm: true`다. 기동 직전 요구하는 워커
가용 115GiB를 그대로 유지하기 위해 TTS만 뒤로 미룬다. 이미 TTS가 실행
중인 상태에서 DS41 LLM을 새로 시작해야 하면 TTS를 잠시 중지했다가 LLM
준비 후 다시 시작한다. LLM이 이미 준비되어 있으면 불필요하게 TTS를 중지하지
않는다. 기존 Expert 캐시, KV 캐시, 문맥 한도와 메모리 보호 기준은 변경하지 않았다.

모델 세트 중지는 TTS도 중지하며, 공용 Extra의 기존 유지 정책은 그대로다.
Qwen 등 다른 세트의 로컬 TTS 배치는 유지한다. 이 연결은 Talk의 모델 세트
관리 기능이며, 독립 LLM 실행 스크립트가 별도로 TTS를 기동하도록 바꾸지는 않았다.

기본 카탈로그와 현재 저장된 사용자 카탈로그 모두 반영했다. 다른 설치의
사용자 편집 카탈로그는 자동으로 덮어쓰지 않으므로 해당 세트에 `magpie-tts`와
워커 binding을 추가해야 한다. GLM 가져오기 예시도 갱신했다.

## 검증

- 워커에 `sparktalk-magpie-tts:v2607-longform2` 이미지를 준비하고 실제 서비스 기동.
- 한국어 음성 생성 2건: PCM 188,416 / 387,072바이트, 무음이 아닌 출력 확인.
- 관측된 컨테이너 메모리 피크 1,517,719,552바이트(약 1.41GiB), OOM 0.
- 세 모델 세트의 TTS 메모리 예상치는 각각 워커 1.6GiB로 설정.
- 세트별 배치·시작 순서, TTS endpoint 전환, 음성 설정 유지와 Qwen 복귀 검사 통과.
- Go orchestrator/config/server 검사와 Linux arm64 빌드 통과.

검증 범위는 독립 워커 TTS 음성 생성과 세트 연결·기동 순서다. 세 LLM 각각을
올려 긴 문맥 추론과 긴 음성 생성을 동시에 수행한 최대 메모리 실측은 아니다.
[실제 음성 생성 기록](cluster-magpie-tts-check.json)을 참조한다.
