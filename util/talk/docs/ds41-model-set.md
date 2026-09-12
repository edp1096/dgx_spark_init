# DeepSeek V4.1 Flash 모델세트

`ds41` 세트는 TP2 SSD expert streaming LLM, 워커 Nemotron ASR 및 헤드 Extra Media·SSH·Collector·Documents를 연결한다. 문맥은 65,536이며 요청별 최대 출력 기본값 16,384는 Talk 설정을 따른다.

`start_support: true`인 세트는 등록된 Extra 서비스도 먼저 시작한다. ASR 등 주변 서비스가 준비된 후 LLM을 시작하고 직전에 메모리를 다시 검사한다. 세트 중지는 기존처럼 공용 Extra를 유지한다. 다른 세트의 시작 동작은 바뀌지 않는다.

LLM은 `ds41-cluster` 컨트롤러가 두 랭크를 함께 관리한다. 실행 코드는 앱에 포함된 `assets/recipes/ds41.tar.gz`를 앱 데이터 디렉터리에 풀고 워커로 전달한다. 실행 시 작업 저장소 경로를 참조하지 않는다. 호스트, HF 캐시, 컨테이너 이름, API 포트와 통신망 설정은 카탈로그에서 전달된다.

현재 준비된 원본 체크포인트와 rank별 packed expert, `dgx-ds41-stream:b12x8` 이미지가 두 호스트에 필요하다. 새 시스템용 이미지 빌드·체크포인트 다운로드·expert packing 자동 설치는 이 등록 작업의 범위에 포함하지 않았다. 기존 `compose_yaml/ds41f_vllm` 준비 절차를 사용한다.

실행값은 기존 검증값을 유지한다: target expert 224개/레이어, draft 128개/레이어, KV 2GiB/rank, DSpark 5, prefill scheduler 4096, native kernel 2048, shared I/O buffer, scratch 384MiB. Decoder row 실험과 probe 제어는 꺼진다. 부족한 메모리에 맞춰 캐시를 자동 축소하지 않고 기존 launch 메모리 검사에서 중단한다.

런타임 변경 후 패키지를 재생성한다:

```sh
python3 util/talk/internal/orchestrator/recipe_sources/ds41/package.py
cd util/talk
go test ./internal/orchestrator ./internal/config ./internal/server
```

패키지 테스트가 독립 실행 소스·패치와 내장 파일의 일치 여부, 경로/환경 안전성 및 세트 구성을 검사한다. 기존 설정에는 내장 카탈로그 revision 2 마이그레이션으로 세트를 추가하며 사용자 선택과 기존 DS41 정의를 덮어쓰지 않는다.

## 2026-09-12 실기동 검증

처음에는 ASR과 Extra 4종을 모두 워커에 배치했으나, 기동 직전 가용 114GiB로 기존 115GiB 검사에서 중단됐다. 최종 배치는 워커 ASR + 헤드 Extra이며 캐시 축소나 검사 우회 없이 기동했다. 중복 생성한 워커 Extra는 중지했다.

양쪽 rank의 target slots 224, shared buffer 1, kernel 2048, scratch 384MiB, batch_overlap 및 decoder row 실험 OFF를 실제 컨테이너에서 확인했다. ASR·Extra가 먼저 준비된 후 LLM 두 rank가 시작되어 API가 정상 응답했다.

Talk `/api/chat`에서 입력 4,686토큰의 새 대화가 숫자 `5`를 반환했다. pp 약 147t/s, ttft 약 31.9초였으며 출력이 2토큰뿐이므로 이 검증의 tg는 성능 비교용이 아니다. `/api/asr/transcribe`에서 합성 영어 음성을 정확히 인식했다. 검증 시에만 받아쓰기 언어를 en-US로 맞추고 원래 ko-KR로 복원했다. 문맥 65,536, 출력 최대 기본값 16,384와 기존 reasoning low를 유지했다.

메모리 기록·기동 순서·결과는 [실측 JSON](ds41-model-set-validation.json)에 보존했다. 관찰 중 두 서버의 재부팅이나 swap 증가가 없었다. Go orchestrator/config/server 테스트, 프런트엔드 89개 테스트와 화면 빌드가 통과했다.

## Expert 사전 적재 적용

2026-09-12 후속 측정으로 기동 시 expert 사전 적재를 적용했다. 커널 준비 후
기존 224개 슬롯을 이전 라우팅 빈도에 따라 채우고 API 준비를 완료한다.
추가 슬롯·양자화·expert 제거는 없다. 사전 적재에는 서버 두 대를 병렬로
약 12.6초가 걸리며, 측정한 첫 요청 ttft는 입력에 따라 약 22~70% 줄었다.
반복 요청이나 tg 전체의 일괄 개선을 뜻하지 않는다.

DS41 서비스 또는 세트 binding의 `runtime_options.DSV41_PRELOAD_COUNT`로
0..224를 지정할 수 있다. 기본값은 224, 0은 끄기다. 캐시 총용량과 별개로
기동 때 채울 개수만 정한다. [상세 측정](../../../compose_yaml/ds41f_vllm/docs/EXPERT_PRELOAD.md)을 참조한다.
