# SparkTalk GGUF 메인 전환 · 2026-09-08

SparkTalk 기본·현재 세트를 `qwen27-gguf`로 변경하고 실행 중인 앱과 Linux/Windows
amd64/arm64 배포 바이너리를 갱신했다. 메인 API는 `127.0.0.1:8000`에서 실행한다.

- 엔진: 공식 llama.cpp e71b805, CUDA GB10 빌드
- MTP 3, Q8 KV, 프롬프트 캐시 1GiB
- 현재 배포 문맥 128K(131,072토큰), Thinking 기본 꺼짐
- 신규 설치용 내장 기본값은 32K이며 현재 배포에는 128K 실행 옵션을 저장
- 세트의 실행·포트 상세 설정에서 64K·128K·256K 선택 가능
- 문맥 선택은 실제 서버 인자, 앱 문맥 한도, 시작 전 메모리 예상량에 함께 반영
- EXL3: `../zzz_not_use/qwen38_27b_exl3/`
- NVFP4: 기존 `../zzz_not_use/sglang_qwen38_27b/`
- 모델 캐시의 `qwen38_27b_gsq_rco_gguf`는 이 폴더의 `data/`에 연결해 재사용

앱의 모델 준비 기능으로 두 GGUF 체크섬을 검증했고, 실제 앱 대화의 계산 답변,
이미지 좌우 색상, document_generate 도구를 통한 HWP·PDF 생성과 다운로드를 확인했다.
생성한 검증 대화는 삭제했으며 기존 대화와 설정은 유지했다.

## 128K 별도 검증

원본 상한은 262,144토큰이다. 131,072 문맥으로 별도 서버를 기동해
124,174토큰 입력의 중간에 숨긴 `CTX128-RIVER-7406`을 정확히 회수했다.
총 응답 시간 268.45초, 입력 처리 약 463.9 tok/s였다.
기존 32K 메인 서버를 유지한 상태에서 별도 검증 서버의 호스트 가용 메모리 감소량은
준비 직후 20.03GiB, 요청 후 21.73GiB였다. 최대 메모리 측정이나 모든 장문 작업의
정확도 보장은 아니다. 검증 서버는 종료했다. 이후 사용자 요청으로 메인 서버도 128K로 변경했다.

원본: [128K 검증 결과](data/deployment-20260908/context128/result.json),
[실제 앱 검사](data/deployment-20260908/app-checks.json),
[모델별 비교](COMPARISON-27B.md).

## Thinking 단계 선택

SparkTalk의 GGUF Thinking을 꺼짐·낮음(low)·중간(medium)·매우 높음(xhigh)으로 확장했다.
`chat_template_kwargs`에 `enable_thinking`과 활성 단계의 `reasoning_effort`를 전달한다.
기존 `on`은 이전 템플릿 기본값인 `xhigh`로 해석한다. 현재 기본 Thinking 꺼짐과 128K 문맥은 유지한다.
Go 설정·요청 테스트, 웹 단위 테스트, UI 선택·저장·재로드 검증을 통과했다.
실제 서버 템플릿과 네 단계의 계산 응답도 검증했다: [결과](data/deployment-20260908/thinking-levels.json).
