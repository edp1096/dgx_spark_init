# SparkTalk EXL3 메인 복귀

GGUF 메인 구성을 보관하고 EXL3를 기본·현재 세트로 복원했다.

검증 시각: 2026-09-08T14:05:26+09:00

- 기본·현재 세트: `qwen27-exl3`
- 현재 문맥: **128K(131,072토큰)**
- 현재 Thinking: **medium**; 꺼짐·low·medium·xhigh 선택 지원
- 모델: `Lygodactylus/Qwen3.8-27B-Uncensored-exl3-4bpw`
- 이미지: `dgx-exl3-qwen38-27b:63b32f0-sparktalk1`
- API: `127.0.0.1:8000` · SparkTalk: 포트 `8585`
- ExLlamaV3 MTP, NVFP4 KV 캐시. 실제 서버 건강 정보와 `--cache_size` 모두 131,072로 확인
- 현재 모델은 실행 중이다. 다른 LLM은 시작하지 않았다.

## 복귀 내용

EXL3 구성은 `compose_yaml/qwen38_27b_exl3`로 복원했다. GGUF 구성·가중치·비교 기록은
`../zzz_not_use/qwen38_27b_gsq_rco_gguf`에 보관했다. GGUF 모델 캐시 심볼릭 링크도 새 위치로 연결했다.
NVFP4는 기존 보관 위치에 유지했다. SparkTalk의 내장 카탈로그·실행 및 빌드 자산·모델 준비 절차·설정 UI·기본 설정을 EXL3로 변경했다.
기존 GGUF 세트는 EXL3로 이전하며 선택된 문맥 크기와 Thinking을 유지한다. 마이그레이션 자체는 자동 기동하지 않는다.

EXL3 API 서버가 `reasoning_effort`를 모델 템플릿으로 전달하도록 연결했다.
내장 Docker 빌드와 독립 실행 구성에 같은 API 파일을 포함했다. Linux/Windows amd64/arm64 배포 바이너리 네 개를 갱신했다.
응답 완료 후 스크롤 위치를 유지하는 수정도 유지했다.

## 확인 결과

- Go 전체 테스트 및 변경된 설정·LLM·실행 구성 테스트 통과
- 웹 단위 테스트 통과
- UI: EXL3 Thinking 선택·저장, 모델 다운로드 화면, 세트 설정, 완료 시 스크롤 유지 검사 통과
- 실제 앱의 모델 준비: 기존 4bpw 가중치 재사용 확인
- 실제 API: 네 Thinking 모드 전달과 생각 출력 유무 확인
- 실제 앱: 기본 medium의 계산 답변 완료, HWP 및 PDF 미리보기 생성·다운로드 확인
- 앱이 제공하는 UI 파일과 새 빌드 파일이 동일함을 확인

Thinking 꺼짐으로 `17+8-9`를 물은 직접 API 검사에서는 `6`이라는 오답이 나왔다.
low·medium·xhigh는 `16`을 반환했다. 모델 정보·계산 정확도는 구성 복귀와 별개의 한계이며,
이전 [반복·환각 검사](../zzz_not_use/qwen38_27b_gsq_rco_gguf/TWO-PROMPT-CHECK.md)도 보관했다.
현재 EXL3 API 어댑터는 이미지 입력을 지원하지 않는다.

[Thinking 검사](results/sparktalk-restore/thinking-checks.json) · [실제 앱 검사](results/sparktalk-restore/app-checks.json) · [배포 정보](results/sparktalk-restore/deployment.json)
