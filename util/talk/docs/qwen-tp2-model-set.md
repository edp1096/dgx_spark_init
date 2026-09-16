# Qwen3.8 Flash-Next TP2 모델 세트

설정의 모델 세트에서 **Qwen3.8 Flash-Next TP2 · 1M**을 선택하고 시작한다. 기존 TP1 세트는 별도로 남아 있다. 저장된 카탈로그도 내장 개정 3에서 새 세트를 추가하며 기존 사용자 정의는 덮어쓰지 않는다.

- LLM: `flash-next-tp2`, SGLang TP2, 기본 1,048,576 문맥, BF16 KV, NEXTN 3/1/4, ko64k 초안 어휘, prefill chunk 1024.
- 체크포인트: `edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4`, 두 노드에 같은 가중치를 사용한다.
- 이미지: `dgx-sglang-qwen38-fn:sm121-tp2-vocab-v1`을 두 노드에 미리 준비해야 한다. 자동 다운로드·이미지 빌드는 이 세트의 기능에 포함하지 않는다.
- API 기본 포트: 8012. 헤드·워커 컨테이너는 `sglang-qwen38-fn-tp2-0/1`.
- FLUX 이미지, Nemotron ASR, Magpie TTS는 헤드에서 LLM 준비 완료 후 시작한다. Extra Media·SSH·Collector·Documents도 포함한다.
- 예상 LLM 메모리: 헤드 83 / 워커 81 GiB. 기동 전 두 노드 각각 가용 110 GiB를 요구한다. 실행 중 기존 호스트 watchdog의 8 GiB 바닥과 Docker 한도를 유지한다.

컨트롤러 `qwen38-cluster`는 앱에 내장된 `assets/recipes/qwen38-tp2.tar.gz`를 데이터 폴더에 풀어 사용한다. 실행 시 작업 저장소에 의존하지 않는다. 호스트·HF 캐시·런타임 캐시·컨테이너·API 주소·QSFP 설정은 카탈로그에서 전달한다. 모델 상태와 진행률, 중지 명령은 기존 클러스터 제어 경로를 사용한다.

서비스 편집의 문맥 선택은 256K·512K·1M을 지원하며 세트의 문맥 크기도 함께 갱신한다. 변경 후 재기동한다. FP8/NVFP4 KV 선택은 아직 노출하지 않는다.

원본 런처·Compose·watchdog·entrypoint는 `compose_yaml/qwen38_fn_sglang`과 내장 패키지가 동일한지 Go 테스트로 검사한다. 래퍼 소스는 `internal/orchestrator/recipe_sources/qwen38-tp2`에 있다. 패키지를 갱신할 때 개인 `.env`나 캐시·벤치마크 파일을 포함하지 않는다.

기존 단독 TP2 실측에서 256K·512K·1M 각각 실제 입력과 검색값 3/3 회수를 검증했다. Talk 통합 검증은 같은 1M 설정을 세트 시작 API로 기동하고, 부가 모델 준비·채팅 응답·성능 지표·세트 종료를 확인한다. 같은 긴 입력 시험을 반복한 성능 비교는 아니다.

## 통합 검증 결과

Talk의 세트 시작 API로 1M/BF16 TP2와 부가 모델 준비를 완료했다. Talk 채팅 API의 임시 대화에서 한국어 산술 응답과 pp·tg·ttft 이벤트를 확인했다. 해당 4,358토큰 입력의 ttft는 약 3.22초였다. 짧은 기능 검증이며 속도 벤치마크는 아니다. 임시 대화는 삭제했다.

세트 중지 API로 LLM 양쪽·FLUX·ASR·TTS가 종료된 것을 확인했고, 최종 Talk 빌드 재시작 후에도 TP2 세트 선택이 유지된다. [검증 기록](qwen-tp2-integration-check.json)을 참고한다.


2026-09-14: 내장 TP2 레시피의 기본 이미지를 `sm121-b12x-head-v1`로 변경했다.
BF16 1토큰 출력층만 B12X를 사용한다. MoE·QSA·GDN·BF16 KV·SSD PLE 설정은
유지한다. 반복 시험 tg 개선은 약 3.5~3.8%, pp는 변화가 없으며 1M 정확도
검증도 통과했다. 이미지 빌드/실측은 [SGLang 문서](../../../compose_yaml/qwen38_fn_sglang/docs/B12X_SGLANG_REVIEW.md)를 참고한다.

## Huihui-RadixArk 기본 모델 전환

TP1 `flash-next`와 TP2 `flash-next-tp2`의 모델 ID는 `edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4`다. 로컬 경로는 두 서버 모두 `~/.cache/huggingface/edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4`다.

카탈로그 revision 4는 이전 내장 모델 ID만 새 ID로 바꾸며 사용자 지정 이름·주소·문맥 크기·다른 모델은 보존한다. 설정된 TP1/TP2 세트 ID와 주변 모델 조합은 유지된다. 내장 TP2 패키지는 `python3 util/talk/internal/orchestrator/recipe_sources/qwen38-tp2/package.py`로 갱신한다.

공개 모델: [edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4](https://huggingface.co/edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4), revision `40f09f531da577a4fcfbd6b368e7b6ebacde403e`. 현재 기본·메인 세트는 TP2다. TP1 LLM 검증은 통과했으나, 이 호스트에서 FLUX 최대 메모리까지 예약한 전체 TP1 세트는 4GiB 최소 여유를 충족하지 못하므로 안전 기준을 유지하고 TP2를 사용한다.

2026-09-15: TP2 MTP shortlist의 전역 ID/로컬 출력층 처리를 수정했다. Compose와 내장 레시피는 같은 ko64k·1024 설정을 사용한다. 청크 증가와 한자 혼입 완화는 일관된 정확도·효과를 확인하지 못해 적용하지 않는다. [측정·검증 기록](../../../compose_yaml/qwen38_fn_sglang/docs/TP2_TUNING_20260915.md).
