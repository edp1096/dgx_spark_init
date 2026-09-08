# 제거 시점 SparkTalk 프로필 기록

아래는 2026-09-08 제거 전 설명이다. 현재 앱의 기본값이 아니다.

### Qwen 27B EXL3 메인 프로필

기본 엔진은 ExLlamaV3 `63b32f0`, Uncensored 4bpw 가중치, MTP, NVFP4 KV 캐시다.
기본 문맥은 128K이며, 세트의 실행·포트 상세 설정에서 32K·64K·128K·256K를 선택한다.
Thinking은 꺼짐·낮음(low)·중간(medium)·매우 높음(xhigh)을 지원한다.
이전 32K 비교에서는 검사 후 추가 메모리 약 20.3GiB와 약 24.95 tok/s를 측정했다.
128K의 메모리 실측과 구분해야 하며 앱의 시작 전 메모리 예약은 보수적으로 32GiB다.

모델은 실행 호스트의 `model_cache/exl3-qwen38-27b-uncensored-4bpw`, 엔진 캐시는
`model_cache`의 상위 디렉터리 아래 `exl3-qwen38-27b`를 사용한다.
모델 준비는 Lygodactylus의 고정된 Uncensored 체크포인트를 내려받고, 기존 모델은 재사용한다.
API 서버와 빌드 파일을 앱에 포함하므로 작업용 저장소 체크아웃은 필요 없다.
현재 EXL3 API 어댑터는 이미지 입력을 지원하지 않는다.

EXL3 구성은 `compose_yaml/qwen38_27b_exl3`에 복원했다. GGUF 구성과 검사 기록은
`compose_yaml/zzz_not_use/qwen38_27b_gsq_rco_gguf`, NVFP4는 기존
`compose_yaml/zzz_not_use/sglang_qwen38_27b`에 보관한다.
저장된 GGUF/NVFP4 세트 선택은 EXL3로 이전하며, GGUF에서 선택한 문맥 한도를 유지한다.
마이그레이션만으로 서버를 자동 기동하지 않는다.
