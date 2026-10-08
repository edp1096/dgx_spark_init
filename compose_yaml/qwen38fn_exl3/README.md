# Qwen 3.8 Flash-Next EXL3 (`qwen38fn_exl3`)

Talk의 **Qwen3.8 Flash-Next EXL3** 세트는 ExLlamaV3 1.5.4와 TabbyAPI를 사용한다.
기존 SGLang의 ARM64·CUDA 13·Torch 2.13 환경을 재사용하며, SGLang 프로세스는 실행하지 않는다.

- 가중치: `alesha-pro/Huihui-Qwen3.8-Flash-Next-abliterated-exl3-3bit-hq_h6_ng6`, revision `3b585c458f9fcf3322e76cff2c635c2cb81c5869`.
- 문맥과 Q8 KV: 각각 1,048,576. 원래 262,144 문맥을 YaRN 4배로 확장한다.
- MTP 초안 최대 3, 비전 켜짐, 추가 INT8 GEMV·전문가 CPU 오프로딩 꺼짐.
- PLE는 원본 128개 샤드를 디스크에서 필요한 행만 읽는다. 가중치를 다시 양자화하거나 큰 표를 메모리에 고정하지 않는다.

**설정 → 시스템 → 모델 준비**에서 이 모델의 전체 준비를 실행한 뒤 AI 세트에서 선택한다.
정상 캐시가 있으면 재사용한다. 새 설치에서는 고정된 공개 SGLang 이미지와 upstream 소스를 받아 CPU에서 SM121 확장을 빌드한다.
앱 실행 파일에는 Dockerfile·기동 파일·캐시 반환 파일이 포함되어 저장소 체크아웃을 요구하지 않는다.

`launch.py`는 읽기 전용 `/hf`의 원본 파일을 `/runtime/models/qwen38fn_exl3`에 연결한다.
자체 `config.json`에만 YaRN 설정을 적용한다. 실제 로드된 모델의 ID·문맥·KV·비전을 확인한 뒤,
Talk가 `release_weight_cache.py`로 닫히고 매핑되지 않은 일반 가중치 7개의 깨끗한 파일 캐시를 반환한다.
GPU 가중치·KV·활성 PLE 파일 캐시는 유지한다. 이 단계는 GB10의 부가 CUDA 서비스 기동을 위한 즉시 빈 메모리를 확보한다.

단독 빌드는 이 디렉터리에서 실행한다.

```sh
docker build --target qwen38fn_exl3-runtime -t sparktalk-qwen38fn_exl3:1.5.4-managed1 .
docker compose up -d
```

이미 검증한 `sparktalk-qwen38fn_exl3:1.5.4-sgl1` 이미지가 있는 머신은 `--target reuse`로 동일 엔진에 기동 파일만 추가할 수 있다.
일반 `docker build`와 Talk의 기본 빌드 대상은 공개 이미지에서 재현 가능한 `qwen38fn_exl3-runtime`이다.

2026-10-06/07 시험에서는 캐시 없는 실제 1,000,019토큰 입력의 약 10%·50%·90% 위치에서 세 값을 찾았다.
최종 성공은 필수 키와 문자열 타입만 지정한 JSON 스키마를 사용했다. 최초 일반 `json_object`는 `{}`를 반환했다.
이 검색 표본은 모든 장문 작업의 품질을 보장하지 않는다. 이미지·30초 ASR·TTS 병행 요청도 시험했으며,
최소 가용 메모리 약 29.6GiB, 새 스왑 쓰기와 모델별 OOM 0을 기록했다.
