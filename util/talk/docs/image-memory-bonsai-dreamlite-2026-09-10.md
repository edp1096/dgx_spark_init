# Bonsai / DreamLite Mobile 메모리 실측 — 2026-09-10

측정 원본·스크립트·출력은 저장소 밖 `/tmp/dgx-spark-image-memory-records-2026-09-10/image-memory-2026-09-10-bonsai-dreamlite/`로 이동했다. 아래 파일명은 이 원본 경로 기준이다.

DGX Spark에서 DreamLite Mobile은 로딩부터 1024 생성·편집까지 추가 약 8.4GiB 이내였다. Bonsai ternary 공식 CUDA 경로는 1024 생성에서 추가 약 20.2GiB로, 이번 구성에서는 메모리 절약 효과가 없었다.

## 측정 조건

- NVIDIA GB10, OS 총 메모리 121.63GiB. 기존 LLM·ASR·TTS·FLUX를 정지하고 모델을 하나씩 실행했다.
- 다운로드·패키지 설치는 계측 전에 완료. 모델 프로세스 시작 전 호스트 사용량은 DreamLite 4.20GiB, Bonsai 4.21GiB였다.
- 호스트 `/proc/meminfo`의 `MemTotal - MemAvailable`을 약 100ms 간격으로 기록했다. 아래 추가량은 각 실험의 모델 미실행 기준 대비다. GPU와 CPU가 공유하는 시스템 메모리를 평가하기 위한 값이며, CUDA allocator 값과 더하지 않는다.
- 각 모델을 한 번 로딩하고 512 생성 2회 → 1024 생성 2회 순서로 실행했다. DreamLite는 이어서 1024 편집 2회 수행했다. 따라서 1024는 별도 프로세스 cold 실행이 아니다. 파일 캐시를 비우지 않았다.
- 동일 꿀병 프롬프트, seed 42/43, 4단계. 편집에는 이전 비교와 동일한 원본 사진의 1024 정사각 crop을 사용했다.
- 관측된 피크이며 모든 입력의 최대 요구량을 보장하지 않는다. 기존 LLM과의 동시 추론은 측정하지 않았다.

## 결과

| 모델 / 단계 | 호스트 추가 피크 | CUDA allocated 피크 | 첫 실행 / 반복 시간 |
|---|---:|---:|---:|
| DreamLite Mobile 로딩 | 8.37GiB | 4.70GiB | 28.83초 |
| DreamLite Mobile 생성 512 | 7.14GiB | 4.89GiB | 2.97 / 0.24초 |
| DreamLite Mobile 생성 1024 | 8.17GiB | 5.36GiB | 1.06 / 0.86초 |
| DreamLite Mobile 편집 1024 | 8.31GiB | 5.36GiB | 1.51 / 1.00초 |
| Bonsai ternary 로딩 | 16.78GiB | 4.27GiB | 13.27초 |
| Bonsai ternary 생성 512 | 17.10GiB | 4.91GiB | 27.09 / 1.52초 |
| Bonsai ternary 생성 1024 | 20.24GiB | 6.69GiB | 20.82 / 5.59초 |

전체 호스트 피크는 DreamLite 12.57GiB, Bonsai 24.46GiB였다. Bonsai의 첫 해상도별 실행에는 Gemlite/Triton 초기 실행 비용이 포함된다. 시간에는 이미지 저장이 포함되며 API 네트워크 시간은 없다. 단계별 원시값은 `phase-summary.json`, 전체 계측은 `*.summary.json`에 보관했다.

이전 동일 장비 비교의 1024 생성 작업 추가량은 FLUX 약 18.0GiB, SANA 약 7.4~7.8GiB였다. DreamLite는 FLUX보다 약 10GiB 적고 SANA와 비슷한 수준이다. Bonsai는 현재 측정한 공식 Python CUDA 경로에서 FLUX보다 약 2GiB 더 썼다. 별도 실행 시점과 구현이 다른 관측 비교다.

Bonsai의 CUDA allocated 6.69GiB만 보면 가벼워 보이지만 호스트 전체 추가량은 20.24GiB였다. 소스의 로딩 경로는 전체 비양자화 CPU 모듈을 생성한 뒤 dtype 변환과 양자화 모듈 교체를 수행한다. CPU 메모리·allocator 잔류가 차이에 영향을 줄 수 있지만 이번에 각 원인의 점유량을 별도로 분해하지는 않았다. 모델 파일 크기나 GPU allocator 수치만으로 DGX의 실제 필요 메모리를 판단하면 안 된다.

## 실행 구성과 기능

DreamLite는 `carlofkl/DreamLite-mobile`의 `diffusers` 브랜치를 BF16 `DreamLiteMobilePipeline`으로 실행했다. transformers 4.57.3, diffusers 0.39.0, huggingface-hub 0.36.2를 사용했다. 메타데이터와 실제 다운로드에서 공개 접근을 확인했고 필요한 가중치를 모두 받아 실행했다.

Bonsai는 `prism-ml/bonsai-image-ternary-4B-gemlite-2bit`와 `PrismML-Eng/image-studio`의 `backend_gpu`를 사용했다. Gemlite int2 transformer, HQQ 4bit 텍스트 인코더, BF16 VAE다. 공식 demo lock에 맞춰 gemlite 0.5.1.post1, hqq 0.2.8.post1, transformers 5.8.1, diffusers 0.38.0을 사용했다. GB10 지원을 위해 Torch는 기존 NVIDIA 26.07 이미지의 2.13.0a0 / CUDA 13.3을 유지했다.

검사한 Bonsai `generate_png` 구현은 텍스트 생성만 제공하고 참조 이미지·마스크를 받지 않아 편집은 측정하지 않았다. 해당 구현에서는 텍스트 인코더 offload가 적용되지 않고 `tiled_vae` 인자를 버린다. 이번 결과에 VAE tiling이나 offload가 적용됐다고 해석하면 안 된다. 소스 최적화는 이번 메모리 확인 범위에 포함하지 않았다.

모델 revision과 로컬 경로는 `models.json`, 패키지 목록·이미지 ID·소스 커밋은 `environment.json`, 실행 스크립트는 `dreamlite_bench.py`, `bonsai_bench.py`에 있다. 설치와 실행은 별도 실험 컨테이너에서 했다.

## 출력 확인

모든 PNG를 디코딩 검사하고 크기·모드·해시를 `image-checks.json`에 기록했다. 출력은 요청한 512 또는 1024 크기다. 대표 1024 생성과 DreamLite 편집을 열어 확인했다.

- Bonsai와 DreamLite 모두 정상적인 두 꿀병 생성 결과를 얻었다.
- DreamLite 편집은 원본의 크기가 다른 두 병과 노란 뚜껑을 대체로 유지하면서 배경을 연한 파란색으로 바꿨다. 질감과 일부 세부는 바뀌었다. 픽셀 보존이나 투명 배경 제거 결과는 아니다.
- 한 사진과 프롬프트로 얻은 결과라 종합적인 편집 품질 순위로 일반화하지 않는다.

개인 사진·생성 출력은 `private/`에 두고 Git에서 제외했다. 다운로드·로그·원시 시계열도 제외했다.

## 서비스 복구

기존 LLM·ASR·TTS·FLUX 4개 컨테이너를 복구했다. 8000/8691/8692/8693 health 모두 HTTP 200, LLM 1토큰 생성 HTTP 200을 확인했다. 실험 컨테이너는 정지했다. Talk에 엔진을 연결하거나 설정을 바꾸지 않았다.

LLM context_length=65,536, max_running_requests=2, mem_fraction_static=0.79를 유지했다. 자동 산정 KV 풀은 이번 재시작에서 157,824토큰이었다(직전 실행 160,832). 수동 설정 변경은 없었다. FLUX는 API 대기 상태이며 이미지 가중치를 다시 워밍하지 않았다. 복구 증거는 원본 경로의 `restoration-checks.json`, `after-restore.json`에 있다.
