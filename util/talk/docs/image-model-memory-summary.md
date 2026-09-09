# 실행한 모델 전체 메모리 비교 — 2026-09-10

DGX Spark / GB10, OS 총 메모리 121.63GiB. 이미지 실험은 다른 모델을 정지하고 단독 실행했다. 아래 수치는 모델 미실행 기준의 **시스템 메모리 추가량**(`MemTotal - MemAvailable`)이며 모델 파일 크기나 GPU allocator 수치가 아니다. GiB 단위, 관측 피크 기준이다.

| 모델·구성 | 로딩 포함 전체 실험 추가 피크 | 생성 작업 추가 피크 | 편집 작업 추가 피크 | 출력 / 설명 |
|---|---:|---:|---:|---|
| FLUX.2 Klein 4B NVFP4 | 18.43 | 약 18.0 | 18.13~18.43 | 1024, 참조 편집·일반 마스크·LanPaint 포함 |
| FLUX.2 Klein 4B NVFP4 | 16.53 | 약 16.0 | 16.53 | 512; 768 편집은 17.54 |
| SANA-Sprint 0.6B + NF4 텍스트 인코더 | 8.13 | 7.4~7.8 | 7.6~7.8 | 1024, 마스크 인페인트·아웃페인트 |
| SANA-Sprint 0.6B + NF4 텍스트 인코더 | 7.89 | 약 7.3 | 미측정 | 512 |
| Mobile-O-0.5B 공식 CUDA | 7.24 | 5.58 | 5.85 | 512, 참조 이미지 편집 |
| DreamLite Mobile BF16 | 8.37 | 7.14 / 8.17 | 8.31 | 생성 512 / 1024, 편집 1024 |
| Bonsai Image ternary 4B Gemlite int2 + HQQ 4bit TE | 20.24 | 17.10 / 20.24 | 미측정 | 생성 512 / 1024; 실행한 CUDA API에 이미지 입력 없음 |
| rembg U²-Net 소형 u2netp, CPU | 0.53 | 해당 없음 | 0.53 | 1920×2560 배경 제거, 다운로드 완료 후 실행 |
| rembg U²-Net 일반 u2net, CPU | 0.78 | 해당 없음 | 0.78 | 1920×2560 배경 제거 |

전체 실험 피크는 모델 로딩부터 작업까지 포함한다. 로딩 단계만 분리한 추가 피크는 DreamLite **8.37GiB**, Bonsai **16.78GiB**, Mobile-O **7.24GiB**다. FLUX/SANA는 첫 작업까지 포함한 cold 실행으로 측정했으므로 로딩 단계만의 값으로 해석하지 않는다.

u2netp 최초 다운로드·초기화 포함 실행은 0.88GiB였다. 일반 u2net 측정에는 잠깐 미리보기 프로세스가 겹쳐 소량의 상향 오차가 가능하다. LanPaint는 별도 이미지 모델이 아니라 FLUX에 적용한 샘플링 방식이다.

## 함께 운영한 모델의 GPU 프로세스 관측값

다음 세 모델은 이번 이미지 실험처럼 단독 실행의 호스트 추가 피크를 측정하지 않았다. 따라서 위 표와 합산하거나 직접 비교할 수 없다. 9월 9일 운영 상태 스냅샷 두 건의 GPU 프로세스 관측 범위다.

| 모델 | GPU 프로세스 메모리 관측값 | 범위의 의미 |
|---|---:|---|
| Qwen3.8 Flash Next / SGLang | 84.76~90.70GiB | 가중치·KV 예약 포함, 두 운영 시점의 스냅샷 |
| Magpie TTS | 1.27~1.52GiB | GPU 프로세스 상주 관측, 음성 생성 중 피크 미측정 |
| Nemotron ASR | 약 1.16GiB | GPU 프로세스 상주 관측, 인식 중 피크 미측정 |

이 값들은 CPU 메모리와 swap을 포함한 모델별 총 사용량이 아니다. LLM은 context 65,536, max_running_requests 2, mem_fraction_static 0.79 설정이며 자동 산정 KV 풀은 재시작 시 달라질 수 있다.

## 해석

1024 이미지 작업은 DreamLite Mobile과 SANA가 추가 약 8GiB 수준이었다. Mobile-O는 512에서 작업 추가량이 6GiB 미만이지만 로딩 때 7.24GiB가 필요했다. Bonsai 공식 CUDA 경로는 1024에서 20.24GiB로 메모리 이점이 나타나지 않았다. 배경 제거만 수행하면 CPU rembg 두 모델이 추가 1GiB 미만이었다.

생성, 참조 편집, 마스크 편집, 배경 제거는 기능이 다르다. 메모리 수치만으로 서로 대체 가능하다는 뜻은 아니다. 한 사진·프롬프트의 실측이며 모든 입력의 요구량이나 동시 실행 피크를 보장하지 않는다.

상세 기록: [FLUX / rembg](image-memory-flux-rembg-2026-09-09.md), [SANA / Mobile-O / FLUX](image-memory-sana-mobileo-2026-09-10.md), [Bonsai / DreamLite](image-memory-bonsai-dreamlite-2026-09-10.md).
