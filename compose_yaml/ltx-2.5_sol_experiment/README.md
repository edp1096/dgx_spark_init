# LTX-2.5 Sol Engine experiment

DGX Spark에서 LTX-2.5 distilled NVFP4의 Stage 2에 NVIDIA Sol-Attn과
Exact AdaLN을 적용해 비교하는 일회성 벤치마크입니다. 운영 API를 변경하지 않고
`media-ltx-models` 볼륨의 기존 모델만 읽습니다.

Dense/Triton 대조군은 Sol Engine
`d0c0a4685ab5dc2336d18b7213d85f13def92418`에 고정했습니다. SM121 이미지는
`edp1096/Sana`의 `809638b437f49bdda969ebf568d12b8e91806c98`을 직접 checkout합니다.
이 커밋은 NVIDIA의 SM120 warp-MMA/TMA CuTe 레시피를 별도 `sm121` 소스 패키지로
분리하고 `sm_121a`로 컴파일합니다. Docker 빌드 중 소스를 생성하거나 치환하지
않습니다. 아직 SM121 전용 타일·파이프라인을 재탐색한 커널은 아니며,
아키텍처별 구현을 분리한 검증용 출발점입니다.

```bash
docker compose build

# Dense 대조군
docker compose run --rm benchmark --dense --no-exact-adaln

# Sol-Attn + Exact AdaLN
docker compose run --rm benchmark

# 1920x1088, 121프레임
docker compose run --rm benchmark \
  --width 1920 --height 1088 --frames 121 \
  --output /output/1080-sol.mp4

# SM121 CuTe 정확도/마이크로벤치
docker compose --profile sm121 build
docker compose --profile sm121 run --rm sm121-probe \
  --tokens 32640 --heads 32 --warmup 2 --iterations 10

# SM121 CuTe 1920x1088 전체 생성
docker compose --profile sm121 run --rm sm121-benchmark \
  --width 1920 --height 1088 --frames 121 --repeats 2 \
  --output /output/1080-cute-sm121.mp4
```

현재 벤치마커는 같은 파이프라인 프로세스에서 두 번 생성합니다. 첫 실행은 워밍업이고
두 번째 값을 비교에 사용합니다. 결과 영상과 JSON은 `output/`에 기록되며 Git에는 포함하지
않습니다. 아래 768x512 Sol 값만 벤치마커의 반복 실행 기능을 추가하기 전에 영속 Triton
캐시를 사용해 별도 측정한 워밍 값입니다.

## 2026-08-26 측정

| 최종 출력 | 구성 | 반복 생성 | Stage 2 denoise | CUDA 피크 |
|---|---|---:|---:|---:|
| 768x512 / 121f | Dense | 38.48초 | 9.42초 | 23.13 GiB |
| 768x512 / 121f | Dense + Exact AdaLN | 42.01초 | 7.84초 | 23.13 GiB |
| 768x512 / 121f | Sol + Exact AdaLN | 46.57초 | 9.93초 | 23.10 GiB |
| 1920x1088 / 121f | Dense | 150.16초 | 69.90초 | 23.13 GiB |
| 1920x1088 / 121f | Sol | 133.59초 | 52.64초 | 23.13 GiB |
| 1920x1088 / 121f | Sol + Exact AdaLN | 126.25초 | 43.69초 | 23.13 GiB |
| 1920x1088 / 121f | SM121 CuTe Sol + Exact AdaLN | **119.67초** | **38.90초** | 23.13 GiB |

기본 해상도에서는 라우팅과 Exact AdaLN 교체 비용이 더 커서 사용하면 안 됩니다. 현재
측정에서는 Stage 2 토큰이 32,640개인 1920x1088부터 조건부 적용 가치가 확인됐습니다.

### SM121 CuTe 검증

| 토큰 / 헤드 | CuTe SM121 | Triton | CuTe 가속 | 상대 L2 | Cosine |
|---|---:|---:|---:|---:|---:|
| 1,024 / 4 | 0.111 ms | 0.110 ms | 0.99x | 0.000037 | 0.9999999 |
| 6,144 / 32 | 2.307 ms | 3.772 ms | 1.64x | 0.001822 | 0.9999983 |
| 32,640 / 32 | 40.258 ms | 74.522 ms | **1.85x** | 0.001481 | 0.9999989 |

동일 seed의 Triton Sol과 SM121 CuTe 전체 영상 비교는 SSIM 0.9275, PSNR
31.77 dB였습니다. 프레임을 직접 확인했을 때 장면·동작·구조는 동일하고 작은 조명 및
위치 차이만 보였습니다. 1080p 워밍 실행 기준 전체 시간은 5.2%, Stage 2 denoise는
11.0% 줄었고 메모리 피크 증가는 없었습니다.
