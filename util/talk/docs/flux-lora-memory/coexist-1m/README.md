# 1M Qwen + Flux + ASR + TTS 공존 검증

2026-09-21, gx10-2 / GB10. TP1 Huginnfork NVFP4 checkpoint,
`dgx-sglang-qwen38-qad:sm121-v3`, context/KV capacity 1,048,576,
FP8 KV, concurrency 1, MTP 0, native SGLang GDN 사용.
Flux는 `sparktalk-flux2-paint:trial6`, pinned memory 비활성화.

실제 1,048,435 입력 토큰을 처리하고 앞/뒤 표식 `MAPLE, COMET`을
정확히 반환했다. 1,104.6초, 정상 stop, retraction 0.
전체 문맥의 일반 품질을 보증하는 벤치마크가 아니라 용량과 표식 회수 시험이다.
같은 prefill 동안 1024 이미지 생성/인페인트/물체 제거와 ASR/TTS 요청을
겹쳐 모두 성공했다. 851,968 KV 토큰 시점에도 추가 인페인트/음성 요청이 성공했다.
최소 MemAvailable은 4.973GiB였다. 상세 요청 시간과 메모리 요약은 동봉 JSON 참조.

MTP 비활성화 + 기존 B12X GDN 조합은 130,925 토큰 시험에서 반복 문자를
출력하여 채택하지 않았다. native GDN으로 변경 후 동일 시험과 1M 시험을 통과했다.
다른 모델이 상주한 채 Qwen을 시작하면 SGLang이 실제 KV 용량을 844,992로
줄일 수 있었으므로, Talk는 `/server_info`의 실제 용량도 확인하도록 보완했다.
공식 LIL 원본에 대한 별도 1M 재벤치마크나 MTP 속도 비교를 수행한 결과는 아니다.

Talk에는 MTP 0 / native GDN / Qwen 예산 97GiB를 반영했다.
Flux 13 + ASR 1.3 + TTS 1.2를 합한 예상량은 112.5GiB이며 최소 여유 4GiB를
유지한다. 더 큰 사용자 예약값은 낮추지 않는다. 전체 서비스가 이미 정상인
세트 시작은 새 할당이 없는 것으로 처리한다. CPU 익명 상주량과 GPU 상주량을
반영하되 회수 가능한 파일 캐시를 익명 메모리처럼 계산하지 않는다.

원시 샘플과 서버 로그는 `/tmp/qad-coexist/`에 보관했다.

## Flux 전 모듈 메모리 점검

위 장문 검증 후 Qwen/ASR/TTS를 유지하고 이미지 요청 20회를 실행했다.
생성, 참조 편집, outpaint/object-remove/background-cleanup LoRA, LanPaint,
CPU rembg, LoRA→rembg 결합, LoRA 3종 × 3회 전환, 기본 편집 복귀를 포함한다.
모든 요청이 성공했고 PNG 크기/디코딩, rembg alpha 0~255를 확인했다.
이 단계의 최소 MemAvailable은 5.094GiB, 최대 swap 사용량은 0.167GiB였다.
이 단계에서는 새 LLM/음성 요청을 겹치지 않았으며, 앞 절의 동시 실행 시험과 구분한다.

- diffusion 본체, text encoder, VAE 각각 모델 객체 하나. LoRA ModelPatcher는
  본체 객체를 공유했다. 일반 편집 복귀 후 LoRA 로더/패치가 제거됐다.
- LoRA 파일당 CPU storage 76,021,760 bytes (72.5MiB). 캐시와 적용 adapter가
  같은 storage를 참조한다. 전역 storage pointer 중복 제거 시 한 번만 집계된다.
  68개 적용 패치의 prepared tensor는 요청 완료 후 0 bytes였다.
- 세 모델의 dynamic host buffer 네 종류 모두 size/pinned_bytes 0,
  전체 pinned allocation 0. 과거 줄였던 약 6GiB staging 중복이 재발하지 않았다.
- 반복 주기별 CPU anonymous resident는 2.382 / 2.374 / 2.385GiB.
  일반 편집 최초 2.262 → 최종 2.369GiB: 초기 캐시 증가가 있지만 반복 횟수에
  따른 지속 증가를 관측하지 않았다. 이 유한 반복 시험이 모든 누수를 배제하지는 않는다.
- CPU 가중치 logical storage 약 6.189GiB는 실제 상주 anonymous 양이 아니다.
  파일 매핑/회수 가능한 캐시가 포함된다. 텍스트 인코더 safetensors 매핑은
  관측 시 논리 3,758,024KiB 중 RSS 110,384KiB, anonymous/dirty 0이었다.
  CPU offload는 여전히 사용하며 통합 메모리에서 별도 RAM 용량을 얻는 것은 아니다.
- GPU 가중치는 Aimdo의 동적 VRAM 캐시에도 있으므로 Torch allocated만으로
  총량을 판단하지 않았다. NVML, loaded_size, host anon/file, MemAvailable을
  함께 기록했다. GPU 가상 주소 예약 크기를 실제 할당량으로 계산하지 않았다.
- rembg는 요청별 CPU worker이며 종료 후 프로세스가 남지 않았다.
  LanPaint 종료 후 별도의 지속 모델 복사본도 관측하지 않았다.

따라서 GPU-only 강제, 매 요청 전체 모델 unload, 추가 LoRA 강제 해제는 적용하지 않았다.
현재 CPU/GPU 동적 로딩과 pinned-memory 비활성화를 유지한다. 측정한 범위에서는
추가로 제거할 거대한 staging 복사본이나 누적 LoRA가 발견되지 않았다.

loopback ComfyUI에만 임시 진단 노드를 설치하여 storage/adapter/host buffer를
읽었다. quantized tensor는 dequantization을 호출하지 않고 `_qdata`와 scale
metadata를 집계했다. 첫 반복 시험은 LoRA loader storage를 집계했으며, 마지막
보완 시험은 WeightAdapter 내부 weights까지 추적해 공유 여부를 확인했다.
진단 노드는 시험 후 제거하고 Flux를 재시작했다. 제품 소스/이미지는 변경하지 않았다.
수치 요약은 `flux-audit-results.json`, 원시 inventory/샘플/PNG와 진단 코드는
`/tmp/flux-memory-audit/`에 있다.
