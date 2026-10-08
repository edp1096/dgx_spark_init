# QWIM / MiniMax H3 DiT 상주 워커

두 DiT를 하나의 프로세스에 유지하면서 텍스트 인코더와 VAE를 사용하는 단계가
끝날 때 해제하는 실험용 워커다. Talk의 이미지 API나 모델 카탈로그를 대체하지 않는다.
일반 텍스트 생성 경로를 검증하며 참조 이미지 편집, 이미지→영상, 여러 작업의
동시 실행은 구현하지 않았다.

## 적재 정책

- QWIM NVFP4 DiT와 H3 NVFP4 DiT는 정적 전체 적재로 유지한다. 메모리 관리자가
  이 두 모델을 퇴거 대상으로 선택하지 못하게 하고 매 단계 뒤 실제 적재량을 검사한다.
- `aux_loading: dynamic`은 텍스트 인코더와 VAE만 Comfy AIMDO 적재 경로를 사용한다.
  보조 모듈의 소유권은 해당 함수에만 두며 함수가 끝나면 참조, 순환 참조,
  Comfy 모델 목록, PyTorch 할당 캐시를 정리한다. 폐기할 모듈을 CPU로 복사하지 않는다.
- 프롬프트 인코딩 결과는 모델과 정확한 프롬프트를 키로 최대 두 건 재사용한다.
  생성 결과는 캐시하지 않는다. 같은 시드의 반복 요청도 모든 샘플링 단계를 실행한다.
- 영상 VAE와 오디오 VAE도 순서대로 사용하고 각각 해제한다.
- 동적 할당기의 작업용 메모리와 CUDA 컨텍스트 등은 모델 적재량과 별도로 존재하므로
  `torch.cuda.memory_allocated()` 외에 호스트 MemAvailable 및 NVML 측정이 필요하다.

## 고정된 검증 조건

- QWIM: Qwen Image 2.1 UC NVFP4, Qwen3-VL 8B W4A8, BF16 RGBA VAE,
  INT8 참조 캐시, 1024×1024, Euler/simple 40스텝, CFG 1.
- H3: FL2VA pruned NVFP4, Qwen3-VL 4B FP8 + ClipProj v3.1,
  INT8 영상 VAE + FP32 오디오 VAE, 864×480, 124프레임/24fps,
  res_multistep/simple 20스텝, BasicGuider.
- ComfyUI `5c460d8172fe30761ff67c0df3d5643bb74e0d70`, PyTorch 2.13,
  comfy-kitchen 0.2.37, comfy-aimdo 0.5.5, PyTorch SDPA.
- 검증 이미지: `sparktalk-mmh3-swap-test:20261007`,
  ID `sha256:a7783884107c4c499f19d7fb2dd3e7171a51d2d8d181a81c1f87838702b9e68d`.
  기존 QWIM 이미지에 고정 리비전의 ComfyUI-ClipProj를 추가한 이미지다.

## 워커 입출력

컨테이너의 `/job/settings.json`에 가중치 위치를 지정한다.
매 실험에는 빈 작업 디렉터리를 사용한다.
`settings.example.json`의 `/models` 경로는 실제 읽기 전용 마운트에 맞춰 조정한다.
기존 HF 캐시를 가리키는 심볼릭 링크를 쓰면 `/hf` 마운트도 필요하다.
워커는 `/opt/ComfyUI`와 ClipProj가 있는 위 이미지에서 `python3 /runner/worker.py`로
실행한다. `/runner`는 이 디렉터리의 읽기 전용 마운트다.

적재 완료 시 `/job/ready.json`이 생긴다. 호스트에서는 다음과 같이 요청한다.

```bash
python3 submit.py /path/to/job qwim 'A red ceramic teapot on a wooden table' --seed 42
python3 submit.py /path/to/job h3 'A red toy car rolls across a wooden table' --seed 42
```

`requests/*.json`을 원자적으로 생성하면 워커가 직렬로 처리한다.
결과와 단계별 시간은 `results/`, 실제 PNG/MP4는 `output/`, 적재량과 진행 기록은
`events.jsonl`에 남는다. `/job/stop`을 생성하면 현재 작업이 끝난 뒤 종료한다.
요청 오류 시 오류 결과를 기록하고 프로세스를 종료한다. 클라이언트 시간 초과는
큐의 작업을 취소하지 않는다.

호스트의 실행 관리는 별도다. 실측 러너는 기존 QWIM의 유휴 상태를 확인한 뒤
중지하고, 실험 컨테이너에 36GiB cgroup 상한을 적용했다. MemAvailable이 5GiB
미만이면 실험 컨테이너만 중단하며, 성공·실패 모두 기존 QWIM을 복구하고 실제
이미지 생성으로 확인한다. LLM·ASR·TTS의 컨테이너와 설정은 유지한다.
이미 실행 중인 전체 QWIM 위에 이 워커를 중복 적재하는 구성으로 검증하지 않았다.

## 2026-10-07 실측

LLM·ASR·TTS를 상주시킨 DGX Spark에서 H3 → QWIM을 세 차례 반복했다.
두 DiT의 최초 적재/프로세스 준비는 46.1초였으며 아래 생성 시간과 별도다.
이후 전환 시 프로세스를 재시작하거나 DiT를 다시 읽지 않았다.

| 조건 | H3 | QWIM |
| --- | ---: | ---: |
| 첫 프롬프트 | 204.7초 | 33.9초 |
| 같은 프롬프트 재사용 | 197.8초 | 27.8초 |
| 프롬프트 변경, 인코더 재적재 | 200.9초 | 32.8초 |

앞선 전체 서비스 교체 실측에서는 H3 → QWIM 첫 이미지까지 74.3초가 걸렸다.
이번에는 33.9초로 약 54% 단축됐다. H3는 앞선 두 번째 생성에서 관측한
412.9초 지연이 재발하지 않았고, 세 번 모두 샘플링 약 180초를 유지했다.
앞선 H3 첫 생성은 207.5초였으므로 본래의 연산 시간 자체를 절반으로 줄였다는
뜻은 아니다. 모든 비교는 위 고정 프로필에 한정하며 동시 추론 부하는 가하지 않았다.

보조 모듈까지 정적으로 복사하는 첫 구현과 비교하면 H3 인코더 적재·처리는
34.0 → 5.9초, QWIM은 43.9 → 5.4초, H3 영상 VAE 적재·변환은
33.2 → 15.0초였다. 파일 캐시 등 시작 상태의 영향도 포함된 실측값이다.

생성 중 관측한 MemAvailable 최저치는 12.8GiB이며 보호장치 중단은 없었다.
유휴 모델 목록에는 매번 두 DiT만 남았다. 같은 입력의 QWIM은 기존 PNG와
전체 픽셀이 일치했고, H3는 비교한 0·60·120번째 프레임이 일치했다.
세 MP4 모두 124프레임과 AAC 오디오를 끝까지 디코딩했다.

실험 후 기존 QWIM을 복구하고 실제 이미지 생성으로 확인했다. LLM·ASR·TTS의
컨테이너 ID와 PID, Talk 설정은 실험 전후 동일하다. 최적화 워커의 Talk API 연결은
이 변경의 범위에 포함하지 않는다.

정확한 단계별 수치와 검증 결과는 [benchmark-results.json](benchmark-results.json)에 있다.
전체 로그·출력·복구 러너는
`~/.cache/model-download-jobs/mmh3-hybrid-20261007/`에 보존했다.
