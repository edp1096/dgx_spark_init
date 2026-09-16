# Huihui GGUF 변경량을 적용한 NVFP4 파생 체크포인트

로컬 출력: `~/.cache/huggingface/edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4` (초기 검증 경로는 호환 링크로 유지)

공식 Huihui NVFP4 릴리스가 아니다. Huihui BF16 원본을 확보한 것이 아니라, 동일한 GGUF 양자화 형식의 일반판과 Huihui판을 비교하여 **변경된 Q8 텐서의 차이**를 고정밀 원본에 이식한 파생 모델이다. 원본 체크포인트는 보존했다. 이후 요청에 따라 compose·Talk의 TP1·TP2 모델 ID를 새 이름으로 전환했다.

## 입력과 전체 비교

| 입력 | 고정 revision |
|---|---|
| [Unsloth GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) UD-Q4_K_XL | `38bb39ee97821de2c9009abb7e93950eec396e66` |
| [Huihui GGUF](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF) UD-Q4_K_XL | `7e3bfc316b880fefeb049596f11c49d6a18e05fb` |
| [RadixArk NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) 기반 체크포인트 | `7b719225242aacd3dbd3f9407468c2ee9a9d2594` |
| [Qwen BF16 원본](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) 추가 expert 텐서 | `de4b8e4d43b917e7706784d8bb445c9af86a3540` |

GGUF 본체와 vision projector를 포함한 **1,558개 텐서 전체 바이트 SHA256**을 비교했다. 앞부분 표본 비교가 아니다. 101개만 달랐고 모두 Q8_0이다.

- shared expert down: 48개
- GDN out: 36개
- full attention output: 12개
- routed expert down: 5개 레이어(2, 4, 30, 46, 47), 각 512 experts
- 그 외 expert, PLE/ngram, vision 텐서는 동일

자세한 원본 오프셋·형식·해시: [audit.json](audit.json).

## 변환

96개 BF16 텐서는 `원본 BF16 + DQ(Huihui Q8) - DQ(Unsloth Q8)`를 BF16 ties-to-even으로 반올림한다. GDN out은 GGUF의 tiled V-head 순서를 HF의 grouped 순서로 먼저 복원한다. K heads=16, V heads=48, head dim=128이다. [llama.cpp의 해당 변환](https://github.com/ggml-org/llama.cpp/blob/391fac16460f15233a7740550d858ac96df3419d/conversion/qwen.py)을 역으로 적용했다.

routed expert down 5개 레이어는 공식 BF16 텐서 범위만 HTTP Range로 추가 다운로드했다(총 8,388,608,000바이트). SHA256과 HTTP 206 Content-Range를 확인한 다음 다운로드 완료 후 변환했다. 원본 BF16에 같은 변경량을 더한 뒤 ModelOpt NVFP4로 재양자화했다. NVFP4를 먼저 복원해 재양자화하지 않는다.

양자화기는 NVIDIA ModelOpt 소스 `87c9f8cf83021957d1a1a575c90c9a4eaaf7ef0c`의 `NVFP4QTensor.quantize`, block size 16, FP8 block scale + FP32 global scale이다. CPU 전용 컨테이너(6GiB 제한)에서 처리했다. 원본 activation input scale은 유지했다. **새 activation calibration을 수행한 모델은 아니다.**

원본 scale로 검사한 expert 20개는 원본 NVFP4와 바이트 단위로 일치했다. CPU에서 global scale을 다시 계산하면 일부가 약 1e-7 상대 차이를 보이며 양자화 경계값에 영향을 준다. [quantizer-probe.json](quantizer-probe.json)에 수치를 남겼다.

## 산출물과 검증

- 206개 샤드, 논리 크기 약 135.2GB
- 24개 샤드만 독립 파일로 작성, 추가 저장 약 23.1GB
- 182개 샤드는 원본의 immutable blob과 하드링크로 공유
- 원본 NVFP4 파일, PLE/ngram, vision, gate/up weights, activation scales 보존
- 변환 대상 밖의 모든 텐서 동일성, 원본 blob SHA256, dtype/shape/offset/header, 후보 전체 샤드 SHA256 검사
- 결과: [verification.json](verification.json), 변환 수치: [transfer-manifest.json](transfer-manifest.json)

원본 BF16과 일반 GGUF Q8 사이의 상대 L2 오차는 dense에서 약 0.56~0.74%, expert에서 약 0.55%다. expert의 새 NVFP4 가중치 상대 L2 양자화 오차는 약 9.47~9.50%다. **이 수치는 모델 정답률 저하율이 아니다.** Q8 양자화 잔차가 변경량에 포함되므로 Huihui BF16 원본과의 동일성을 주장하지 않는다.

### 실행 검증 완료 (2026-09-15)

60번 단독 TP1, 64K 문맥 설정, SGLang `sm121-vocab1`, 메모리 비율 0.79, NEXTN 3 steps / 4 draft tokens, thinking off로 두 모델을 동일하게 실행했다. 초기 단독 비교 당시에는 Talk 설정을 변경하지 않았다.

| 검사 | RadixArk 일반판 | 새 파생 NVFP4 |
|---|---:|---:|
| 산술·JSON·정렬·색상·키 회수 정답 | 5/5 | 5/5 |
| 생성한 Python 함수 실행 검사 | 4/4 | 4/4 |
| 도구 함수명·JSON 인자 | 통과 | 통과 |
| 한국어 존댓말·번역·창작 수동 확인 | 정상 | 정상 |
| tg 중앙값 (256토큰 × 3회) | 28.23 tok/s | 28.51 tok/s |
| ttft 중앙값 (위 요청) | 0.260초 | 0.258초 |
| 최소 시스템 가용 메모리 | 21.53GiB | 21.49GiB |
| OOM / 메모리 안전장치 발동 / 재부팅 | 없음 | 없음 |

생성 요청은 입력 41토큰, 출력 256토큰으로 고정했다. tg는 첫 내용 수신 이후 클라이언트 스트리밍 시간으로 계산했다. 일반판 3회는 26.72/28.23/48.53, 후보는 28.51/27.34/28.96 tok/s다. 중앙값 차이는 약 +1%지만 출력 내용과 MTP 적중률에 따른 편차가 있으므로 속도 우열을 입증한 결과로 해석하지 않는다.

키 회수 입력은 6,037토큰이며 두 모델 모두 정답이다. 64K는 서버 문맥 설정이고, 실제 64K 전체 입력이나 TP2/1M를 이번에 검증한 것은 아니다. 기본 기능 검사이지 종합 품질·검열 해제 평가가 아니다. 원본 Huihui BF16과의 동일성도 주장하지 않는다.

초기 시험에서 메모리 비율 0.70은 hybrid/MTP state cache 예산이 0.55GB 부족해 기동에 실패했다. 호스트 OOM은 아니었고 기존 사용 비율 0.79로 재기동하여 양쪽 모두 통과했다. 이 재기동과 각 모델의 긴 적재 시간 때문에 최초 예상 시간보다 오래 걸렸다.

초기 단독 비교의 테스트 컨테이너와 watchdog은 정리했고 당시 60번 가용 메모리는 약 117GiB로 돌아왔다. 이후 메인 전환 검증은 아래와 같이 별도로 진행했다.

결과: [runtime-summary.json](runtime-summary.json), [quality-review.json](quality-review.json), [일반판 응답](radix-runtime.json), [파생 모델 응답](candidate-runtime.json). 압축된 원시 서버·메모리 기록은 `runtime-evidence/`에 보존했다.

## 재현 도구

`../` 아래:

- `audit.py`: 두 GGUF 전체 텐서 비교
- `fetch_source.py`: 필요한 공식 BF16 텐서 범위 다운로드·해시 확인
- `build.py`: 96개 BF16 변경량 이식, GDN 순서 복원
- `run_quant_container.py --probe-only`: 원본 NVFP4 재현 검사
- `run_quant_container.py`: expert 5개 레이어 변경량 이식·재양자화
- `verify.py`: 변경 대상·비대상, 원본과 결과 해시 검사
- `launch_validation.py`, `evaluate.py`: 분리된 worker 실행·요청 검사

모델 경로가 이미 있으면 덮어쓰지 않는다. `.partial` 디렉터리는 완성 모델로 사용하지 않는다. 수정 샤드는 원본과 별도 inode인지 검사한 후 작성한다. 다운로드 완료 확인 전에는 변환을 진행하지 않는다.

## 메인 모델 전환과 공개 조건

모델 ID: `edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4`.

- 정식 SGLang compose의 TP1·TP2, Talk 내장 TP1 compose·TP2 recipe·카탈로그를 모두 전환했다.
- 카탈로그 revision 4는 이전 내장 모델 ID를 갱신하되 사용자 지정 이름·주소·문맥·다른 모델을 보존한다.
- TP2는 1,048,576 문맥 설정으로 기본 검사와 별도 66,038토큰 입력의 정확한 키 회수를 통과했다. 실제 1M 전체 입력 시험은 아니다.
- TP1은 실제 운영 옵션 `ko64k`, 동시 요청 상한 2, 65,536 문맥 설정으로 기본 검사를 다시 통과했다. 최소 가용 메모리는 18.46GiB였다.
- TP1 전체 세트는 FLUX 최대 13GiB를 함께 예약하면 사전 검사에서 여유가 3.5GiB로 계산돼 4GiB 기준을 충족하지 못했다. 안전 기준을 낮추지 않았고, TP1 LLM 검증에서는 FLUX를 내렸다. ASR·TTS는 유지했다.
- 모든 주변 모델을 함께 쓰는 실제 메인·기본 세트는 `flash-next-tp2`로 선택했다.

실제 메인 서비스와 Talk 대화까지 통과해야 `release-validation.json`을 생성한다. 공개 도구는 이 파일과 증거 해시를 확인하기 전에는 저장소를 생성하거나 업로드하지 않는다. `publish.py`는 로컬 경로·호스트 정보를 담은 내부 보고서를 제외하고 모델·설정·라이선스·정리한 출처만 업로드한다. 원격 206개 샤드의 SHA256이 일치해야 업로드 중 표시를 제거하고 완료로 기록한다.

자료: [TP1 운영 검사](tp1-main-runtime.json), [TP2 검사](tp2-runtime.json), [TP2 긴 입력](tp2-long-input.json), [TP2 메모리](tp2-memory.json), [공개용 모델 카드](MODEL_CARD.md). 공개 완료 여부와 revision은 `publication-status.json`에 기록한다.

공개 완료: [edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4](https://huggingface.co/edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4), revision `40f09f531da577a4fcfbd6b368e7b6ebacde403e`. 익명 접근·원격 206개 샤드 해시·공개 메타데이터 해시와 현재 Talk 메인 상태를 확인했다. 결과는 [completion-check.json](completion-check.json)에 있다.
