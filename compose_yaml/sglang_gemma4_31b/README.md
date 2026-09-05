# SGLang Gemma 4 31B NVFP4 + DFlash

DGX Spark에서 다음 조합을 SparkTalk용 OpenAI 호환 API로 실행한다.

- Target: `lyf/Huihui-gemma-4-31B-it-abliterated-v2-NVFP4`
- Draft: `z-lab/gemma-4-31B-it-DFlash`
- API: `http://127.0.0.1:8000/v1`
- Context: 64K
- Target/draft KV cache: FP8 E4M3
- Initial DFlash verify window: 7
- Attention backend: Triton (이 SGLang 커밋의 Gemma 4 지원 backend)
- Reasoning/tool parser: Gemma 4

체크포인트의 `compressed-tensors` NVFP4 메타데이터를 자동 인식하므로
`--quantization modelopt_fp4`를 강제로 지정하지 않는다. DFlash는 공식 BF16 target용으로
학습됐으므로 이 abliterated NVFP4 파생 target과의 품질 및 acceptance는 실측 대상이다.

현재 SGLang 커밋은 이 체크포인트가 BF16로 제외한 vision tower의 weight 이름을 fused
vision layer에 올바르게 매핑하지 못한다. 잘못 NVFP4 처리된 vision layer가 실패하므로
이미지 입력은 지원한다고 간주하지 않는다. 텍스트 경로 검증을 위해 서버의 VLM 자동
warmup은 건너뛴다.

## 실행

```bash
cd ~/workspace/dgx_spark_init/compose_yaml/sglang_gemma4_31b
docker compose up -d --build
docker logs -f sglang-gemma4-31b
```

첫 실행에는 target과 draft 다운로드가 필요하다. 다음 문구가 나오면 준비된 것이다.

```text
The server is fired up and ready to roll!
```

상태와 모델 이름을 확인한다.

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
```

종료한다.

```bash
docker compose down
```

SparkTalk 설정의 API endpoint는 `http://127.0.0.1:8000`, 기본 모델은
`Huihui-gemma-4-31B-it-abliterated-v2-NVFP4`로 지정한다.
Gemma 4는 Qwen 계열처럼 단계별 reasoning effort를 받지 않고 `enable_thinking`을
켜거나 끈다. SparkTalk 설정에서 모델 유형을 `gemma4`로 명시하면 해당 입력을
Thinking 켜짐/꺼짐으로 표시하고 SGLang의 `chat_template_kwargs`로 변환한다.
`--enable-strict-thinking`을 사용하므로 Thinking 요청에
`custom_params.thinking_budget`을 전달하면 지정한 토큰 수에서 생각을 종료하고
최종 답변으로 넘어간다. SparkTalk의
`model.thinking_budget`이 이 값을 관리하며 `0`은 제한하지 않는다.
현재 SGLang은 전역 상한이 `-1`이면 요청별 예산이 있어도 strict token filter를
초기화하지 않으므로 compose에서 `SGLANG_MAX_THINK_TOKENS=65536`을 설정한다.
요청별 값이 이 상한을 덮어쓰며, 65536은 이 프로필의 context 상한이라 사실상
무제한 기본값으로 동작한다.

체크포인트에 포함된 구형 템플릿 대신 MiaAI 레시피의 최신 Google Gemma 4
canonical template을 이미지에 고정한다. OpenAI 형식의 연속 도구 호출,
tool response 연결 및 reasoning content 순서를 올바르게 처리한다.

### Thinking budget 실측

Huihui NVFP4 + DFlash block 7, temperature 0에서 같은 확률 추론 문제를 측정했다.
강제 종료 토큰이 하나 포함되므로 실제 reasoning 집계는 설정값보다 1토큰 많았다.

| 예산 | reasoning | 완료 토큰 | 시간 | 처리량 | 종료 | 결과 |
|---:|---:|---:|---:|---:|---|---|
| 64 | 65 | 1536 | 44.80초 | 34.29 tok/s | length | 제어 토큰 노출 및 답변 중복 |
| 256 | 257 | 1276 | 45.65초 | 27.95 tok/s | stop | 정답·형식 정상 |
| 512 | 513 | 1474 | 54.67초 | 26.96 tok/s | stop | 정답·형식 정상 |
| 무제한 | 1142 | 1536 | 50.81초 | 30.23 tok/s | length | 최종 답변 도중 출력 한도 도달 |

별도의 동일 256-token 출력 비교에서 무제한은 45.11 tok/s, DFlash acceptance
53.3%였고 64-token 강제 제한은 29.51 tok/s, acceptance 46.7%였다. 너무 작은
예산은 문장 중간에서 채널을 바꾸어 품질과 speculative acceptance를 함께 낮춘다.
따라서 SparkTalk 기본값은 512이며 최소 256 이상을 권장한다.

초기 프로필은 동작 검증을 위해 Torch compile을 사용하지 않는다. 텍스트와 도구 호출,
DFlash acceptance를 확인한 뒤 compile 및 block size 7/11/16을 비교한다.
이 DFlash draft는 공식 Google target용으로 학습됐다. 공식 NVIDIA NVFP4 target에서의
첫 실측은 384-token 단일 요청에서 baseline 약 6.42 tok/s, DFlash 약 9.0 tok/s로
DFlash가 약 40% 빨랐다. Huihui target과 같은 프롬프트로 비교한 결과는 다음과 같다.

| 요청 | NVIDIA NVFP4 + DFlash | Huihui NVFP4 + DFlash |
| --- | ---: | ---: |
| 한국어 문학 문체 | 6.57 tok/s | 9.61 tok/s |
| Go 코딩 | 13.24 tok/s | 20.20 tok/s |
| 확률 추론 | 17.70 tok/s | 25.50 tok/s |

Huihui target의 로드된 가중치는 19.66GB로 NVIDIA target의 31.87GB보다 작았다.
문학 요청의 DFlash acceptance는 3~7%로 낮았지만 코딩·추론은 후반 39~51%까지
올라갔다. temperature 0 출력도 baseline과 bit-identical하지 않았으므로 업데이트
후에는 속도뿐 아니라 도구 호출과 출력 품질을 함께 회귀 점검한다.

## DSpark 비교

`Hikari07jp/DSpark-Gemma-4-31B-draft`의 full-vocabulary Markov DSpark를 동일한
Huihui target과 프롬프트로 비교했다. SGLang의 Gemma LM-head 호환 처리를 보완하고
target verify, draft verify, folded greedy/sampling CUDA graph가 모두 활성화된 상태의
결과다.

| 요청 | DFlash | Hikari DSpark |
| --- | ---: | ---: |
| 한국어 문학 문체 | 9.61 tok/s | 6.81 tok/s |
| Go 코딩 | 20.20 tok/s | 7.10 tok/s |
| 확률 추론 | 25.50 tok/s | 7.16 tok/s |

DSpark acceptance는 대부분 0~1%, 평균 accepted length는 약 1.0~1.2에 그쳤다.
이 DSpark의 Markov 보정은 NVIDIA 공식 target에서 학습됐기 때문에 Huihui abliterated
target과 잘 맞지 않는 것으로 판단해 운영 구성에서는 제외했다. RedHatAI 계열 DSpark는
입력 262K/output 32K의 비대칭 draft vocabulary를 사용하는데 현재 SGLang DSpark는
동일 vocabulary만 지원하므로 비교 대상에서 제외했다.
