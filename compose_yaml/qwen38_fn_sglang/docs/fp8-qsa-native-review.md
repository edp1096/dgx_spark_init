# QSA · SM121 네이티브 FP8 조사

결론: GB10은 FP8 MMA를 실행한다. 현재 서비스 이미지의 QSA에는 바로 사용할 수 있는 네이티브 FP8 경로를 확인하지 못했다. 운영 설정은 BF16 그대로다.

## 현재 이미지에서 확인한 사실

- QSA decode의 `qwen38_qsa_sm121_varlen`은 BF16 D=256, 12:1 GQA, TP1 24Q/2KV 또는 TP2 12Q/1KV를 요구한다. 단순 FP8 KV 설정으로는 이 계약을 만족하지 못한다.
- 원본 sparse prefill은 Q와 K/V를 `tl.dot`에 전달한다. 조사한 공개 FP8 지원 패치들은 K/V를 Q의 dtype으로 변환한다.
- 모델을 올리지 않은 작은 GPU 시험: FP8 E4M3 입력의 Triton `tl.dot`은 max_num_imprecise_acc=0/32/256 모두 FP16 MMA로 생성됐다. 입력 dtype만 FP8이라고 FP8 연산이라 부를 수 없다.
- 같은 이미지와 실제 SM121에서 inline PTX `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`를 실행했다. 128개 결과가 모두 기대값 32와 일치했다. 하드웨어 명령 지원 확인이며 일반 수치 정확도·속도 검증은 아니다.

[시험 코드·결과](../bench/results/fp8-native-review/result.json). 두 시험 컨테이너는 종료·제거했고 모델을 기동하거나 서비스 이미지를 바꾸지 않았다.

## 공개 후보

- [blake-snc/sm121-kernels](https://github.com/blake-snc/sm121-kernels), 조사 커밋 `e5a28a311bf63c6e2f1a0e7ca5bc84b62b5af33f`: `ptx/attention/templates/fa_fp8_v12c_vt.ptx.in`에서 QK와 PV 모두 FP8 MMA를 사용한다. D=128, V 전치 레이아웃, causal/GQA/varlen 변형을 제공한다. 우리 D=256 QSA의 즉시 교체품은 아니다. 같은 저장소의 D=256 FP8-KV decode는 BF16 Q와 KV 복원 방식이다.
- [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000): sparse prefill의 K/V는 Q dtype으로 변환한다. `FP8.md`의 online FP8 개선은 주로 projection 가중치·활성화이며 QSA 표현을 바꾸지 않는다고 명시한다.
- [Triton 이슈 7188](https://github.com/triton-lang/triton/issues/7188): FP8 tl.dot이 FP16 MMA가 되는 관련 보고. 현재 이미지에서는 직접 생성 명령을 확인했다.

다음 구현에는 D=256의 QSA 타일/선택 KV gather, Q와 softmax 확률의 FP8 스케일링, FP32 누산, MTP 및 CUDA Graph 연결이 필요하다. 작은 커널의 이득과 전체 모델의 이득은 별도로 측정해야 한다. 특히 decode는 Q 토큰 수가 적어 FP8 Tensor Core 활용도가 낮을 수 있으므로 pp/tg 개선을 미리 보장하지 않는다.

## eugr 및 공식 레시피 추가 조사

- eugr/spark-vllm-docker `3e1578b3c898255e7c79532569fd8d194261c3fd`의 `recipes/qwen3.8-flash-next-nvfp4-cluster.yaml`: TP2, FP8 KV, B12X MoE/linear/GDN, 기본 문맥 262144. `--exp-b12x`는 `local-inference-lab/vllm`의 `dev/jovian-judgement`를 선택한다. 이 레시피 자체는 1M 검증 근거가 아니다.
- 위 vLLM 포크 조사 tip `66c293578412417476f842c1da5805d3a3d959a8`: `vllm/models/qwen3_8_flash_next/nvidia/qsa.py`가 별도 b12x QSA를 사용한다.
- b12x 조사 tip `ce419b52681b7922bb0972d4b58b590a3fd005b2`: `b12x/attention/paged/_selected_forward.py`의 `_load_fp8_vector_to_bf16_shared`가 FP8 K/V를 BF16 shared memory로 변환한다. `_tiled_mma_qk`와 `_tiled_mma_pv`는 모두 `MmaF16BF16Op(BFloat16, Float32, (16,8,16))`다. FP8 저장 + BF16 계산이며 네이티브 FP8 Attention은 아니다.
- 이 B12X 경로는 이미 D=256과 선택 위치 기반 paged GQA를 다룬다. 선택한 KV를 캐시에서 직접 읽는 CuTe 구현이므로 현재 gather-then-attention 경로와의 비교 후보로 유의미하다. 이득은 미측정이며 SGLang 이식 시 cache layout·metadata·MTP·graph 계약 검증이 필요하다.
- SGLang 공식 Qwen cookbook은 Spark에 `lmsysorg/sglang:dev-qwen38-next-local`와 `qwen4-main-squashed`를 안내한다. PR36644의 FP8 QSA는 K/V를 Q dtype으로 넓히는 방식이다. 공개 TP8 RTX5090 A/B에서 KV 용량 증가는 크지만 output throughput 차이는 0~0.8% 수준이었다. 이를 우리 TP2 성능으로 환산하지 않는다.
- vLLM 공식 Qwen recipe의 FP8은 모델 체크포인트 정밀도와 구분해야 한다. QSA FP8 KV PR55557은 별도 변경이며, 이것도 저장/읽기 경로 지원이지 QK/PV 네이티브 FP8 계산 증거가 아니다.

소스: [eugr recipe](https://github.com/eugr/spark-vllm-docker/blob/main/recipes/qwen3.8-flash-next-nvfp4-cluster.yaml), [B12X kernel](https://github.com/lukealonso/b12x/blob/ce419b52681b7922bb0972d4b58b590a3fd005b2/b12x/attention/paged/_selected_forward.py), [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next), [SGLang PR36644](https://github.com/sgl-project/sglang/pull/36644), [vLLM cookbook](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next), [vLLM PR55557](https://github.com/vllm-project/vllm/pull/55557).
