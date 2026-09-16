# Ornith 투기적 디코딩 조사 (2026-09-15)

현재 실측: 60번, 공식 NVFP4, vLLM b40673cd0-v3, TP1, 262144,
FP8 KV 32 GiB, chunk1024, b12x MoE, 투기적 디코딩 없음.
워밍업의 norm 입력 크기 오류 수정 후 health 및 계산/한국어/코드/도구/이미지 검사 통과.
한국어 tg 76.74, 코드 77.45 tok/s. 짧은 요청 기능 검사이며 장문/정식 성능 평가와 구분한다.
원본/Huihui 다운로드 및 변환은 별도 진행 중.

## 우선순위

1. 원본 MTP K=1: 추가 다운로드 없는 비교 기준. 기본 활성화하지 말고 AR과 같은 요청으로 비교.
2. Ornith 전용 DFlash2: 단일 Spark NVFP4의 직접 비교 수치가 있어 가장 유력한 가속 후보.
3. 교체 MTP: 원본 head보다 높은 수용률 보고가 있지만 구현 정확성 관련 미해결 보고가 있어 보류.
4. DSpark: Ornith 1.5 35B에 맞는 검증된 공개 draft는 이번 조사에서 찾지 못함.
   Ornith 1.0 9B/35B용 가중치를 그대로 호환된다고 볼 수 없음.

## 확인한 자료

- https://github.com/MiaAI-Lab/Ornith-1.5-35B-A3B-DGX-Spark
  start.sh와 README 확인. vLLM b12x+MTP1, 단일 스트림 86.3 tok/s 보고.
  440 tok/s는 24개 동시 요청 합산. 현재 기본 AR ~77과 동조건 비교가 아니므로 개선율 계산 금지.
  단일 MTP layer가 반드시 draft 1개만 지원한다는 설명은 다른 K=3 구동 사례와 불일치.
- https://github.com/sojufx/Ornith-1.5-35B-A3B-NVFP4-DGX-Spark
  기본 Marlin과 BF16 MTP의 flashinfer_cutlass backend를 분리.
- https://github.com/sfxnz/Ornith-1.5-35B-A3B-NVFP4-DGX-Spark
  MTP3/triton, chunk8192. 서로 다른 벤치마크 수치를 하나의 tg로 합치지 말아야 함.
- https://github.com/ultimatechris/ornith-dflash-sglang
  config_35b.py 확인: BF16, TP2 A100, context4096, block16, Qwen3.5 draft.
  202.5→470.2 tok/s(2.32x)는 우리의 TP1 NVFP4/1M 수치가 아님.
- https://huggingface.co/jzinno/Ornith-1.5-35B-A3B-DFlash2
  공식 Ornith NVFP4 + FP8 KV + Marlin + FlashInfer, 단일 Spark C1:
  AR79.3, NEXTN72.1, Qwen DFlash103.6, Ornith DFlash2 114.2 tok/s(전체 처리량).
  DFlash2는 AR보다44% 높음. TTFT118→145.1ms라 첫 토큰 단축과는 구분.
  전용 draft 6층, 훈련 block16, 평가 draft10, sliding window4096,
  max_position_embeddings262144. 1M 가동/정확성을 입증한 자료는 아님.
  평가 SGLang commit710267dc4c817b38d9965346390cd56b59b54eda.
  로컬 vLLM도 registry의 DFlash2DraftModel→DFlash2Qwen3ForCausalLM 및
  config/vllm.py의 DFlash2 분기 존재 확인. 엔진 전환이 필수라고 단정하지 않음.
- https://huggingface.co/shisa-ai/Ornith-1.5-35B-A3B-MTP
  교체 head만 약1.689GB. BF16 전체 모델을 다시 받을 필요 없이 head-only 후보 존재.
  원본 MTP 부진, 교체 head 수용률 개선, 긴 문맥 DFlash 수용률 저하 보고.
  vLLM0.27.1에서 AR/투기적 모드의 출력 정확성 문제를 저자가 명시.
  Spark/우리 NVFP4+Huihui 조합에서 검증되지 않음.
- https://github.com/r0b0tlab/ornith-15-35b-nvfp4-w4a16-sm121-sglang
  별도 양자화 체크포인트. 235K 입력 MTP NIAH 한계 명시; 우리의 공식 혼합 정밀도와 다름.
- https://github.com/siren2345/Ornith-1.5-35B-A3B-NVFP4-RTX-5090
  vLLM DFlash2 코드 경로 사례. 32GB 5090의 메모리/WSL 제약을 Spark에 그대로 적용하지 않음.

결정: 본체 1M 검증과 투기적 가속을 분리한다. 동일 프롬프트/정밀도/동시성에서
AR→MTP1→DFlash2를 비교하고 tg/ttft/수용률/메모리/출력 검증을 기록해야 한다.
Huihui 변환 완료 후에는 원본용 draft의 수용률을 다시 측정한다.
