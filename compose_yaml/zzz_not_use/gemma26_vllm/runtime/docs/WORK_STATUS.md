> 과거 vLLM 실험·운영 이력이다. 아래의 “최신”은 기록 당시 상태이며 현재 운영 설정이 아니다.

# 최신 운영 상태: coolthor FP8 (2026-09-16)

사용자 지시로 비교 시험을 생략하고61번에 FP8 모델을 배치·기동했다.

- 모델: `coolthor/Huihui-gemma-4-26B-A4B-it-abliterated-FP8-Dynamic`, revision `c4bbb7689dc16de4aaa468647625a5a02f98e22f`.
- 공식 Google assistant1, 전체 초안 어휘, MOE_BACKEND=auto(TRITON FP8 선택), 청크4096, FP8 KV32GiB, 문맥1,048,576.
- Compose와 Talk revision9 반영, 메모리 예상82GiB로 조정. 61번 모델 적재·기동·health/models 정상 확인.
- **FP8의1M 품질·속도·툴콜 비교는 수행하지 않았다.** 이전 NVFP4 검증을 FP8 검증으로 간주하지 않는다.
- 전용 GB10/128expert/704 FP8 MoE 튜닝 파일이 없어 기본 커널 설정 사용. 성능 튜닝은 추후.
- 다운로드 기록: `~/.cache/model-download-jobs/gemma26-fp8/`.

아래는 이전 모델 선택과 검증 이력이다.

# 최신 운영 선택: 사카마키 QAT (사용자 재지정)

61번 Compose와 Talk revision8을 사카마키 QAT + 전용 assistant1 / 전체 초안 어휘 / 청크4096 / FP8 KV32GiB / 1M 설정으로 전환했다. 아래 자체 모델 복원 기록은 이전 선택 이력이다.

# 운영 선택 변경 — 자체 Gemma 유지

사용자 지시에 따라 QAT 교체를 취소하고 자체 `edp1096/Huihui-Gemma-4-26B-A4B-it-NVFP4`로 복원 완료.
공식 `google/gemma-4-26B-A4B-it-assistant`, assistant1 + ko64k, 청크1024, FP8 KV32GiB, 1M 설정이다.
이 구성의 기존1M 검색 검증을 재사용했다. 복원 후60번 기동·산술 응답323 확인 통과, 시험 서비스는 중지했다.
61번에 본체3shard·공식 assistant·v6 이미지가 있으며 Compose/Talk 배포와 저장 설정revision7까지 확인했다. 실행 위치는61번 local, 기본 Ornith 유지.

아래 QAT 교체 기록은 과거 실험·적용 이력이며 현재 운영 선택이 아니다.

# 현재 상태 (2026-09-16 20:22 KST)

**Gemma 비교·실측·적용 완료.** [최종 비교 결과](GEMMA-REPLACEMENT-EVALUATION.md)

- 자체 변환본 대신 sakamakismile QAT768 + 전용 assistant1을 Gemma 운영 항목으로 선택. 비QAT768은 한국어 훼손·일본어 반복 붕괴로 제외.
- 최종 QAT 설정: Marlin / 청크4096 / 전체 초안 어휘 / v6 / TP1 / FP8 KV32GiB / 문맥1,048,576. ko64k는 한국어 이득·일본어 감속이 공존하여 선택 옵션으로 보존.
- 1,039,980 입력 + 512 출력 검증: 세 정답과 순서 일치, 총10,978.61초, TTFT10,953.60초, tg20.43tok/s. 일부 한영 글자 혼입·오타가 남아 장문 품질 보증으로 해석하지 않음.
- 60번: 정식 Compose 기동·healthy·이미지 JSON/도구 통합 검사 통과.
- 61번: Ornith 운영 유지. Talk 새 바이너리 배포 및 QAT ID 마이그레이션 완료, 기본 Ornith·사용자 세트 이름/순서/음성 설정 보존 확인. 주변 서비스 health 정상.
- 기존 자체 가중치는 운영 기본 경로에서 제외하고 복구용으로 보존. 공개 저장소 변경·삭제 및 Git 커밋·푸시는 하지 않음.
- 원시 결과: 양쪽 `~/.cache/model-download-jobs/ornith-gemma26/sequential/`. 모든 비교 작업 완료, 진행 중인 1M 요청 없음.

## 이전 작업 기록

# Ornith / Gemma 26B TP1 작업 상태

목표: 원본 BF16 + 공식/기존 NVFP4 + Huihui BF16로 파생 NVFP4를 만들고,
각각 단일 Spark / 1,048,576 문맥으로 Compose와 Talk에 등록·실측한다.
사용자는 검증 시 기존 Qwen TP2 중지를 승인했고, 완료 후 Ornith 35B를 기본으로 유지하도록 지정했다.
HF 업로드는 요청하지 않았다.

- 다운로드 작업: `/home/edp1096/.cache/model-download-jobs/ornith-gemma26/download.py`
- 상태: 같은 폴더의 `*-original.json`, `*-nvfp4.json`, `*-ablit.json`, `gemma-assistant.json`
- 변환 대기 작업: `convert_when_ready.py`; 모델별 세 입력 완료 후 CPU 변환, 메모리 한도 8 GiB.
- 총 가중치 약 287.7 GB. 2026-09-15 21:07 KST 기준 21.5 GB, 약 11 MB/s.
- 사용자 안내 기준 QoS 해제 예상: 2026-09-16 00:35 KST 전후.
- 원본/양자화/ablit의 text_config 일치 확인. Gemma 두 BF16 파일의 텐서 키·shape·dtype도 일치.
- 원본 fused expert와 NVFP4 개별 expert를 연결하는 읽기 전용 view 추가. Gemma `.experts` → `.moe.experts` 명시적 매핑 포함. 28개 변환기 테스트 통과.
- Ornith: 공식 권장 YaRN factor 4. Gemma: proportional RoPE의 factor 4 위치 보간 실험.
- vLLM의 중첩 hf_overrides가 text_config를 dict로 남기는 것을 확인해 사용하지 않음. 가중치를 참조하는 별도 런타임 config view로 교체. 실제 get_config로 두 모델의 256K/512K/1M 로딩 검사 통과.
- vLLM의 Gemma proportional factory/클래스가 factor를 소비하도록 수정. factor 1/2/4의 주파수와 비회전 영역이 Transformers 구현과 일치하는 CPU 검사 통과. 1M 품질/기동은 미검증.
- 이미지: `dgx-vllm-moe-tp1:b40673cd0-v2`. 부모 eugr b40673cd0(2026-09-13), Transformers 5.17.0.
- 기존 이미지의 누락된 export blob 때문에 로컬 rootfs를 재포장한 부모 `dgx-vllm-moe-base:b40673cd0-local` 사용. 추가 외부 이미지 다운로드 없음.
- 60번으로 v2 이미지 전달 완료. launcher와 패치된 RoPE 두 파일의 SHA256 일치 확인. QSFP SSH: `ssh -o HostKeyAlias=192.168.100.60 10.200.0.2`.
- 2026-09-15 22:12: 사용자 요청으로 Qwen TP2 세트 중지. 60번에서 Ornith NVFP4 원본 TP1 기본 문맥(262144) 기동 검증 시작. 새 Talk 바이너리 배포/최종 모델 전환은 아직 하지 않음.
- 새 Compose: Ornith와 Gemma vLLM. 이 런타임은 현재 Gemma vLLM 보관본 내부에 있다.
- Talk 내장 카탈로그와 revision 5 마이그레이션 추가, config/orchestrator 테스트 통과. 기본값 변경은 검증 후 API로 수행 예정.

남은 작업: 다운로드 → 변환 결과 감사 → 원본 NVFP4/파생 native/파생 1M 비교 →
도구·한국어·코드·이미지·장문 검색·메모리 확인 → 필요한 엔진/변환 수정 →
Talk/Compose 최종 적용, 기본 Ornith 활성화, 측정 결과 기록. 1M는 단순 설정 성공만으로 검증 완료 처리하지 않는다.

## 원본 사전 검증 (2026-09-15 22:13 KST 시작)

- Ornith NVFP4 다운로드 완료. 60번 `/home/edp1096/.cache/huggingface/validation/ornith-nvfp4-base` 복사 후 3개 shard의 공식 SHA256 검증 통과.
- 원본/Huihui 및 Gemma 다운로드는 계속 진행.
- 60번 vllm-ornith35: v2 이미지, TP1, 기본 262144, FP8 KV 32GiB, b12x MoE, CUDA graph 활성. 기동 성공/품질/1M 실측 결과는 아직 미확정.
- 검증 전용 override: 작업 캐시 `ornith-base-override.json`, worker `/tmp/ornith-base-override.json`. 원본 가중치/설정은 변경하지 않음.

## 2026-09-16 Huihui Ornith 검증 시작

- 다운로드/변환/60번 전송 완료. 변경 가중치 4644개(4626 NVFP4, 18 FP8), 기존 텐서 80479개 보존. 변환 오차 검사 통과, 최대 상대 L2 약0.09537.
- 60번 후보 전체 파일 SHA256 대조 통과(`/tmp/ornith-candidate-worker-integrity.log`).
- v3 이미지: Qwen3.5 per-head norm 가중치에 맞춰 워밍업 입력 shape 수정. 원본 GPU 구동과 계산/한국어/코드/도구/이미지 검사 통과.
- 양쪽 v3 이미지 빌드 완료, Compose/Talk 내장 자산의 버전도 v3로 변경. 실행 중 Talk 바이너리의 자산 재배포는 최종 모델 적용 때 필요.
- `validate_ornith_candidate.py`가 60번을 Huihui 후보로 전환해 native+64K 입력 검사 → 1M 기동+1,040,000 입력 토큰 검사를 순차 실행한다. 실패하면 이후 단계 중단.
- 상태 `ornith-runtime-validation.json`, 로그 `ornith-runtime-validation.log`; 두 파일 모두 모델 다운로드 작업 폴더에 위치.
- 현재 1M 결과 미확정. Gemma 다운로드와 독립 진행.
