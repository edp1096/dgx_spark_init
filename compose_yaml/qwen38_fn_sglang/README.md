# Qwen3.8 Flash-Next · SGLang · 단일 DGX Spark

`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`를 단일 GB10에서
SGLang/B12X로 실행한다. SparkTalk의 TP1 Flash-Next 구성도 같은 소스를 포함한다.
TP2는 기존 Huihui 체크포인트와 기존 이미지를 유지한다.
검증 단계와 실제 API 결과는 [검증 기록](experiments/lil-qad-tp1/VALIDATION.md)을 따른다.
FP8 KV·B12X GDN·체크포인트 PLE 경로에서 **1,048,431토큰 입력과 정답 생성**을
완료했다. 17분 25초가 걸렸고 시험 전체의 최소 호스트 가용 메모리는 14.21GiB였다.
ASR/TTS를 GPU에 올리지 않은 단독 LLM 조건이다. 기본 구성은 64K·동시 2요청을 유지한다.

같은 1M 입력의 [참조 vLLM 시험](experiments/lil-vllm-tp1/README.md)은 20분 29초였다.
짧은 입력·출력 속도는 조건에 따라 다르며, 두 엔진의 로더·출력층·KV 풀 크기까지
같은 것은 아니다. [대응 관계와 측정 조건](experiments/lil-qad-tp1/PARITY.md)을 참고한다.
이전 BF16 KV 시험의 3GiB 중단 결과는 검증 기록에 별도로 보존한다.

## 구성

- 네이티브 컨텍스트 중 65,536 토큰 사용
- 동시 실행 요청 2개
- 내장 NEXTN MTP: 3단계, draft 4개
- 본체 NVFP4 전문가·MXFP8 선형층, MTP/비전 W4A16을 B12X로 실행
- 양자화된 PLE 약 26.8 GiB를 원본 체크포인트에서 조회하며 별도 파일로 재작성하지 않음
- 큰 프리필은 제한된 io_uring 행 버퍼, 그래프 디코딩은 체크포인트 UVA 사용
- 체크포인트 파일 매핑 RSS를 합계 기본 4 GiB 기준으로 정리
- 본체·MTP 모두 FP8 E4M3 KV 풀 131,072 토큰
- B12X GDN 프리필·디코딩·MTP 검증과 SGLang의 수락/프리픽스 상태 관리 연결
- MTP는 체크포인트 인덱스에서 필요한 1개 샤드만 적재
- 적재가 끝난 체크포인트 페이지 캐시는 즉시 반환
- Docker에서도 보이도록 본체·MTP 샤드 진행률과 ETA를 구조화된 줄로 출력
- SM121 QSA는 장문 정확성을 검증한 KDA 커널과 Triton fallback 사용
- 비전 인코더 활성화; 실제 이미지 API 검증. 영상 입력은 이번 검증 범위 밖

소스는 재현 가능하도록 다음 커밋으로 고정한다.

- SGLang PLE backend PR #37068: `0977d22bb005695fef0aee4bc59adfab45b7a496`
- 단일 Spark SM121 recipe: `4f425ca561f767997738e894ee578673e79b01b1`
- QSA FP8 runtime backport PR #36644: `3df8e1e7dbc5807696622afe2929b6c33c185ca3`
- B12X: `9043b448622764a598969518d413b3fd8b3c0c07` · CuTe DSL 4.7.0
- 체크포인트: `7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd`

## 모델 준비

TP1은 아래 revision을 Hugging Face의 일반 Hub 캐시에 준비한다. Compose가
이 snapshot 경로를 직접 사용하므로 `--local-dir`을 붙이지 않는다.

```sh
hf download local-inference-lab/Qwen3.8-Flash-Next-NVFP4 \
  --revision 7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd \
  --cache-dir "$HOME/.cache/huggingface/hub"
```

기존 모델 파일은 삭제하지 않는다. TP2의 모델 준비는 [TP2 문서](docs/tp2.md)를 따른다.

## 빌드

```sh
cd /home/edp1096/workspace/dgx_spark_init/compose_yaml/qwen38_fn_sglang
docker compose build
```

첫 빌드에는 `lmsysorg/sglang:qwen38flashnext` 기반 이미지가 필요하다. 빌드
중에는 PLE 정식 backend와 SM121 QSA 패치가 실제 Python 모듈에 반영됐는지
검증한다. Compose는 `qad-tp1` 빌드 대상을 선택한다. Dockerfile의 기본 대상은
TP2 기반 이미지 호환을 위해 기존 런타임을 유지한다. 새 태그는
`dgx-sglang-qwen38-qad:sm121-v3`이며 기존 TP2 태그를 덮어쓰지 않는다.

## 단독 실행과 확인

```sh
docker compose up -d
docker compose logs -f sglang
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

PLE 파일과 SGLang 컴파일 캐시는 소스 저장소 밖의 다음 디렉터리에 저장된다.

```text
~/.local/share/sparktalk/cache/sglang-flash-next-qad/
```

첫 기동은 체크포인트 적재, 커널 컴파일과 CUDA Graph 캡처 때문에 수 분 걸릴 수 있다.
원본 PLE 파일은 읽기만 하며, 컴파일 캐시는 다음 기동에도 사용한다.
io_uring을 위해 이 컨테이너에는 무제한 memlock과 `seccomp=unconfined`를 설정한다.

## 명시적인 1M 단독 LLM 구성

ASR/TTS를 GPU에서 중지한 뒤 아래 override를 사용한다. 동시 1요청, FP8 KV
1,048,576토큰, Mamba 8슬롯, 프리필 4096, YaRN 4배의 검증 설정이다.
1M 구성은 자동 입력 잘림을 사용하지 않는다.

```sh
docker compose -f compose.yaml -f compose.context1m.yaml up -d
```

이 override는 단독 Compose용이며 Talk의 기본 64K 설정을 자동 변경하지 않는다.
기본 구성으로 돌아갈 때는 `docker compose up -d`를 사용한다.

## 종료

```sh
docker compose down
```

SparkTalk 관리 화면으로 실행할 때는 이 폴더의 Compose 파일을 직접 올리지 않는다.
SparkTalk에 내장된 동일한 실행 구성이 컨테이너를 관리한다.

### 한국어 포함 초안 어휘 64K

기본값은 한국어 포함 64K(`ko64k`)다. `ko64k`는 MTP 초안의 출력 어휘만 줄이며,
완성형 한글을 포함하는 토큰을 보존한다. 가중치를 다시 받거나 패치하지 않는다.
아래 2026-09-06/09 수치는 **기존 Huihui 체크포인트**의 결과이며 새 QAD 모델의 성능 수치가 아니다.
당시 3회 재시험에서는 전체 처리량 약 19.5% 개선이 있었다.
[시험 결과와 안정성 한계](bench/results/2026-09-06-recheck/README.md)를 먼저 확인한다.

2026-09-09의 [16K·32K·64K 비교](bench/results/2026-09-09-ko16k-retry/README.md)에서도
합산 생성 속도는 세 설정 모두 약 30 tok/s였고, 더 작은 어휘의 품질 개선은 확인되지 않았다.
이 결과와 사용자 결정에 따라 `ko64k`를 기준 설정으로 유지한다.

```sh
docker compose build
docker compose up -d
# 기존 전체 어휘로 되돌리기
SPARKTALK_FLASH_NEXT_DRAFT_VOCAB=off docker compose up -d
```

이미지 `dgx-sglang-qwen38-qad:sm121-v3`에는 실행 래퍼와 검증된 어휘 목록이
포함된다. `ko64k`는 토크나이저 SHA256이 다르면 모델 로드 전에 중단한다.
기존 `sm121` 이미지에는 이 옵션이 없으므로 새 이미지를 먼저 준비해야 한다.

SparkTalk에서는 설정 → 시스템 → 서비스 구성의 해당 Flash-Next 서비스에서
`초안 어휘`를 선택하고 저장한 뒤 서비스를 다시 시작한다. 저장 키는
`runtime_options.DRAFT_VOCAB` (`off`/`ko64k`)다. SparkTalk은 내장 Compose로
환경변수를 전달하며 외부 compose_yaml 파일을 읽지 않는다. 실행 호스트에는
위 새 이미지가 필요하다. 이 옵션은 SGLang TP1 전용이며 EXL3에는 적용하지 않는다.

## 두 Spark의 TP2 실험

기존 TP1 설정과 별도로 `compose.tp2.yaml`과 `manage_tp2.py`를 사용한다.
BF16 KV와 256K·512K·1M 시험 절차는 [TP2 문서](docs/tp2.md)를 참고한다.


TP2의 기본 이미지는 `dgx-sglang-qwen38-fn:sm121-b12x-head-v1`이다.
BF16 출력층 일부만 최적화해 tg를 약 3.5~3.8% 개선했으며, pp는 그대로다.
빌드와 1M 검증 결과는 [TP2 문서](docs/tp2.md)를 참고한다.
