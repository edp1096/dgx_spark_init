# Qwen3.8 Flash-Next · vLLM · 단일 DGX Spark

`dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4`를 단일 GB10에서 vLLM으로
실행하는 재현 가능한 비교·복구용 런타임이다. Qwen3.8 27B 런타임은
`../vllm_qwen38_27b`에 별도로 있으며 이 폴더와 자산을 공유하지 않는다.

현재 SparkTalk의 정식 Flash-Next 세트는 SGLang을 사용한다. 이 구성은 동일한
모델을 vLLM으로 비교하거나 SGLang 경로를 복구할 수 없을 때 사용한다.

## 출처와 고정 버전

GB10 패치의 원본은 다음 저장소의 커밋으로 고정해 포함했다.

- 원본: `https://github.com/blazux/qwen3.8-Flash-DGX`
- 커밋: `209646cd98290035ccbddef29b14c460460a8709`
- 라이선스: Apache-2.0 (`LICENSE`)

Dockerfile의 vLLM 기반 이미지도 digest로 고정되어 있다. 원본 저장소의
`Dockerfile`, `src`, `scripts`, `tools`, 상세 문서와 라이선스를 함께 보존했으므로
이 폴더만으로 동일한 패치 이미지를 다시 빌드할 수 있다.

## 포함된 GB10 보정

- 47.7 GiB PLE 테이블을 NVMe에서 `mmap`으로 조회
- GB10 FLA 공유메모리 경계와 워프 문제 보정
- Mamba 상태 복사 경합 방지와 범위 검사
- prefix cache의 Mamba 블록 크기 오류 수정
- 결정론적인 QSA exact top-k
- NVFP4 전문가와 FP8 측면 레이어를 조합하는 선택적 Hybrid 모드
- 선택적 FP8 KV cache 지원

세부 원리와 실측은 `docs/HOW-IT-WORKS.md`에 보존했다.

## 이미지 빌드

```sh
cd /home/edp1096/workspace/dgx_spark_init/compose_yaml/vllm_qwen38fn
docker compose build
```

생성 이미지와 기본 컨테이너 이름은 다음과 같다.

```text
image:     dgx-vllm-qwen38fn:sm121
container: vllm-qwen38fn
```

## 모델 다운로드

모델이 로컬 Hugging Face 캐시에 없다면 먼저 실행한다. 다운로드는 중단 후 다시
실행해도 이어받는다.

```sh
HF_TOKEN=... scripts/download-weights.sh
```

기본 모델은 dealignai Abliterated NVFP4다. 다른 체크포인트를 사용하려면
`MODEL`과 Compose의 `VLLM_QWEN38_FLASH_MODEL_PATH`를 함께 지정한다.

## 실행과 검증

```sh
docker compose up -d
docker compose logs -f vllm
scripts/smoke-test.sh
```

기본 API는 호스트의 `127.0.0.1:8000`에만 바인딩된다. 외부 접속이 필요할 때만
다음처럼 명시적으로 변경한다.

```sh
VLLM_BIND_ADDRESS=0.0.0.0 docker compose up -d
```

기본 운영값은 SparkTalk의 비교 조건과 맞춘 64K 문맥, 동시 요청 2개, 7 GiB
KV cache, MTP 2토큰이다. 자동 메모리 비율만 사용하면 필요 이상으로 KV가
예약되어 FLUX·ASR·TTS와 공존할 여유가 줄어드므로 7 GiB 고정을 유지한다.

## Hybrid 준비

Hybrid는 원본 체크포인트를 변경하지 않고 형제 스냅샷을 만든다. 약 13 GB의
추가 디스크가 필요하다.

```sh
scripts/prepare-hybrid.sh
```

생성된 `-fp8hybrid` 스냅샷 경로를 확인한 뒤 실행한다.

```sh
VLLM_FP8_HYBRID=1 \
VLLM_QWEN38_FLASH_MODEL_PATH=/hf/hub/models--dealignai--Qwen3.8-Flash-Next-ABLITERATED-NVFP4/snapshots/<revision>-fp8hybrid \
docker compose up -d
```

## 독립 실행 도구

`scripts/serve.sh`는 Compose와 별개로 컨텍스트·동시성·YaRN·KV 정밀도를 바꿔
실험할 때 사용한다. 일반적인 재기동과 운영은 Compose를 사용한다.

## 종료

```sh
docker compose down
```
