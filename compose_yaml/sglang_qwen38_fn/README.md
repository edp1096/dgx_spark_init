# Qwen3.8 Flash-Next · SGLang · 단일 DGX Spark

`dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4`를 단일 GB10에서
SGLang으로 실행한다. SparkTalk의 Flash-Next 세트가 사용하는 정식 런타임이며,
이전 vLLM 구성은 비교와 복구 목적으로 별도 폴더에 남겨 둔다.

## 구성

- 네이티브 컨텍스트 중 65,536 토큰 사용
- 동시 실행 요청 2개
- 내장 NEXTN MTP: 3단계, draft 4개
- ModelOpt NVFP4 MoE는 FlashInfer CUTLASS로 고정
- PLE 47.7 GiB는 NVMe 파일에서 직접 조회
- PLE 매핑 RSS는 기본 8 GiB 한도로 정리
- 적재가 끝난 체크포인트 페이지 캐시는 즉시 반환
- Docker에서도 보이도록 본체·MTP 샤드 진행률과 ETA를 구조화된 줄로 출력
- SM121 QSA는 장문 정확성을 검증한 KDA 커널과 Triton fallback 사용
- 이미지·영상 입력을 보존하기 위해 `--language-only`를 사용하지 않음

소스는 재현 가능하도록 다음 커밋으로 고정한다.

- SGLang PLE backend PR #37068: `0977d22bb005695fef0aee4bc59adfab45b7a496`
- 단일 Spark SM121 recipe: `4f425ca561f767997738e894ee578673e79b01b1`

## 빌드

```sh
cd /home/edp1096/workspace/dgx_spark_init/compose_yaml/sglang_qwen38_fn
docker compose build
```

첫 빌드에는 `lmsysorg/sglang:qwen38flashnext` 기반 이미지가 필요하다. 빌드
중에는 PLE 정식 backend와 SM121 QSA 패치가 실제 Python 모듈에 반영됐는지
검증한다.

## 단독 실행과 확인

```sh
docker compose up -d
docker compose logs -f sglang
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

PLE 파일과 SGLang 컴파일 캐시는 소스 저장소 밖의 다음 디렉터리에 저장된다.

```text
~/.local/share/sparktalk/cache/sglang-flash-next/
```

첫 기동은 체크포인트 적재, PLE 파일 작성, 커널 컴파일과 CUDA Graph 캡처 때문에
수 분 걸릴 수 있다. 이후에도 체크포인트 검증은 수행하지만 컴파일 캐시와 PLE
파일은 유지된다.

## 종료

```sh
docker compose down
```

SparkTalk 관리 화면으로 실행할 때는 이 폴더의 Compose 파일을 직접 올리지 않는다.
SparkTalk에 내장된 동일한 실행 구성이 컨테이너를 관리한다.
