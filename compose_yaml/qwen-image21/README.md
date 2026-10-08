# Qwen Image 2.1 네이티브 이미지 서비스

기본 FLUX 서비스를 대체하는 Qwen Image 2.1 Uncensored NVFP4 API다.
ComfyUI 커밋은 `5c460d8172fe30761ff67c0df3d5643bb74e0d70`으로 고정하며,
원본 W4A8 Qwen3VL 인코더와 BF16 RGBA VAE, 네이티브 INT8 참조 캐시를 사용한다.
가중치의 정확한 리비전과 SHA-256은 `prepare_models.py`에 있다.

```bash
docker compose build
docker compose up -d
curl http://127.0.0.1:8691/health
```

호스트 Hugging Face 캐시를 읽기 전용 `/hf`로 연결한다. 시작 시 세 체크포인트의
전체 SHA-256을 검사한 뒤 ComfyUI를 실행한다. 누락된 가중치는 Talk의 준비/다운로드로
취득하거나 동일 이미지에서 `python3 /opt/qwen-image21/prepare_models.py --download --check`를
오프라인 환경 변수 없이 쓰기 가능한 `/hf` 마운트로 실행한다.

- 생성: `POST /v1/images/generations`, 모델 `qwen-image-2.1-uc-nvfp4`, PNG `b64_json`.
- 참조 편집: `POST /v1/images/edits`, 순서대로 복수 `image` 필드. `operation`은
  `identity_edit` 또는 `reference_generate`다.
- 네이티브 작업: `/v1/loras`의 `operations`. 별도 LoRA/rembg 의존성이 없다.
- 사용 상태: `/v1/runtime/memory`. 실행과 대기를 모두 `busy`로 보고한다.
- `POST /v1/runtime/quiesce`: 작업이 있으면 409, 유휴 상태에서 새 API 작업을 차단한다.
  종료가 실패하면 `/v1/runtime/resume`으로 다시 허용한다.

운영 QAD 512K 세트는 `SPARKTALK_IMAGE_RESIDENCY=dit`와
`SPARKTALK_KEEP_MODELS_LOADED=1`을 사용한다. NVFP4 DiT 약 3.91GiB를 GPU에
고정하고 W4A8 TE·BF16 VAE는 단계가 끝나면 해제한다. 기존 생성·참조·마스크 편집
API를 그대로 사용하며, 요청 취소 시에도 DiT는 유지한다. 기동 예산 7GiB,
작업 전체 예산 14GiB, 추가 작업 여유 하한 6GiB다.

기존 `legacy` 모드는 Comfy 서버를 사용한다. 상주 설정이 없는 Talk 세트는
연속 요청을 재사용하고 2분 유휴 또는 메모리 부족 시 프로세스 전체를 종료한다.
독립 Compose만 실행하면 자동 유휴 종료 타이머는 없으며 `docker compose stop`으로 반환한다.
기동 예산 2.5GiB, 작업 전체 예산 14GiB, 추가 작업 여유 하한 10GiB는 Talk의 입장 검사값이다.
cgroup 112GiB와 swap 상한 112GiB는 선할당이나 모든 입력의 피크 보장이 아니다.

CUDA 없이 API 계약 시험:

```bash
docker run --rm --network none --entrypoint python3 \
  -v "$PWD/test_api.py:/opt/qwen-image21/test_api.py:ro" \
  sparktalk-qwen-image21:nvfp4-int8-v5-dit-resident -B /opt/qwen-image21/test_api.py
```
