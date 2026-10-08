# FLUX.2 Klein vLLM-Omni API

별도 웹 UI 없이 vLLM-Omni의 OpenAI 호환 이미지 생성 API만 실행합니다.
공통 런타임 이미지는 먼저 `../vllm_omni`에서 빌드합니다.

```bash
docker compose -f ../vllm_omni/compose.yaml build
docker volume create media-hf-cache
HF_TOKEN=hf_xxx ./prepare-model.sh
docker compose up -d
curl http://127.0.0.1:8691/health
```

기본 모델은 공식 `black-forest-labs/FLUX.2-klein-4B`의 이미지 생성 구성에
`ponpoke/flux2-klein-4b-uncensored-text-encoder`의 **Safetensors** 텍스트 인코더를
결합한 로컬 조립 모델입니다. `prepare-model.sh`는 저장소의 GGUF 파일을 받지 않고
`flux2-klein-4b-uncensored-text-encoder/` 하위 파일만 선택해서
`media-hf-cache` 볼륨의
`local/flux2-klein-4b-uncensored`에 조립합니다. 공식 모델 캐시는 그대로 유지합니다.
교체 체크포인트의 저장 형식은 FP16이지만 공식 FLUX 본체와 맞도록 로드 시 BF16으로
변환합니다.

API의 기본 모델 식별자는 `flux2-klein-4b-uncensored`입니다. 이전 Media 앱 설정과의
호환성을 위해 `black-forest-labs/FLUX.2-klein-4B`도 보조 별칭으로 허용합니다.

공식 텍스트 인코더로 되돌리려면 다음처럼 모델을 지정합니다.

```bash
MEDIA_IMAGE_MODEL=black-forest-labs/FLUX.2-klein-4B docker compose up -d --force-recreate
```

생성 API는
`POST /v1/images/generations`, 이미지 편집 API는 `POST /v1/images/edits`입니다.
모델은 `media-hf-cache` Docker 볼륨에 저장됩니다.
