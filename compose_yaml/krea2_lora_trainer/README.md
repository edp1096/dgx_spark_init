# Krea 2 LoRA Trainer

생성 스튜디오가 사용하는 Krea 2 Turbo LoRA 학습 백엔드입니다. Ostris
`ai-toolkit`의 고정 커밋을 CLI 엔진으로 사용하며 upstream UI는 실행하지 않습니다.

```bash
export HF_TOKEN=hf_...
docker volume create media-hf-cache
docker volume create media-krea-user-loras
docker compose build
docker compose up -d
curl http://127.0.0.1:8704/health
```

- API: `http://127.0.0.1:8704`
- 데이터셋과 작업 기록: `./data`
- 등록된 사용자 LoRA: `media-krea-user-loras` Docker volume
- 동시에 하나의 학습만 실행합니다.
- 학습 중단은 ai-toolkit 프로세스 그룹을 종료해 GPU 메모리를 반환합니다.

Spark Media의 `LoRA 제작소`에는 Civitai Krea 2 LoRA 주소 또는 모델 버전 ID를
붙여 넣어 공유 저장소로 가져오는 기능도 있습니다. 설정 화면에서 저장한 Civitai API
키를 재사용하며 Krea 2 기반 safetensors만 허용하고 다운로드 뒤 SHA-256을 검증합니다.
가져온 LoRA는 학습 결과와 동일하게 이미지 생성의 `사용자 LoRA` 목록에 나타납니다.
슬라이더·억제형 LoRA를 위해 적용 강도는 `-2.00..2.00`을 지원합니다.

학습 설정은 Krea 2 Turbo와 공식 training adapter, BF16 LoRA, QFloat8
동결 가중치 조합으로 생성됩니다. 업스트림 갱신은 Dockerfile과 compose.yaml의
`AI_TOOLKIT_COMMIT`을 함께 변경한 후 다시 검증합니다.
