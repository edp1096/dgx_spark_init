# Krea 2 LoRA Trainer

Ostris `ai-toolkit`의 고정 커밋을 DGX Spark용으로 빌드해 공식 웹 UI를 실행합니다.
Spark Media의 `LoRA` 탭에서 현재 접속 중인 호스트의 8675 포트로 바로 이동할 수
있습니다. 기존 Spark Media 전용 학습 API는 호환 확인용 `legacy-api` 프로필로만
남겨 두며 기본 실행에는 포함하지 않습니다.

```bash
export HF_TOKEN=hf_...
docker volume create media-hf-cache
docker volume create media-krea-user-loras
docker compose build
docker compose up -d
curl -I http://127.0.0.1:8675
```

- 공식 UI: `http://127.0.0.1:8675`
- 데이터셋·출력·UI 작업 기록: `./data`
- 등록된 사용자 LoRA: `media-krea-user-loras` Docker volume
- Hugging Face 토큰은 `.env` 또는 공식 UI 설정에서 입력할 수 있습니다.
- LAN 밖으로 공개한다면 `.env`의 `AI_TOOLKIT_AUTH`에 접근 암호를 지정해야 합니다.

이전에 만든 전용 FastAPI 학습기를 확인할 때만 다음처럼 실행합니다. 공식 UI와 전용
API에서 학습을 동시에 시작하지 마세요.

```bash
docker compose --profile legacy-api up -d api
curl http://127.0.0.1:8704/health
```

Spark Media의 `LoRA 제작소`에는 Civitai Krea 2 LoRA 주소 또는 모델 버전 ID를
붙여 넣어 공유 저장소로 가져오는 기능도 있습니다. 설정 화면에서 저장한 Civitai API
키를 재사용하며 Krea 2 기반 safetensors만 허용하고 다운로드 뒤 SHA-256을 검증합니다.
가져온 LoRA는 학습 결과와 동일하게 이미지 생성의 `사용자 LoRA` 목록에 나타납니다.
슬라이더·억제형 LoRA를 위해 적용 강도는 `-2.00..2.00`을 지원합니다.

학습 설정은 Krea 2 Turbo와 공식 training adapter, BF16 LoRA, QFloat8
동결 가중치 조합으로 생성됩니다. 업스트림 갱신은 Dockerfile과 compose.yaml의
`AI_TOOLKIT_COMMIT`을 함께 변경한 후 다시 검증합니다.
