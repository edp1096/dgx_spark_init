# Garment Extractor

Spark Media의 `의상 추출` 후처리용 로컬 API입니다. FASHN Human Parser를 이용해
선택한 의상 종류를 분할하고 투명 PNG와 흑백 마스크를 반환합니다.

```bash
docker compose build
docker compose up -d
curl http://127.0.0.1:8705/health
```

모델은 첫 요청 때 `media-hf-cache`에 내려받으며 이후에는 오프라인으로 재사용합니다.
모델 revision은 `1f80c34dbab321c5730dda5c3fea279fd3e97498`로 고정했습니다.

여러 입력 이미지를 보내면 옷이 크게 보이고, 화면 밖으로 잘리지 않았으며, 선명한
이미지를 자동으로 선택합니다. 서로 다른 자세의 픽셀을 억지로 합성하지 않으므로
존재하지 않는 무늬를 만들지 않습니다.

- 모델: <https://huggingface.co/fashn-ai/fashn-human-parser>
- 라이선스: NVIDIA SegFormer Source Code License
