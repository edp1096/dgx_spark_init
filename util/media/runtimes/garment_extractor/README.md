# Garment Extractor

Spark Media의 `의상 추출` 후처리용 로컬 API입니다. FASHN Human Parser를 이용해
선택한 의상 종류를 분할하고 다음 결과를 반환합니다.

- `cutout_b64`: 선택 영역 밖의 알파와 RGB를 모두 제거한 투명 PNG
- `mask_b64`: 선택 영역의 흑백 마스크
- `reference_b64`: 생성 모델 참조용 1024×1024 중립 배경 이미지

`reference_b64`는 원본 착용자의 얼굴·머리·피부와 배경이 이미지 생성 모델의
컨디셔닝에 섞이지 않도록 선택한 의상 픽셀만 회색 캔버스에 배치합니다. 투명 PNG를
참조 이미지로 바로 사용하면 일부 로더가 알파를 무시할 수 있으므로, H3 같은 생성
모델에는 `reference_b64`를 사용합니다.

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

사람의 팔·머리카락 등으로 가려진 의상 영역은 의미 분할만으로 복원하지 않습니다.
가림이 큰 사진은 별도 의상 사진을 사용하거나 생성형 복원 단계를 선택적으로 거치는
편이 정확합니다.

- 모델: <https://huggingface.co/fashn-ai/fashn-human-parser>
- 라이선스: NVIDIA SegFormer Source Code License
