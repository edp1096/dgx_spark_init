# Krea 2 Turbo NVFP4 API

`Comfy-Org/Krea-2`의 Krea 2 Turbo NVFP4 transformer와 Qwen3-VL 4B FP8
텍스트 인코더를 사용하는 OpenAI 호환 Text-to-Image API입니다. ComfyUI는 컨테이너
내부의 `127.0.0.1:8188`에만 바인딩되고 API만 호스트의 `8691`에서 수신합니다.

Krea 2 Turbo 권장 추론 설정인 8 steps, CFG 1, Euler/simple을 사용합니다.
`control_image`를 base64로 전달하면 Depth Anything V2로 깊이 지도를 만들고
`Patil/Krea-2-depth-controlnet`을 적용합니다. 응답의 `control_b64_json`에서 실제
사용된 깊이 지도를 확인할 수 있습니다.

`source_image`를 전달하면 `conradlocke/krea2-identity-edit` v1.2와 전용
`comfyui-krea2edit` 조건화 노드를 사용해 인물·장면의 정체성을 유지하는 지시 편집을
실행합니다. 이 모드는 기본 10 steps이며 `ref_boost=4`, `grounding_px=768`이
권장 시작값입니다. `reference_image`까지 전달하는 2-image 모드에서는
`source_image`가 장면, `reference_image`가 삽입할 인물입니다.

`style`에는 Krea 공식 스타일 LoRA 9종(`darkbrush`, `dotmatrix`, `kidsdrawing`,
`neondrip`, `rainywindow`, `retroanime`, `softwatercolor`, `sunsetblur`,
`vintagetarot`)을 지정할 수 있습니다.
Identity Edit, 스타일 LoRA와 Depth Control은 한 요청에서 함께 사용할 수 있습니다.
예를 들어 `source_image`에는 인물 사진, `control_image`에는 원하는 자세·구도 이미지를
넣으면 얼굴 정체성과 Depth 구도를 동시에 조건으로 사용합니다. 필터 우회 LoRA는
포함하지 않습니다.

추가 요청 필드:

- `source_image`, `reference_image`, `control_image`: PNG/JPEG base64 또는 data URL
- `identity_strength`: Identity Edit LoRA 강도, 기본 `1.0`
- `ref_boost`: 참조 충실도, 기본 `4.0`
- `grounding_px`: Qwen3-VL 참조 해상도, 기본 `768`
- `steps`: 일반 생성 기본 `8`, Identity Edit 기본 `10`
- `style`, `style_strength`: 공식 스타일 LoRA와 강도, 기본 `1.0`
- `control_strength`: Depth Control 강도, 기본 `1.0`
- `vision_images`: Qwen3-VL 의미 기반 참조 이미지 배열, 최대 4장
- `vision_mode`: `descriptor` 또는 `instruct`
- `vision_megapixels`: 의미 기반 참조 해석 해상도, 기본 `1.0`
- `style_reference_images`: Ostris 스타일 이미지 참조, 최대 2장
- `style_reference_strength`: 스타일 이미지 참조 강도, 기본 `1.0`
- `nk2e_image`: NK2E 참조 이미지 1장
- `nk2e_mode`: `edit` 또는 `canny`
- `nk2e_strength`: NK2E LoRA 강도, 기본 `0.7`

스타일 이미지 참조는 공식 ComfyUI 템플릿과 같은 INT8 ConvRot 모델 및
`krea2_style_reference` LoRA를 사용합니다. 현재는 품질 검증 범위를 명확히 하기 위해
Identity, Depth, 일반 스타일 LoRA, Qwen3-VL 의미 참조와 동시에 사용할 수 없습니다.

NK2E v0.3는 짧은 지시의 국소 편집에, 실험적인 Canny v0.1은 참조 이미지의 윤곽과
자세를 반영하는 데 사용합니다. 두 기능 모두 현재 NVFP4 기본 모델에서 실행하며 다른
Krea 모듈과 동시에 적용하지 않습니다. Canny 모드에서는 서버가 OpenCV Canny 맵을
만들고 응답의 `control_b64_json`으로 돌려줍니다. NK2E는 초기 단계의 커뮤니티 LoRA와
커스텀 노드이므로 정식 편집 모델이나 픽셀 단위 인페인팅처럼 취급하지 않습니다.

Identity Edit는 2MP 이하에서 사용해야 하며, Turbo는 큰 삭제 작업보다 인물 재배치,
의상·배경 변경, 스타일 변환에 적합합니다. 실제 사람의 얼굴은 동의받은 용도로만
사용해야 합니다.

```bash
docker volume create media-hf-cache
HF_TOKEN=hf_xxx docker compose build
HF_TOKEN=hf_xxx docker compose up -d
curl http://127.0.0.1:8691/health
```

모델 가중치는 Krea 2 Community License의 적용을 받습니다. 회사와 계열사의 최근
12개월 총매출이 미화 100만 달러 이상이면 상업 이용 전에 별도 Enterprise 라이선스가
필요합니다. 배포 시 해당 라이선스의 콘텐츠 필터링 의무도 확인해야 합니다.
