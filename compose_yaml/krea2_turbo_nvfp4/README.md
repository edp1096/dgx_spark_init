# Krea 2 Turbo NVFP4 API

`Comfy-Org/Krea-2`의 Krea 2 Turbo NVFP4 transformer와 Qwen3-VL 4B FP8
텍스트 인코더를 사용하는 OpenAI 호환 Text-to-Image API입니다. ComfyUI는 컨테이너
내부의 `127.0.0.1:8188`에만 바인딩되고 API만 호스트의 `8691`에서 수신합니다.
DGX Spark의 CPU와 GPU가 같은 물리 메모리를 공유하므로 ComfyUI는 `--gpu-only`로
실행합니다. 기본 CPU 오프로딩을 사용하면 CUDA 실행본과 CPU 캐시가 같은 통합
메모리에 중복되어, 기본 생성 후 실측 사용량이 약 15 GiB 증가했습니다.

Krea 2 Turbo 권장 추론 설정인 8 steps, CFG 1, Euler/simple을 기본으로 사용합니다.
`sampler_name=er_sde`, `scheduler=simple`을 지정하면 디테일 탐색 프리셋을 사용할 수
있습니다. 허용 조합은 `euler|er_sde`와 `simple`로 제한하며, Detail Enhancer는
ER-SDE/simple을 기본으로 사용합니다.
`control_image`를 base64로 전달하면 Depth Anything V2로 깊이 지도를 만들고
`Patil/Krea-2-depth-controlnet`을 적용합니다. 응답의 `control_b64_json`에서 실제
사용된 깊이 지도를 확인할 수 있습니다.

`source_image`를 전달하면 `conradlocke/krea2-identity-edit` v1.2와 전용
`comfyui-krea2edit` 조건화 노드를 사용해 인물·장면의 정체성을 유지하는 지시 편집을
실행합니다. 이 모드는 기본 10 steps이며 `ref_boost=4`, `grounding_px=768`이
권장 시작값입니다. `reference_image`까지 전달하는 2-image 모드에서는
`source_image`가 장면, `reference_image`가 삽입할 인물입니다.

`reid_image`를 전달하면 [`yijunwang2/krea2-reid`](https://huggingface.co/yijunwang2/krea2-reid)의
Rank 32 ReID LoRA와 Ostris Edit latent 주입을 사용해 서로 독립적으로 생성하는 장면들에
같은 인물 정체성을 조건으로 전달합니다. 공식 체크포인트 전용이며 INT8 ConvRot 본체,
Qwen3-VL BF16 인코더, Qwen VAE, Euler/simple 8 steps 조합을 사용합니다. 사람의 얼굴·머리·
체형에는 효과가 크지만 참조 이미지의 액세서리가 같이 유지될 수 있고, 로봇·갑옷의 정확한 부품
구조나 영상 키프레임 수준의 시간적 연속성을 보장하지 않습니다.

`character_sheet_image`를 전달하면
[`Alissonerdx/CharacterSheet`](https://huggingface.co/Alissonerdx/CharacterSheet)의
`QuadView_krea2_v1` LoRA로 1536×1024 얼굴 확대·정면·측면·후면 시트 후보를 생성합니다.
Krea INT8 ConvRot 본체, 기본 FP8 텍스트 인코더, Ostris Edit latent 주입과 Euler/simple 10 steps를
고정으로 사용합니다. 이 결과는 원본 정체성을 재해석할 수 있으므로 Spark Media는 자동 ReID 앵커로
채택하지 않고 원본과 비교해 승인한 경우에만 Gemma 외형 분석용 보조 참조로 보관합니다.

`style`에는 Krea 공식 스타일 LoRA 9종(`darkbrush`, `dotmatrix`, `kidsdrawing`,
`neondrip`, `rainywindow`, `retroanime`, `softwatercolor`, `sunsetblur`,
`vintagetarot`)을 지정할 수 있습니다.
Identity Edit, 스타일 LoRA와 Depth Control은 한 요청에서 함께 사용할 수 있습니다.
예를 들어 `source_image`에는 인물 사진, `control_image`에는 원하는 자세·구도 이미지를
넣으면 얼굴 정체성과 Depth 구도를 동시에 조건으로 사용합니다. 필터 우회 LoRA는
포함하지 않습니다.

추가 요청 필드:

- `source_image`, `reference_image`, `control_image`, `reid_image`, `character_sheet_image`: PNG/JPEG base64 또는 data URL
- `reference_images`: Identity Edit 보조 참조 배열, 최대 3장. 여러 장은 ComfyUI의 `ImageStitch`로 순서대로 연결해 의상·포즈·소품 참조로 전달
- `identity_strength`: Identity Edit LoRA 강도, 기본 `1.0`
- `ref_boost`: 참조 충실도, 기본 `4.0`
- `source_ref_boost`: 원본(source A) 유지 강도, 기본 `1.0`
- `grounding_px`: Qwen3-VL 참조 해상도, 기본 `768`
- `steps`: 일반 생성 기본 `8`, Identity Edit 기본 `10`
- `sampler_name`, `scheduler`: `euler|er_sde`, `simple`; 기본 `euler/simple`
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

Identity Edit는 공개된 의상·포즈 예제와 같은 노드 v1.2.3을 사용합니다. 요청의
`identity_model`은 `convrot` 또는 `selected`, `identity_encoder`는 `heretic` 또는
`default`를 받습니다. 검증된 시작 조합은 `Krea2_Turbo_convrot_int8mixed`와 Heretic
INT8 ConvRot 텍스트 인코더이며, Moody/Ray 같은 선택 체크포인트 조합도 비교 실험할 수
있습니다. 입력 해상도는 Krea 공식 워크플로우와 같은 8픽셀 배수이며,
참조 이미지는 중간에 손실 압축하지 않은 PNG로 전달하는 것을 권장합니다. 이 모드는
길고 설명적인 프롬프트보다 `she is now wearing the black lace bodysuit.`처럼 짧고
직접적인 변경 지시를 더 안정적으로 따릅니다.

ConvRot 본체는
[`Winnougan/Krea-2-Base-Turbo-NVFP4-FP8-INT8`](https://huggingface.co/Winnougan/Krea-2-Base-Turbo-NVFP4-FP8-INT8)의
`Krea2_Turbo_convrot_int8mixed.safetensors`를 컨테이너 시작 시 자동으로 받습니다.
Heretic 인코더는 Civitai의
[`Qwen3 VL Instruct 4b Heretic 7refusal +convrot`](https://civitai.com/models/2728378?modelVersionId=3099765)이며,
새 설치에서는 Spark Media 설정의 `Krea 모델 준비`에 API 키를 입력하고 버튼을 누르면
체크포인트와 함께 영구 캐시에 내려받습니다. Heretic이 아직 없으면 화면은 기본
Qwen3-VL FP8로 안전하게 전환합니다.

스타일 이미지 참조는 공식 ComfyUI 템플릿과 같은 INT8 ConvRot 모델 및
`krea2_style_reference` LoRA를 사용합니다. 현재는 품질 검증 범위를 명확히 하기 위해
Identity, Depth, 일반 스타일 LoRA, Qwen3-VL 의미 참조와 동시에 사용할 수 없습니다.

NK2E v0.3는 짧은 지시의 국소 편집에, 실험적인 Canny v0.1은 참조 이미지의 윤곽과
자세를 반영하는 데 사용합니다. 두 기능 모두 현재 선택한 공식 생성 모델에서 실행하며 다른
Krea 모듈과 동시에 적용하지 않습니다. Canny 모드에서는 서버가 OpenCV Canny 맵을
만들고 응답의 `control_b64_json`으로 돌려줍니다. NK2E는 초기 단계의 커뮤니티 LoRA와
커스텀 노드이므로 정식 편집 모델이나 픽셀 단위 인페인팅처럼 취급하지 않습니다.

AnyPaint는 `anypaint_image`와 선택적인 `anypaint_mask`를 받아 인페인트·아웃페인트를
처리합니다. SparkTalk처럼 텍스트로 수정 대상을 지정하는 클라이언트를 위해
`POST /v1/masks/segment`도 제공합니다. 입력 이미지와 `prompt`(예: `the red jacket`)를
보내면 Grounding DINO Tiny로 상자를 찾고 SAM 2.1 Small로 경계를 정제한 PNG 마스크를
`mask_b64_json`으로 반환합니다. 두 모델은 요청할 때만 순차 적재되고 Krea 생성 잠금과
공유하므로 대형 생성과 동시에 메모리를 점유하지 않습니다.

`GET /v1/loras`는 `/opt/ComfyUI/models/loras/user`에 실제로 설치된 사용자 LoRA만
열거합니다. 필터 모드로 이동한 `skc3vo.safetensors`는 목록에서 제외됩니다.

## Civitai Krea 2 체크포인트

Spark Media의 `설정 → 연결 → Krea 체크포인트 준비`에서 Civitai API 키를 한 번
입력하면 SSH나 `.env` 수정 없이 다음 제공본을 영구 `media-hf-cache` 볼륨에 받습니다.
키는 `0600` 권한의 비밀 파일로 저장되고 상태 API나 로그로 다시 노출하지 않습니다.
다운로드는 Bearer 인증 헤더, 이어받기, 정확한 파일 크기와 SHA-256 검증을 사용합니다.

- `ray-v1`: Ray Artshoot Krea2 NSFW V1 FP8
- `ray-v2`: Ray Artshoot Krea2 NSFW V2 FP8
- `ray-v3`: Ray Artshoot Krea2 NSFW V3 INT8
- `ray-v4`: Ray Artshoot Krea2 NSFW V4 INT8
- `moody-v7`: Moody Krea 2 Mix V7.0 NVFP4 — 범용·사실적
- `moody-cutie-v4`: Moody Cutie Mix V4.0 NVFP4 — 동양권 SNS 미형
- `moody-amateur-v1`: Moody Amateur Mix V1.0 NVFP4 — 자연스러운 복고 스냅

상태와 준비 API를 직접 사용할 수도 있습니다.

```bash
curl http://127.0.0.1:8691/v1/checkpoints/status
curl -X POST http://127.0.0.1:8691/v1/checkpoints/prepare \
  -H 'Content-Type: application/json' \
  -d '{"civitai_token":"YOUR_KEY","variants":["ray-v1","ray-v2","ray-v3","ray-v4","moody-v7","moody-cutie-v4","moody-amateur-v1"]}'
```

생성 요청에는 `checkpoint=official-int8|official|ray-v1|ray-v2|ray-v2-nvfp4|ray-v3|ray-v4|ray-v4-nvfp4|moody-v7|moody-cutie-v4|moody-amateur-v1`을 지정합니다.
`official-int8`은 얼굴·LoRA·문자 충실도를 우선하는 기본값이고, `official`은 메모리와 첫 적재 속도를 우선하는 NVFP4 고속 선택지입니다.
외부 체크포인트에는 제작자의 조정이 이미 병합돼 있으므로 API는 필터 LoRA 중첩을 허용하지
않으며 `filter_mode=off`를 요구합니다. V1 FP8과 V3 INT8은 제작자 제공본을 그대로
유지합니다. Moody 세 모델은 제작자 권장값인 `Euler Ancestral + Beta`, 8 steps, CFG 1을
`sampling_preset=moody`로 선택할 수 있고, Spark Media에서는 모델 선택 시 자동 적용합니다.

V2와 V4는 준비 화면에서 BF16 원본을 인증 다운로드하고 NVFP4로 자동 변환할 수 있습니다.
변환 전에 ComfyUI 적재 모델을 해제하고, Krea 2의 입력·출력·시간·텍스트 융합 계층은
BF16으로 보존하면서 주 DiT 선형층을 `comfy-kitchen` 네이티브 NVFP4로 양자화합니다.
변환 파일은 safetensors/양자화 메타데이터 검사와 실제 512px 생성을 통과해야 선택 목록에
활성화됩니다. BF16 원본 삭제는 검증 성공 뒤에만 적용되는 선택 사항입니다. 프로필은
[`tritant/ComfyUI_Kitchen_nvfp4_Converter`](https://github.com/tritant/ComfyUI_Kitchen_nvfp4_Converter)
커밋 `2eabdc38abde1337a73f35fa90977322d3305965`를 기준으로 고정했습니다.

```bash
curl -X POST http://127.0.0.1:8691/v1/checkpoints/convert-nvfp4 \
  -H 'Content-Type: application/json' \
  -d '{"civitai_token":"YOUR_KEY","variants":["ray-v2","ray-v4"],"remove_bf16_sources":false}'
```

스타일 참조 모듈은 자체 공식 INT8 체크포인트를 고정 사용하는 별도 워크플로이므로 Ray
체크포인트 선택과 결합하지 않습니다. 스타일 LoRA, Depth, Identity Edit, AnyPaint는 V1~V4
제공본에서 동일 조건 생성 검증을 통과했습니다.

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
