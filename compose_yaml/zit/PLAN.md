# ZIT — Z-Image-Turbo 단일 모델 이미지 생성/편집 UI

## 개요

ZIFK(Z-Image + FLUX.2 Klein 통합 UI)에서 **FLUX/Klein을 완전 제거**하고,
Z-Image-Turbo(ZIT) 단일 패밀리 + ControlNet + FaceSwap(TensorRT)으로
이미지 생성/편집을 통합하는 새 프로젝트.

### FLUX/Klein 제거 이유
- **메모리**: ZIT + Klein 이중 로딩/전환 제거
- **모델 관리**: 단일 패밀리로 단순화
- **검열**: FLUX 내장 콘텐츠 필터 제거

### 핵심 요구사항
- **포즈 유지 + 얼굴 identity 보존** 편집이 가능해야 함
- 모든 파라미터 UI에 노출 (숨기지 않음)
- zifk 폴더는 건드리지 않음 (복사 후 수정)
- 모델 디렉토리 zifk와 완전 독립

---

## 환경

- **Hardware**: DGX Spark (aarch64, Blackwell GPU, 128GB unified memory)
- **Base Image**: NGC `nvcr.io/nvidia/pytorch:26.02-py3`
- **컨테이너 확인 완료 패키지**:
  - torch, transformers, gradio, opencv-python: 있음
  - tensorrt 10.15.1: 있음
  - timm: **없음** (pip install 필요)
  - onnxruntime: **없음** (aarch64 공식 빌드 미제공, TensorRT로 대체)
  - insightface: **없음** (TensorRT 직접 구현으로 대체)

---

## 모델 디렉토리

```
~/.cache/huggingface/hub/zit/              # zifk와 완전 독립
├── Z-Image-Turbo/                         # 베이스 모델 (ZIT만, ZIB 없음)
├── controlnet/
│   └── Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps.safetensors  (~1.9GB)
├── preprocessors/
│   ├── yolox_l.onnx                       # DWPose 인체 감지 (~200MB)
│   ├── dw-ll_ucoco_384.onnx               # DWPose 관절 추정 (~90MB)
│   ├── ZoeD_M12_N.pt                      # ZoeDepth (~350MB)
│   └── ControlNetHED.pth                  # HED 소프트 엣지 (~29MB)
├── faceswap/
│   ├── scrfd_10g_bnkps.onnx               # 얼굴 감지 (SCRFD)
│   ├── w600k_r50.onnx                     # 얼굴 인식 (ArcFace)
│   ├── inswapper_128.onnx                 # 얼굴 교체
│   └── *.engine                           # TRT 변환 엔진 (빌드 시 생성)
└── loras/
```

### 모델 다운로드 URL

| 모델 | URL |
|------|-----|
| Z-Image-Turbo | `Tongyi-MAI/Z-Image-Turbo` (HuggingFace) |
| ControlNet Union lite 2602 | `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1` → `*-lite-2602-8steps.safetensors` |
| DWPose det | `https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx` |
| DWPose pose | `https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx` |
| ZoeDepth | `https://huggingface.co/lllyasviel/Annotators/resolve/main/ZoeD_M12_N.pt` |
| HED | `https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth` |
| inswapper | 별도 확인 필요 (InsightFace 배포) |

---

## 폴더 구조

```
compose_yaml/zit/
├── PLAN.md                                # 이 문서
├── app/
│   ├── app.py                             # Gradio UI
│   └── ui/
│       ├── generators.py                  # 생성 함수 (T2I/ControlNet/Inpaint/FaceSwap)
│       ├── worker.py                      # GPU worker 프로세스
│       ├── pipeline_manager.py            # ZIT + ControlNet + FaceSwap 로딩
│       ├── face_swap.py                   # TensorRT 기반 FaceSwap (SCRFD+ArcFace+inswapper)
│       ├── i18n.py                        # 다국어
│       ├── zit_config.py                  # 모델 경로, 해상도, 기본값
│       └── preprocessors/
│           ├── __init__.py                # preprocess("canny", img) 통합 인터페이스
│           ├── canny.py                   # cv2.Canny (~20줄, 모델 없음)
│           ├── depth.py                   # ZoeDepth (VideoX-Fun에서 복사)
│           ├── dwpose.py                  # DWPose (VideoX-Fun dwpose_utils/ 복사, cv2.dnn 사용)
│           ├── hed.py                     # HED (~80줄 직접 구현, ControlNetHED.pth 사용)
│           └── gray.py                    # PIL grayscale (~5줄, 모델 없음)
├── builder/
│   ├── compose.yml                        # 빌드용
│   ├── setup.sh                           # 의존성 설치
│   └── update.sh
├── compose.yaml                           # 실행용
└── tests/
```

---

## 코드 베이스

### 복사 원본: compose_yaml/zifk/
zifk의 worker/process 아키텍처를 유지하되, FLUX/Klein 코드를 전부 제거하고 교체.

### 삭제 대상 (zifk → zit 복사 후)
| 파일 | 삭제 내용 |
|------|----------|
| zit_config.py (← zifk_config.py) | `KLEIN_*`, `LORAS_KLEIN_DIR`, `KLEIN_AE_*` 상수 전부 |
| pipeline_manager.py | `load_klein()`, `cleanup_klein()`, `current_family` 전환 로직, `flux2` import |
| worker.py | `_klein_generate()`, `_run_klein_*` 핸들러 4개, `flux2` import |
| generators.py | `generate_klein_*` 4개 함수, Compare의 Klein 부분 |
| app.py | Generate의 Klein/Klein Base 라디오, Edit 탭 전체 (→ ControlNet/Inpaint 으로 교체) |
| download_models.py | Klein 모델 다운로드 |

### 추가 대상
| 파일 | 내용 |
|------|------|
| zit_config.py | ControlNet, 전처리기, FaceSwap 경로 추가 |
| pipeline_manager.py | `load_controlnet()`, `load_faceswap()`, `get_preprocessor()` 추가 |
| worker.py | `_run_controlnet`, `_run_inpaint`, `_run_outpaint`, `_run_faceswap` 핸들러 |
| generators.py | ControlNet/Inpaint/Outpaint/FaceSwap 생성 함수 |
| app.py | ControlNet/Inpaint/FaceSwap 탭 추가 |
| face_swap.py | TensorRT 기반 SCRFD+ArcFace+inswapper (신규) |
| preprocessors/ | Canny/DWPose/ZoeDepth/HED/Gray (신규) |

---

## 핵심 설계 결정

### 1. VideoX-Fun 모델 클래스로 통일
Z-Image repo의 모델 클래스 대신 **VideoX-Fun의 모델 클래스**를 사용.
- 이유: ControlNet(`ZImageControlTransformer2DModel`)이 VideoX-Fun의 `ZImageTransformer2DModel`을 상속
- Z-Image repo 코드와 VideoX-Fun 코드의 transformer 구조가 다를 수 있어 호환성 문제 회피
- FP8 q8_kernels 패치는 VideoX-Fun transformer에 맞게 재적용 필요

### 소스 위치
- VideoX-Fun: `/mnt/d/dev/pcbangstudio/workspace/VideoX-Fun/`
  - 파이프라인: `videox_fun/pipeline/pipeline_z_image_control.py`
  - ControlNet 모델: `videox_fun/models/z_image_transformer2d_control.py`
  - 베이스 모델: `videox_fun/models/z_image_transformer2d.py`
  - 전처리기: `comfyui/annotator/` (dwpose_utils/, zoe/)
- Z-Image: `/mnt/d/dev/pcbangstudio/workspace/Z-Image/`
  - 기존 참조용 (직접 사용하지 않음)
- ZIFK (복사 원본): `/mnt/d/dev/pcbangstudio/workspace/dgx_spark_init/compose_yaml/zifk/`

### 2. ControlNet Union — 단일 모델, 다중 모드
ControlNet Union 하나로 Pose/Canny/HED/Depth/Scribble/Gray/Inpaint 전부 처리.
- 모델이 모드를 구분하지 않음 → 전처리된 control image를 그대로 받음
- 전처리는 외부에서 수행 (preprocessors/)

### 3. Inpaint/Outpaint = 같은 파이프라인
별도 파이프라인 아님. 같은 `ZImageControlPipeline`에 `mask_image` 파라미터 추가.
- `control_in_dim=33`: `[control_latents(3) + mask(3) + inpaint_latents(27)]` concat
- Outpaint = 캔버스 확장 + 빈 영역 마스크 → Inpaint 호출

### 4. FaceSwap — TensorRT 직접 구현
onnxruntime이 aarch64에서 공식 미지원 (GitHub issue #26351).
NGC 컨테이너에 TensorRT 10.15.1이 있으므로:
- SCRFD (얼굴 감지) ONNX → TRT 엔진
- ArcFace (얼굴 인식) ONNX → TRT 엔진
- inswapper (얼굴 교체) ONNX → TRT 엔진
- insightface 패키지 사용하지 않음, 직접 구현

### 5. 메모리 전략
ZIT + ControlNet **동시 상주** (128GB에서 충분).
zifk처럼 family 전환/cleanup 불필요.
- FaceSwap: 온디맨드 로딩
- 전처리기: lazy 로딩 (첫 사용 시)

### 6. 전처리기 — 선별적 복사
AIO Preprocessor 전체가 아닌 필요한 것만:
| 전처리기 | 소스 | 의존성 |
|----------|------|--------|
| Canny | cv2.Canny 직접 | cv2만 |
| DWPose | VideoX-Fun `comfyui/annotator/dwpose_utils/` | cv2.dnn (onnxruntime 불필요!) |
| ZoeDepth | VideoX-Fun `comfyui/annotator/zoe/` | torch + timm |
| HED | 직접 구현 (~80줄, ControlNetHED_Apache2) | torch + cv2 |
| Gray | PIL .convert('L') | 없음 |
| MLSD | 미포함 (나중에 필요하면 추가) | - |

---

## UI 탭 구성

### Tab 1: Generate
모든 파라미터 메인 영역에 노출 (Advanced 아코디언 사용 안 함):
- Prompt, Negative Prompt
- Resolution (드롭다운 + 커스텀)
- Seed, Num Images
- Steps (1~100, 기본 8)
- **Time Shift** (1.0~12.0, 기본 3.0) — 디노이징 스텝 집중 분배 조절
- Guidance Scale (0.0~10.0, 기본 0.5)
- CFG Normalization (체크박스)
- CFG Truncation (0.0~1.0, 기본 0.9)
- Max Sequence Length (64~1024, 기본 512)
- Attention Backend (드롭다운)

#### Time Shift 설명
FlowMatchEulerDiscreteScheduler의 `shift` 파라미터.
`sigma' = shift × sigma / (1 + (shift - 1) × sigma)`
- 1.0: 균등 분배 (디테일 중시)
- 3.0: ZIT 8스텝 최적 (기본값)
- 6~12: 스텝 수 늘릴 때 (구도에 집중)
모델 리로딩 없이 매 생성마다 변경 가능.

### Tab 2: ControlNet
- Control Mode 라디오: Canny / Pose / Depth / HED / Scribble / Gray
- Input Image 업로드
- Control Preview + Preview 버튼 (전처리 결과 미리보기)
- Prompt, Negative Prompt
- Resolution + Match Image Size 버튼
- Seed
- Steps, Time Shift, Control Scale (0.0~1.0), Guidance Scale, CFG Truncation, Max Seq Length

### Tab 3: Inpaint/Outpaint
- Mode 라디오: Inpaint / Outpaint
- [Inpaint] gr.ImageEditor (브러시로 마스크 그리기, 흰색=재생성)
- [Outpaint] Image + Direction(Left/Right/Up/Down/All) + Expand Size(64~512)
- Prompt, Negative Prompt
- Seed, Steps, Time Shift, Control Scale, Guidance Scale

### Tab 4: FaceSwap
- Target Image (얼굴을 교체할 이미지)
- Source Face (참조 얼굴 이미지)
- Swap Face 버튼
- Result 이미지

### Tab 5: History
- zifk 히스토리와 동일 구조

### Tab 6: Settings
- Language 선택
- Model Directory
- Check Models
- LoRA 다운로드 (ZIT용만)
- 설치된 LoRA 목록

---

## 아키텍처

```
Main (Gradio) process              Worker process (GPU)
─────────────────────              ─────────────────────
app.py                             worker.py
  ├── generators.py                  ├── pipeline_manager.py
  │   submit_and_wait()  ──queue──►  │   ├── load_zit()        # 상주
  │                      ◄─queue──   │   ├── load_controlnet()  # 상주
  │                                  │   ├── load_faceswap()    # 온디맨드
  └── preprocessors/                 │   └── get_preprocessor() # lazy
      (전처리 미리보기는 메인에서)      └── 핸들러
                                         zit_t2i / controlnet /
                                         inpaint / outpaint / faceswap
```

### Worker 핸들러
```python
HANDLERS = {
    "zit_t2i":      _run_zit_t2i,       # T2I 기본 생성
    "controlnet":   _run_controlnet,     # Pose/Canny/HED/Depth/Scribble/Gray
    "inpaint":      _run_inpaint,        # 마스크 기반 부분 편집
    "outpaint":     _run_outpaint,       # 캔버스 확장 (→ inpaint 호출)
    "faceswap":     _run_faceswap,       # TensorRT FaceSwap
}
```

### Pipeline Manager
```python
class PipelineManager:
    # ZIT (항상 상주)
    zit_components = None       # transformer, vae, text_encoder, tokenizer, scheduler

    # ControlNet (항상 상주)
    controlnet = None

    # FaceSwap (온디맨드)
    face_detector = None        # SCRFD TRT
    face_recognizer = None      # ArcFace TRT
    face_swapper = None         # inswapper TRT

    # 전처리기 (lazy)
    _preprocessors = {}

    def load_zit(self):         # VideoX-Fun 모델 클래스 + FP8 q8_kernels 패치
    def load_controlnet(self):  # ControlNet Union lite 로딩
    def load_faceswap(self):    # ONNX → TRT 엔진 로드
    def get_preprocessor(self, mode):  # lazy 로딩
```

---

## 의존성

### pip install 필요 (setup.sh)
```bash
pip install timm       # ZoeDepth 백본 (순수 Python, aarch64 문제없음)
pip install einops     # 텐서 변환 (순수 Python)
```

### NGC 컨테이너에 이미 있음
- torch, torchvision, transformers, accelerate
- gradio, opencv-python, numpy, PIL
- tensorrt 10.15.1
- safetensors

### 설치 불필요 (대체됨)
- ~~onnxruntime-gpu~~ → TensorRT로 대체
- ~~insightface~~ → TensorRT 직접 구현으로 대체
- ~~controlnet_aux~~ → 선별 복사 + 직접 구현

---

## 구현 순서

| 순서 | 작업 | 비고 |
|------|------|------|
| 1 | zifk 복사 → Klein 코드 전부 제거, ZIB 제거 | 기반 정리 |
| 2 | zit_config.py + 모델 경로 재구성 | 독립 디렉토리 |
| 3 | VideoX-Fun 모델 클래스로 전환 (ZImageTransformer2DModel) | 핵심 전환 |
| 4 | Generate 탭: time_shift 등 파라미터 전부 노출 | UI |
| 5 | ControlNet 파이프라인 통합 (VideoX-Fun pipeline 분리) | 핵심 |
| 6 | 전처리기 구현 (Canny → DWPose → ZoeDepth → HED → Gray) | 선별 복사 |
| 7 | ControlNet 탭 UI + Preview | UI |
| 8 | Inpainting (gr.ImageEditor + ControlNet Inpaint) | 핵심 |
| 9 | Outpainting (캔버스 확장 + Inpaint) | 8 확장 |
| 10 | FaceSwap TRT 구현 (SCRFD+ArcFace+inswapper) | TensorRT |
| 11 | FaceSwap 탭 UI | UI |
| 12 | download_models.py | 유틸 |
| 13 | builder/setup.sh + compose.yaml | 배포 |
| 14 | 테스트 | 전체 검증 |

---

## 확인 필요 사항 (DGX 테스트)

| 항목 | 상태 | 내용 |
|------|------|------|
| FP8 q8_kernels + VideoX-Fun transformer | **미확인** | 기존 패치가 VideoX-Fun 모델 클래스에 적용 가능한지 |
| ControlNet Union lite 품질 | **미확인** | lite(1.9GB)가 실사용에 충분한지 |
| ONNX → TRT 변환 (inswapper) | **미확인** | TRT 빌드 성공 여부 |
| timm 설치 후 ZoeDepth 동작 | **미확인** | aarch64에서 정상 동작 확인 |
| gr.ImageEditor | **미확인** | NGC Gradio 버전에서 지원 여부 |

---

## 참고 자료

### ControlNet Union 모델
- HuggingFace: `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`
- 지원 모드 (2602): Canny, Depth, Pose, MLSD, HED, Scribble, Gray
- Inpainting: 같은 파이프라인에 mask_image 파라미터
- control_context_scale: 0.65~1.0 (높을수록 강한 제어)

### HED 모델 (직접 구현)
- 가중치: `ControlNetHED.pth` (29.4MB, Apache 2.0)
- VGG 스타일 5블록 conv net, ~80줄
- Scribble도 HED + NMS로 생성

### onnxruntime aarch64 이슈
- GitHub: `microsoft/onnxruntime#26351`
- 공식 aarch64 GPU wheel 미제공
- 해결: TensorRT 10.15.1로 대체 (NGC에 이미 설치됨)

### VideoX-Fun
- 로컬: `/mnt/d/dev/pcbangstudio/workspace/VideoX-Fun/`
- ControlNet 파이프라인: standalone 분리 가능 (프레임워크 강결합 아님)
- 전처리기: `comfyui/annotator/` (DWPose=cv2.dnn, ZoeDepth=torch+timm)
