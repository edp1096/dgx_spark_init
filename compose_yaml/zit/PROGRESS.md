# ZIT 구현 진행 결과

## 완료 일시: 2026-03-16

---

## Phase 1: zifk 복사 + Klein 제거 — 완료

### 생성/수정된 파일
- `app/ui/zit_config.py` — 신규 생성. Klein 상수 전부 제거, ControlNet/FaceSwap/전처리기 설정 추가
- `app/ui/pipeline_manager.py` — Klein 제거 (load_klein, cleanup_klein, current_family 전환 로직). FP8 패치 코드 유지
- `app/ui/worker.py` — Klein 핸들러 4개 + _klein_generate 삭제, ZIT 전용
- `app/ui/generators.py` — Klein 생성 함수 4개 + Compare 삭제, generate_zit_t2i + time_shift
- `app/app.py` — Klein/ZIB 라디오, Edit/Compare 탭 삭제, 파라미터 전부 노출
- `app/ui/download_models.py` — Klein 다운로드 제거, ControlNet/전처리기/FaceSwap 추가
- `app/ui/i18n.py` — ZIFK→ZIT, Klein/Edit/Compare 번역 제거, ControlNet/Inpaint/FaceSwap 추가

### 삭제된 파일
- `app/ui/zifk_config.py`
- `tests/test_generate_klein.py`
- `tests/test_generate_klein_base.py`
- `tests/test_generate_zib.py`

---

## Phase 2: VideoX-Fun 모델 클래스 통합 — 완료

### 생성된 파일 (app/ui/videox_models/)
| 파일 | 원본 | 수정 내용 |
|------|------|----------|
| `__init__.py` | 신규 | export 정의 |
| `attention_kernel.py` | VideoX-Fun models/ | 그대로 |
| `attention_utils.py` | VideoX-Fun models/ | local import 수정 |
| `fp8_optimization.py` | VideoX-Fun utils/ | 그대로 |
| `pipeline_z_image_control.py` | VideoX-Fun pipeline/ | `from ..models` → `from .` 수정, AutoencoderKL diffusers import |
| `z_image_transformer2d.py` | VideoX-Fun models/ | multi-GPU/dist 코드 제거, SDPA fallback |
| `z_image_transformer2d_control.py` | VideoX-Fun models/ | multi-GPU 코드 제거 |

### pipeline_manager.py load_zit() 구현
- ZImageControlTransformer2DModel로 통합 (T2I + ControlNet 단일 모델)
- base ZIT 가중치 로딩 → ControlNet adapter 오버레이 (strict=False)
- FP8 q8_kernels 패치 적용
- diffusers AutoencoderKL, transformers AutoModelForCausalLM 사용
- ZImageControlPipeline 빌드

### worker.py _run_zit_t2i() 재작성
- VideoX-Fun ZImageControlPipeline 사용 (control_image=None → 순수 T2I)
- time_shift: scheduler shift 파라미터로 매 생성마다 변경

---

## Phase 3: ControlNet + Inpaint/Outpaint — 완료

### worker.py 핸들러 추가
- `_run_controlnet` — 전처리된 control_image + prompt → ZImageControlPipeline
- `_run_inpaint` — image + mask + prompt → 같은 파이프라인 (mask_image 파라미터)
- `_run_outpaint` — 캔버스 확장 + 마스크 자동 생성 → _run_inpaint 호출
- `_run_faceswap` — placeholder (Phase 5에서 구현)

### generators.py 생성 함수 추가
- `generate_controlnet()` — control_image 전처리 → worker 제출
- `generate_inpaint()` — gr.ImageEditor에서 mask 추출 → worker 제출
- `generate_outpaint()` — 캔버스 확장 + 마스크 → worker 제출
- `preview_preprocessor()` — 메인 프로세스에서 전처리기 실행

---

## Phase 4: 전처리기 — 완료

### 생성된 파일 (app/ui/preprocessors/)
| 파일 | 소스 | 의존성 |
|------|------|--------|
| `__init__.py` | 신규 | preprocess() 통합 인터페이스 |
| `canny.py` | 직접 구현 (~20줄) | cv2만 |
| `gray.py` | 직접 구현 (~5줄) | PIL만 |
| `hed.py` | 직접 구현 (~120줄) | torch + cv2. ControlNetHED_Apache2 + NMS(scribble) |
| `dwpose.py` | 래퍼 | zit_config에서 경로 로드 |
| `depth.py` | 래퍼 | zit_config에서 경로 로드 |
| `dwpose_utils/` | VideoX-Fun comfyui/annotator/dwpose_utils/ 복사 | cv2.dnn (onnxruntime 제거) |
| `zoe/` | VideoX-Fun comfyui/annotator/zoe/ 복사 | torch + timm |

### 핵심 수정
- DWPose: onnxruntime 코드 경로 전부 제거, cv2.dnn.readNetFromONNX 단일 경로
- ZoeDepth: 불필요 파일 정리 (ROS, TF, training, NK variant 등)

---

## Phase 5: FaceSwap TensorRT — 완료

### 생성된 파일
- `app/ui/face_swap.py` — TensorRT 기반 FaceSwap 전체 파이프라인

### 구현 내용
- `TRTEngine` — 범용 ONNX → TRT 변환 + 캐싱 + 추론 래퍼
- `align_face()` — SCRFD 5-point landmark → affine transform (112x112 ArcFace, 128x128 inswapper)
- `paste_back()` — swapped face → 타겟 이미지에 feathered blending
- `FaceSwapPipeline` — detect → align → recognize → swap → paste back 통합

### 주의사항
- SCRFD output parsing은 모델별로 다름 — DGX 테스트 필요
- TRT 엔진 첫 빌드 시 30~60초 소요, .engine 파일로 캐싱

---

## Phase 6: Gradio UI — 완료

### app.py 탭 구성
1. **Generate** — ZIT T2I, 모든 파라미터 메인 노출 (Steps, Time Shift, Guidance, CFG Norm, CFG Trunc, Max Seq, Attention Backend)
2. **ControlNet** — 모드 라디오 (Canny/Pose/Depth/HED/Scribble/Gray) + Preview + 전체 파라미터
3. **Inpaint/Outpaint** — gr.ImageEditor(마스크 그리기) + Outpaint 방향/크기
4. **FaceSwap** — Target + Source + Swap
5. **Settings** — Language, Model Dir, Check Models, LoRA (ZIT만)
6. **History** — zifk와 동일

---

## Phase 7: Builder/Deploy — 완료

### 수정된 파일
- `builder/compose.yml` — flux2 제거, Z-Image만 clone, videox_models/preprocessors 다운로드, timm 추가
- `builder/setup.sh` — 동일 방향, 파일 목록 업데이트
- `builder/update.sh` — zifk→zit 리네임, flux2 검증 제거, 포트 7862
- `compose.yaml` — zit-gradio-spark, 포트 7862
- `app/ui/download_models.py` — ControlNet Union, 전처리기 가중치, FaceSwap 모델 다운로드

---

## DGX 테스트 필요 사항

| # | 항목 | 상태 | 비고 |
|---|------|------|------|
| 1 | `pip install timm` | 미확인 | 순수 Python, 문제없을 것 |
| 2 | FP8 q8_kernels + VideoX-Fun transformer | **미확인** | nn.Linear dtype 체크 기반이라 동작할 것으로 예상 |
| 3 | ZImageControlTransformer2DModel.from_pretrained() | **미확인** | Z-Image-Turbo config.json 호환성 |
| 4 | ControlNet adapter 로딩 (strict=False) | **미확인** | lite 모델이 adapter만 포함하는지 확인 |
| 5 | ZImageControlPipeline T2I (control_image=None) | **미확인** | zero control_context로 동작하는지 |
| 6 | ControlNet Canny 생성 | **미확인** | 전체 파이프라인 E2E |
| 7 | DWPose (cv2.dnn) | **미확인** | ONNX 모델 cv2.dnn 호환성 |
| 8 | ZoeDepth + timm | **미확인** | DPT_BEiT_L_384 백본 로딩 |
| 9 | HED (ControlNetHED.pth) | **미확인** | 가중치 로딩 + 추론 |
| 10 | gr.ImageEditor | **미확인** | NGC Gradio 버전 지원 여부 |
| 11 | ONNX → TRT 변환 (FaceSwap) | **미확인** | SCRFD/ArcFace/inswapper |
| 12 | Inpaint 파이프라인 | **미확인** | mask + image + control_context 조합 |

### 테스트 순서 (권장)
1. `pip install timm` → import 확인
2. `python download_models.py` → 모델 다운로드
3. `python app/app.py --port 7862` → Generate 탭 T2I 테스트
4. ControlNet Canny → Preview + 생성
5. Inpaint → 마스크 + 생성
6. FaceSwap → TRT 빌드 + 얼굴 교체

---

## 파일 구조 최종

```
compose_yaml/zit/
├── PLAN.md                                    # 설계 문서
├── PROGRESS.md                                # 이 문서 (진행 결과)
├── app/
│   ├── app.py                                 # Gradio UI (6탭)
│   └── ui/
│       ├── zit_config.py                      # 설정 (모델 경로, 기본값)
│       ├── pipeline_manager.py                # ZIT + ControlNet 파이프라인 관리
│       ├── worker.py                          # GPU worker (5개 핸들러)
│       ├── generators.py                      # 생성 함수 (T2I/CN/Inpaint/Outpaint/FaceSwap)
│       ├── download_models.py                 # 모델 다운로드
│       ├── face_swap.py                       # TensorRT FaceSwap
│       ├── i18n.py                            # 다국어
│       ├── videox_models/                     # VideoX-Fun 모델 클래스 (7개)
│       │   ├── __init__.py
│       │   ├── attention_kernel.py
│       │   ├── attention_utils.py
│       │   ├── fp8_optimization.py
│       │   ├── pipeline_z_image_control.py
│       │   ├── z_image_transformer2d.py
│       │   └── z_image_transformer2d_control.py
│       └── preprocessors/                     # 전처리기 (6개 모드)
│           ├── __init__.py
│           ├── canny.py
│           ├── depth.py
│           ├── dwpose.py
│           ├── gray.py
│           ├── hed.py
│           ├── dwpose_utils/                  # DWPose ONNX (cv2.dnn)
│           └── zoe/                           # ZoeDepth (torch + timm)
├── builder/
│   ├── compose.yml                            # Docker 빌드 (7단계)
│   ├── setup.sh                               # 독립 설치 스크립트
│   └── update.sh                              # GitHub 핫 업데이트
├── compose.yaml                               # 프로덕션 실행 (포트 7862)
└── tests/
    ├── test_fp8_nan_check.py
    ├── test_fp8_weight_info.py
    └── test_generate_zit.py
```
