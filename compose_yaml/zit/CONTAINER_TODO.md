# ZIT 컨테이너 작업 지시

## 배경

FaceSwap 방식을 변경했습니다.

### 기존 (실패)
```
SCRFD(TRT) → ArcFace(TRT) → inswapper(TRT) → CodeFormer → paste back
```
- TRT 엔진이 Blackwell(GB10)에서 깨진 출력 생성 (대머리 마네킹)
- inswapper, ArcFace, CodeFormer 전부 제거

### 변경 (ComfyUI Z-Image Headswap 방식)
```
SCRFD(cv2.dnn) → 자동 마스크 생성 → ZIT Inpaint → 얼굴 재생성
```
- inswapper 없이 ZIT Inpaint로 얼굴 영역을 프롬프트 기반 재생성
- SCRFD는 cv2.dnn 백엔드 (DWPose와 동일, aarch64 검증됨)
- 추가 모델: SCRFD ONNX 하나만 (preprocessors/ 폴더에)

---

## 변경된 파일 (호스트에서 수정 완료)

### 1. `app/ui/zit_config.py`
- `FACESWAP_DIR`, `ARCFACE_FILE`, `INSWAPPER_FILE`, `CODEFORMER_*` 삭제
- `SCRFD_FILE`, `SCRFD_URL` 추가 (preprocessors/ 폴더에 저장)

### 2. `app/ui/face_swap.py` — 완전 재작성
- TRTEngine, ONNXEngine, InswapperONNX, CodeFormerRestorer, ArcFaceRecognizer 전부 삭제
- `SCRFDDetector`: cv2.dnn 기반 얼굴 감지 (bbox + 5-point landmarks)
- `create_face_mask()`: 감지된 얼굴에서 soft elliptical 마스크 자동 생성
- `preview_face_detection()`: 감지 결과 + 마스크 오버레이 미리보기
- 싱글톤 `get_detector(model_dir)` — lazy 로딩

### 3. `app/ui/download_models.py`
- `download_faceswap()` → `download_scrfd()`로 교체
- ArcFace, inswapper, CodeFormer 다운로드 제거
- SCRFD는 preprocessors/ 폴더에 다운로드

---

## 컨테이너에서 해야 할 작업

### 작업 1: FaceSwap 탭 UI 수정 (`app/app.py`)

기존 FaceSwap 탭 (target + source + swap):
```python
# 삭제해야 할 것
fs_target = gr.Image(label="Target Image (face to replace)")
fs_source = gr.Image(label="Source Face (reference)")
fs_swap = gr.Button("Swap Face")
```

새로운 FaceSwap 탭 (자동 마스크 + inpaint):
```python
with gr.Tab("FaceSwap", id="faceswap"):
    with gr.Row():
        with gr.Column(scale=1):
            fs_image = gr.Image(label="Input Image", type="numpy")
            fs_detect_btn = gr.Button("Detect Faces", variant="secondary", size="sm")
            fs_preview = gr.Image(label="Detection Preview", interactive=False)
            fs_face_index = gr.Number(value=0, label="Face Index (0=largest, -1=all)", precision=0)
            fs_padding = gr.Slider(1.0, 2.0, value=1.3, step=0.1, label="Mask Padding")
            fs_prompt = gr.Textbox(label="Prompt (describe the new face)", lines=3)
            fs_neg = gr.Textbox(label="Negative Prompt", lines=2)
            # ... 기존 inpaint 파라미터들 (steps, time_shift, control_scale, guidance, etc.)
            fs_generate = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            fs_result = gr.Image(label="Result")
            fs_info = gr.Textbox(label="Info", interactive=False)
```

동작 흐름:
1. 이미지 업로드
2. "Detect Faces" → `preview_face_detection()` 호출 → 감지 결과 미리보기
3. 프롬프트 입력 (예: "beautiful young woman with soft makeup, natural skin")
4. "Generate" → SCRFD 자동 마스크 + ZIT Inpaint

### 작업 2: generators.py의 generate_faceswap 수정

```python
def generate_faceswap(image, face_index, padding, prompt, negative_prompt,
                      resolution, seed, num_steps, guidance_scale,
                      cfg_truncation, control_scale, time_shift,
                      max_sequence_length,
                      progress=gr.Progress(track_tqdm=True)):
    """Auto-mask face + ZIT Inpaint."""
    from face_swap import create_face_mask
    from zit_config import MODEL_DIR

    # 1. SCRFD 자동 마스크 생성
    mask, faces = create_face_mask(image, str(MODEL_DIR),
                                    face_index=int(face_index),
                                    padding=float(padding))
    if mask is None:
        raise gr.Error("No face detected in image.")

    # 2. mask + image를 inpaint 파이프라인으로 전달
    # generate_inpaint와 동일하지만, editor_value 대신 직접 image+mask 전달
    # ... worker에 "inpaint" gen_type으로 제출
```

### 작업 3: worker.py의 _run_faceswap 수정

기존 복잡한 FaceSwap 파이프라인 대신, 단순히:
1. `create_face_mask()` 호출
2. `_run_inpaint()` 호출

```python
def _run_faceswap(kwargs, task_id):
    from face_swap import create_face_mask

    image = PILImage.open(kwargs["image_path"]).convert("RGB")
    image_np = np.array(image)

    mask, faces = create_face_mask(
        image_np, mgr.model_dir,
        face_index=int(kwargs.get("face_index", 0)),
        padding=float(kwargs.get("padding", 1.3)),
    )
    if mask is None:
        raise ValueError("No face detected")

    # Save mask to temp file
    mask_pil = PILImage.fromarray(mask)
    tmp_mask = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=str(OUTPUT_DIR))
    mask_pil.save(tmp_mask.name)
    tmp_mask.close()

    # Delegate to inpaint
    inpaint_kwargs = {
        **kwargs,
        "mask_path": tmp_mask.name,
    }
    return _run_inpaint(inpaint_kwargs, task_id)
```

### 작업 4: pipeline_manager.py의 load_faceswap 수정

```python
def load_faceswap(self):
    """Load SCRFD face detector (cv2.dnn)."""
    if self.faceswap_pipeline is not None:
        return
    from face_swap import get_detector
    get_detector(self.model_dir)
    self.faceswap_pipeline = True  # just a flag, detector is singleton
    logger.info("SCRFD face detector ready")
```

### 작업 5: 기존 .engine 파일 정리

기존 TRT 엔진 캐시가 남아있으면 삭제:
```bash
rm -f ~/.cache/huggingface/hub/zit/faceswap/*.engine
```

faceswap/ 폴더 자체도 불필요 (SCRFD는 preprocessors/에):
```bash
rm -rf ~/.cache/huggingface/hub/zit/faceswap/
```

### 작업 6: SCRFD 다운로드 확인

```bash
cd ~/zit-ui/app/ui && python -c "
from download_models import download_scrfd
download_scrfd()
"
```

### 작업 7: SCRFD cv2.dnn 동작 테스트

```bash
cd ~/zit-ui/app/ui && python -c "
from face_swap import SCRFDDetector
import cv2, numpy as np
# 테스트 이미지로 얼굴 감지
img = cv2.imread('/tmp/test_face.jpg')
if img is not None:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    det = SCRFDDetector('/root/.cache/huggingface/hub/zit/preprocessors/scrfd_10g_bnkps.onnx')
    faces = det.detect(img_rgb)
    print(f'Detected {len(faces)} faces')
    for bbox, lm in faces:
        print(f'  bbox: {bbox}, landmarks: {lm.shape}')
else:
    print('No test image — generate one first via Generate tab, then test')
"
```

---

## 아키텍처 변경 요약

```
Before:
  FaceSwap 탭 → target + source → SCRFD(TRT) → ArcFace(TRT) → inswapper(TRT) → CodeFormer → paste

After:
  FaceSwap 탭 → image + prompt → SCRFD(cv2.dnn) → 자동 마스크 → ZIT Inpaint → 결과
```

- TRT 의존성 제거 (FaceSwap에서)
- ArcFace, inswapper, CodeFormer 모델 불필요
- 추가 모델: SCRFD ONNX (~16MB) 하나만
- ZIT Inpaint 파이프라인 재활용 (이미 동작 중)
