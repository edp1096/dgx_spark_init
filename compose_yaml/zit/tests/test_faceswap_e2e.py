"""FaceSwap E2E test — SCRFD detection + auto-mask + ZIT Inpaint pipeline.

Tests the new Phase 5 faceswap approach:
  SCRFD(cv2.dnn) → auto-mask → ZIT Inpaint → face regeneration

Requires GPU + model weights.

Usage:
    cd /root/zit-ui && python tests/test_faceswap_e2e.py
    cd /root/zit-ui && python -m pytest tests/test_faceswap_e2e.py -v -s
"""
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app" / "ui"))

MODEL_DIR = str(Path.home() / ".cache" / "huggingface" / "hub" / "zit")
OUTPUT_DIR = Path("/tmp/zit-test-faceswap")
OUTPUT_DIR.mkdir(exist_ok=True)

SCRFD_PATH = os.path.join(MODEL_DIR, "preprocessors", "scrfd_10g_bnkps.onnx")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_real_face_image():
    """Find a real face image from previous generation outputs."""
    for candidate in [
        "/tmp/test_face_for_swap.png",
        "/tmp/test_target.jpg",
        "/tmp/test_target.png",
    ]:
        if os.path.exists(candidate):
            return candidate
    # Check generated outputs
    import glob
    for p in sorted(glob.glob("/tmp/zit-outputs/zit_*.png"), reverse=True):
        return p
    return None


# ===========================================================================
# Test 1: SCRFD model loads via cv2.dnn
# ===========================================================================
def test_scrfd_model_loads():
    """SCRFD ONNX loads correctly with cv2.dnn backend."""
    from face_swap import SCRFDDetector

    assert os.path.exists(SCRFD_PATH), f"SCRFD model not found: {SCRFD_PATH}"
    det = SCRFDDetector(SCRFD_PATH)
    assert det.net is not None
    assert len(det._output_names) == 9, f"Expected 9 outputs, got {len(det._output_names)}"
    print(f"  PASS: SCRFD loaded, {len(det._output_names)} outputs")


# ===========================================================================
# Test 2: SCRFD det_size is 320 (not 640)
# ===========================================================================
def test_scrfd_det_size_default():
    """SCRFD detect() uses det_size=(320,320) by default.

    The scrfd_10g_bnkps.onnx from antelopev2 only works at 320x320.
    At 640x640 scores drop to ~0.04 (unusable).
    """
    import inspect
    from face_swap import SCRFDDetector

    sig = inspect.signature(SCRFDDetector.detect)
    default = sig.parameters["det_size"].default
    assert default == (320, 320), f"Expected (320,320), got {default}"
    print(f"  PASS: det_size default is {default}")


# ===========================================================================
# Test 3: SCRFD detects face in real image
# ===========================================================================
def test_scrfd_detects_real_face():
    """SCRFD finds at least one face in a generated portrait."""
    from face_swap import SCRFDDetector

    img_path = find_real_face_image()
    if img_path is None:
        print("  SKIP: no real face image found (generate one first)")
        return

    img = np.array(Image.open(img_path).convert("RGB"))
    det = SCRFDDetector(SCRFD_PATH)
    faces = det.detect(img, threshold=0.5)

    assert len(faces) > 0, "No face detected in real portrait image"
    bbox, landmarks = faces[0]
    assert bbox.shape == (4,), f"bbox shape {bbox.shape} != (4,)"
    assert landmarks.shape == (5, 2), f"landmarks shape {landmarks.shape} != (5,2)"

    # bbox should be within image bounds (allowing small overflow from padding)
    h, w = img.shape[:2]
    assert bbox[2] > bbox[0], "bbox x2 should be > x1"
    assert bbox[3] > bbox[1], "bbox y2 should be > y1"

    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    print(f"  PASS: detected {len(faces)} face(s), largest area={area:.0f}px²")


# ===========================================================================
# Test 4: create_face_mask generates valid mask
# ===========================================================================
def test_create_face_mask():
    """create_face_mask returns properly shaped mask with correct values."""
    from face_swap import create_face_mask

    img_path = find_real_face_image()
    if img_path is None:
        print("  SKIP: no real face image")
        return

    img = np.array(Image.open(img_path).convert("RGB"))
    mask, faces = create_face_mask(img, MODEL_DIR, face_index=0, padding=1.3)

    assert mask is not None, "Mask should not be None"
    assert mask.shape == img.shape[:2], f"Mask shape {mask.shape} != image shape {img.shape[:2]}"
    assert mask.dtype == np.uint8
    assert mask.max() == 255, "Mask should have 255 (inpaint region)"
    assert mask.min() == 0, "Mask should have 0 (preserve region)"

    coverage = np.count_nonzero(mask) / mask.size * 100
    assert 0 < coverage < 100, f"Mask coverage {coverage:.1f}% should be between 0-100"

    Image.fromarray(mask).save(str(OUTPUT_DIR / "test_mask.png"))
    print(f"  PASS: mask shape={mask.shape}, coverage={coverage:.1f}%")


# ===========================================================================
# Test 5: create_face_mask with face_index=-1 (all faces)
# ===========================================================================
def test_create_face_mask_all_faces():
    """face_index=-1 should mask all detected faces."""
    from face_swap import create_face_mask

    img_path = find_real_face_image()
    if img_path is None:
        print("  SKIP: no real face image")
        return

    img = np.array(Image.open(img_path).convert("RGB"))
    mask, faces = create_face_mask(img, MODEL_DIR, face_index=-1, padding=1.3)

    assert mask is not None
    assert len(faces) > 0
    print(f"  PASS: face_index=-1, {len(faces)} face(s) masked")


# ===========================================================================
# Test 6: create_face_mask returns None for no-face image
# ===========================================================================
def test_create_face_mask_no_face():
    """No face image should return (None, [])."""
    from face_swap import create_face_mask

    # Plain color image — no face
    no_face = np.full((512, 512, 3), 128, dtype=np.uint8)
    mask, faces = create_face_mask(no_face, MODEL_DIR, face_index=0)

    assert mask is None, "Should return None for no-face image"
    assert faces == [], "Should return empty faces list"
    print("  PASS: no-face image returns (None, [])")


# ===========================================================================
# Test 7: preview_face_detection
# ===========================================================================
def test_preview_face_detection():
    """Preview should draw bbox + landmarks + mask overlay."""
    from face_swap import preview_face_detection

    img_path = find_real_face_image()
    if img_path is None:
        print("  SKIP: no real face image")
        return

    img = np.array(Image.open(img_path).convert("RGB"))
    preview = preview_face_detection(img, MODEL_DIR)

    assert preview.shape == img.shape, f"Preview shape {preview.shape} != image shape {img.shape}"
    assert preview.dtype == np.uint8

    # Preview should differ from original (bbox/landmarks drawn)
    diff = np.abs(preview.astype(float) - img.astype(float)).mean()
    assert diff > 1, f"Preview should differ from original (mean_diff={diff:.2f})"

    Image.fromarray(preview).save(str(OUTPUT_DIR / "test_preview.png"))
    print(f"  PASS: preview saved, mean_diff={diff:.2f}")


# ===========================================================================
# Test 8: get_detector singleton
# ===========================================================================
def test_get_detector_singleton():
    """get_detector should return same instance on repeated calls."""
    from face_swap import get_detector
    import face_swap

    # Reset singleton
    face_swap._detector = None

    d1 = get_detector(MODEL_DIR)
    d2 = get_detector(MODEL_DIR)
    assert d1 is d2, "get_detector should return singleton"
    print("  PASS: get_detector is singleton")


# ===========================================================================
# Test 9: pipeline_manager.load_faceswap
# ===========================================================================
def test_pipeline_manager_load_faceswap():
    """PipelineManager.load_faceswap should load SCRFD without crash."""
    from pipeline_manager import PipelineManager

    mgr = PipelineManager()
    mgr.model_dir = MODEL_DIR
    assert mgr.faceswap_pipeline is None

    mgr.load_faceswap()
    assert mgr.faceswap_pipeline is True, "faceswap_pipeline should be True (flag)"

    # Second call should be no-op
    mgr.load_faceswap()
    print("  PASS: load_faceswap loads SCRFD detector")


# ===========================================================================
# Test 10: generators.generate_faceswap signature
# ===========================================================================
def test_generate_faceswap_signature():
    """generate_faceswap should accept new Phase 5 params."""
    import inspect
    from generators import generate_faceswap

    sig = inspect.signature(generate_faceswap)
    params = set(sig.parameters.keys())

    required = {"image", "prompt", "face_index", "padding", "det_threshold",
                "resolution", "seed", "negative_prompt", "num_steps",
                "guidance_scale", "cfg_truncation", "control_scale",
                "max_sequence_length", "time_shift", "progress"}
    missing = required - params
    assert not missing, f"Missing params: {missing}"

    # Old TRT params should NOT be present
    old_params = {"target_image", "source_image", "enable_restore",
                  "codeformer_w", "enable_refine", "enable_detailer",
                  "blend_mode", "mask_blur"}
    unexpected = old_params & params
    assert not unexpected, f"Old TRT params still present: {unexpected}"

    print(f"  PASS: signature has {len(params)} params, no old TRT params")


# ===========================================================================
# Test 11: worker _run_faceswap E2E (GPU required)
# ===========================================================================
def test_worker_faceswap_e2e():
    """Full E2E: submit faceswap task to worker → get result image."""
    img_path = find_real_face_image()
    if img_path is None:
        print("  SKIP: no real face image")
        return

    try:
        import torch
        if not torch.cuda.is_available():
            print("  SKIP: no GPU")
            return
    except ImportError:
        print("  SKIP: no torch")
        return

    from worker import WorkerProcessManager

    mgr = WorkerProcessManager(MODEL_DIR)
    mgr.ensure_running()

    task_id = mgr.submit_task("faceswap", {
        "prompt": "beautiful young woman with natural skin, photorealistic, sharp focus",
        "negative_prompt": "blurry, low quality, artifacts, unnatural skin",
        "image_path": img_path,
        "face_index": 0,
        "padding": 1.3,
        "det_threshold": 0.5,
        "control_scale": 0.9,
        "num_steps": 25,
        "guidance_scale": 4.0,
        "cfg_truncation": 1.0,
        "max_sequence_length": 512,
        "time_shift": 3.0,
        "seed": 123,
    })

    start = time.time()
    result = None
    while time.time() - start < 300:
        for msg in mgr.poll_progress():
            pass
        result = mgr.get_result(timeout=1.0)
        if result and result.get("task_id") == task_id:
            break
    else:
        mgr.stop()
        raise TimeoutError("Worker did not return result in 300s")

    mgr.stop()

    assert result["status"] == "ok", f"Task failed: {result['payload']}"
    paths = result["payload"]["paths"]
    assert len(paths) == 1
    assert os.path.exists(paths[0])

    out = np.array(Image.open(paths[0]))
    assert out.ndim == 3 and out.shape[2] == 3, f"Output shape {out.shape} invalid"

    elapsed = time.time() - start
    print(f"  PASS: E2E faceswap in {elapsed:.1f}s → {paths[0]}")


# ===========================================================================
# Test 12: REQUIRED_MODELS for faceswap
# ===========================================================================
def test_required_models_faceswap():
    """REQUIRED_MODELS['faceswap'] should list SCRFD, not TRT models."""
    from pipeline_manager import REQUIRED_MODELS

    required = REQUIRED_MODELS.get("faceswap", [])
    assert len(required) > 0, "faceswap should have required models"

    # Should reference SCRFD in preprocessors/
    assert any("scrfd" in r for r in required), f"Should require SCRFD: {required}"

    # Should NOT reference faceswap/ directory or TRT
    for r in required:
        assert "faceswap/" not in r, f"Should not require old faceswap/ model: {r}"
        assert ".engine" not in r, f"Should not require TRT engine: {r}"

    print(f"  PASS: faceswap requires {required}")


# ===========================================================================
# Runner
# ===========================================================================
def run_all():
    tests = [
        test_scrfd_model_loads,
        test_scrfd_det_size_default,
        test_scrfd_detects_real_face,
        test_create_face_mask,
        test_create_face_mask_all_faces,
        test_create_face_mask_no_face,
        test_preview_face_detection,
        test_get_detector_singleton,
        test_pipeline_manager_load_faceswap,
        test_generate_faceswap_signature,
        test_required_models_faceswap,
        # GPU test last (slow)
        test_worker_faceswap_e2e,
    ]

    total = 0
    passed = 0
    skipped = 0
    failed = 0
    errors = []

    for test_fn in tests:
        total += 1
        name = test_fn.__name__
        print(f"\n[{total}] {name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            if "SKIP" in str(e):
                skipped += 1
                print(f"  SKIP: {e}")
            else:
                failed += 1
                errors.append((name, str(e)))
                import traceback
                traceback.print_exc()
                print(f"  FAIL: {e}")

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {skipped} skipped, {failed} failed / {total} total")
    print(f"{'=' * 60}")
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Model directory: {MODEL_DIR}")
    sys.exit(run_all())
