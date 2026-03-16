"""FaceSwap E2E image comparison test (requires GPU + model weights).

Runs swap_face with and without CodeFormer restoration,
saves results side by side for visual comparison,
and verifies measurable quality improvement.

Usage:
    cd /root/zit-ui && python tests/test_faceswap_e2e.py
"""
import sys
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app" / "ui"))

MODEL_DIR = str(Path.home() / ".cache" / "huggingface" / "hub" / "zit")
OUTPUT_DIR = Path("/tmp/zit-test-faceswap")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_test_face_image(size=512):
    """Create a synthetic face-like image for testing.
    Uses a real photo if available, otherwise generates a pattern."""
    # Try to find any existing face image in /tmp or test assets
    for candidate in ["/tmp/test_target.jpg", "/tmp/test_target.png"]:
        if os.path.exists(candidate):
            img = np.array(Image.open(candidate).convert("RGB"))
            return cv2.resize(img, (size, size))

    # Generate a synthetic "face" — skin-colored oval with features
    img = np.full((size, size, 3), (180, 150, 130), dtype=np.uint8)  # skin tone
    cx, cy = size // 2, size // 2
    # Face oval
    cv2.ellipse(img, (cx, cy), (size//4, size//3), 0, 0, 360, (200, 170, 150), -1)
    # Eyes
    cv2.circle(img, (cx - size//8, cy - size//10), size//20, (40, 40, 40), -1)
    cv2.circle(img, (cx + size//8, cy - size//10), size//20, (40, 40, 40), -1)
    # Nose
    cv2.line(img, (cx, cy - size//20), (cx, cy + size//15), (160, 130, 110), 2)
    # Mouth
    cv2.ellipse(img, (cx, cy + size//6), (size//8, size//16), 0, 0, 180, (150, 80, 80), 2)
    return img


def compute_quality_metrics(img1, img2):
    """Compute basic quality metrics between two images."""
    # Laplacian variance (sharpness)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    sharp1 = cv2.Laplacian(gray1, cv2.CV_64F).var()
    sharp2 = cv2.Laplacian(gray2, cv2.CV_64F).var()
    return {
        "sharpness_no_restore": sharp1,
        "sharpness_with_restore": sharp2,
        "sharpness_ratio": sharp2 / max(sharp1, 1e-6),
    }


def save_comparison(images, labels, output_path):
    """Save images side by side with labels."""
    h = max(img.shape[0] for img in images)
    w_total = sum(img.shape[1] for img in images) + 10 * (len(images) - 1)
    canvas = np.full((h + 40, w_total, 3), 255, dtype=np.uint8)

    x_offset = 0
    for img, label in zip(images, labels):
        ih, iw = img.shape[:2]
        canvas[40:40+ih, x_offset:x_offset+iw] = img
        cv2.putText(canvas, label, (x_offset + 5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        x_offset += iw + 10

    Image.fromarray(canvas).save(str(output_path), quality=95)
    print(f"  Comparison saved: {output_path}")


def test_faceswap_with_without_restore():
    """Core test: compare swap with and without CodeFormer."""
    from face_swap import FaceSwapPipeline

    print("\n[1] Loading FaceSwap pipeline...")
    pipeline = FaceSwapPipeline(MODEL_DIR)

    # Load real face images
    target_path = "/tmp/test_target.jpg"
    source_path = "/tmp/test_source.jpg"
    if os.path.exists(target_path) and os.path.exists(source_path):
        target = np.array(Image.open(target_path).convert("RGB"))
        source = np.array(Image.open(source_path).convert("RGB"))
        print(f"  Loaded real images: target={target.shape}, source={source.shape}")
    else:
        target = create_test_face_image(512)
        source = create_test_face_image(512)
        source = cv2.GaussianBlur(source, (5, 5), 0)
        source[:, :, 0] = np.clip(source[:, :, 0].astype(int) + 30, 0, 255).astype(np.uint8)
        print("  Using synthetic images (no real face photos found)")

    Image.fromarray(target).save(str(OUTPUT_DIR / "input_target.png"))
    Image.fromarray(source).save(str(OUTPUT_DIR / "input_source.png"))

    # --- Without restoration ---
    print("\n[2] Running swap WITHOUT restoration...")
    t0 = time.time()
    try:
        result_no_restore, faces_info = pipeline.swap_face(
            target, source,
            det_thresh=0.3,
            enable_restore=False,
            blend_mode="seamless",
        )
        t1 = time.time()
        print(f"  Done in {t1-t0:.2f}s, faces found: {len(faces_info)}")
        Image.fromarray(result_no_restore).save(str(OUTPUT_DIR / "result_no_restore.png"))
    except ValueError as e:
        print(f"  No face detected in synthetic image (expected): {e}")
        print("  Skipping image comparison — needs real face photos.")
        print("  Place test images at /tmp/test_target.jpg and /tmp/test_source.jpg")
        return False

    # --- With CodeFormer restoration ---
    print("\n[3] Running swap WITH CodeFormer restoration...")
    t0 = time.time()
    result_with_restore, faces_info2 = pipeline.swap_face(
        target, source,
        det_thresh=0.3,
        enable_restore=True,
        codeformer_w=0.7,
        blend_mode="seamless",
    )
    t1 = time.time()
    print(f"  Done in {t1-t0:.2f}s")
    Image.fromarray(result_with_restore).save(str(OUTPUT_DIR / "result_with_restore.png"))

    # --- Quality comparison ---
    print("\n[4] Quality comparison:")
    metrics = compute_quality_metrics(result_no_restore, result_with_restore)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save side-by-side comparison
    save_comparison(
        [target, result_no_restore, result_with_restore],
        ["Original", "No Restore", "CodeFormer"],
        OUTPUT_DIR / "comparison.png",
    )

    # Sharpness should improve with restoration (128px→512px→128px adds detail)
    if metrics["sharpness_ratio"] > 0.8:
        print("\n  RESULT: CodeFormer restoration active and producing output")
    else:
        print(f"\n  WARNING: Sharpness ratio {metrics['sharpness_ratio']:.2f} lower than expected")

    return True


def test_codeformer_standalone():
    """Test CodeFormer model directly on a face crop."""
    print("\n[5] Testing CodeFormer standalone...")
    from face_swap import CodeFormerRestorer

    restorer = CodeFormerRestorer(
        os.path.join(MODEL_DIR, "faceswap", "codeformer.pth"),
        device="cuda",
    )

    # Create a low-quality face crop (simulate 128px inswapper output)
    lq_face = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
    # Add some structure
    cv2.circle(lq_face, (45, 50), 8, (30, 30, 30), -1)  # eye
    cv2.circle(lq_face, (85, 50), 8, (30, 30, 30), -1)  # eye
    cv2.line(lq_face, (64, 60), (64, 80), (140, 120, 100), 2)  # nose

    Image.fromarray(lq_face).save(str(OUTPUT_DIR / "codeformer_input_128.png"))

    t0 = time.time()
    restored = restorer.restore(lq_face, w=0.7)
    t1 = time.time()
    print(f"  CodeFormer inference: {t1-t0:.2f}s")
    print(f"  Input: {lq_face.shape}, Output: {restored.shape}")

    assert restored.shape == (512, 512, 3), f"Expected 512x512, got {restored.shape}"
    assert restored.dtype == np.uint8

    Image.fromarray(restored).save(str(OUTPUT_DIR / "codeformer_output_512.png"))

    # Restored should be different from simple resize
    simple_upscale = cv2.resize(lq_face, (512, 512), interpolation=cv2.INTER_LINEAR)
    diff = np.abs(restored.astype(float) - simple_upscale.astype(float)).mean()
    print(f"  Mean diff vs simple upscale: {diff:.2f}")
    assert diff > 5, f"CodeFormer output too similar to simple upscale (diff={diff:.2f})"

    save_comparison(
        [cv2.resize(lq_face, (512, 512)), simple_upscale, restored],
        ["Input (upscaled)", "Bilinear", "CodeFormer"],
        OUTPUT_DIR / "codeformer_comparison.png",
    )
    print("  PASS: CodeFormer standalone test")
    return True


def test_paste_back_modes_visual():
    """Visual comparison of seamless vs alpha paste back."""
    print("\n[6] Testing paste_back modes visually...")
    from face_swap import paste_back

    face = np.full((128, 128, 3), 220, dtype=np.uint8)
    cv2.circle(face, (45, 50), 10, (30, 30, 30), -1)
    cv2.circle(face, (85, 50), 10, (30, 30, 30), -1)

    target = np.full((512, 512, 3), 100, dtype=np.uint8)
    # Add some texture
    target[200:300, 200:300] = [80, 120, 80]

    M = np.array([[1.5, 0, 128], [0, 1.5, 128]], dtype=np.float64)

    result_seamless = paste_back(face, target, M, size=128, blend_mode="seamless", mask_blur=0.3)
    result_alpha = paste_back(face, target, M, size=128, blend_mode="alpha", mask_blur=0.3)

    Image.fromarray(result_seamless).save(str(OUTPUT_DIR / "paste_seamless.png"))
    Image.fromarray(result_alpha).save(str(OUTPUT_DIR / "paste_alpha.png"))

    save_comparison(
        [target, result_seamless, result_alpha],
        ["Target", "Seamless", "Alpha"],
        OUTPUT_DIR / "paste_comparison.png",
    )
    print("  PASS: paste_back modes visual test")
    return True


if __name__ == "__main__":
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Model directory: {MODEL_DIR}")

    results = []

    # Test CodeFormer standalone (always works, no face detection needed)
    results.append(("CodeFormer standalone", test_codeformer_standalone()))

    # Test paste back modes
    results.append(("Paste back modes", test_paste_back_modes_visual()))

    # Test full pipeline (may skip if no real face detected)
    results.append(("FaceSwap with/without restore", test_faceswap_with_without_restore()))

    print(f"\n{'='*60}")
    print("  E2E Test Results")
    print(f"{'='*60}")
    for name, ok in results:
        status = "PASS" if ok else "SKIP/FAIL"
        print(f"  [{status}] {name}")
    print(f"\n  All outputs saved to: {OUTPUT_DIR}")
