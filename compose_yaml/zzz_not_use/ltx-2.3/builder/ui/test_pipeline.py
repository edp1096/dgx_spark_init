"""CLI test script for LTX-2 pipelines — organized by UI tabs.

Tests each pipeline through PipelineManager (same path as the real worker).

Usage:
    python test_pipeline.py                        # Test all available pipelines
    python test_pipeline.py --tab distilled        # Test specific tab
    python test_pipeline.py --tab ti2vid --frames 9  # Quick test (9 frames)
    python test_pipeline.py --list                 # List available pipelines
"""

import argparse
import gc
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from config import MODEL_DIR as DEFAULT_MODEL_DIR

OUTPUT_DIR = Path("/tmp/ltx2-test-outputs")
MODEL_DIR = DEFAULT_MODEL_DIR

# IC-LoRA model files (from pipeline_manager.py)
IC_LORA_MAP = {
    "Union Control": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    "Motion Track": "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors",
}

MODEL_FILES = {
    "dev": ("ltx-2.3-22b-dev-fp8.safetensors", "~28GB"),
    "distilled": ("ltx-2.3-22b-distilled-fp8.safetensors", "~26GB"),
    "upscaler": ("ltx-2.3-spatial-upscaler-x2-1.0.safetensors", "~950MB"),
    "distilled_lora": ("ltx-2.3-22b-distilled-lora-384.safetensors", "~7.1GB"),
    "gemma": ("gemma-3-12b-it-qat-q4_0-unquantized", "~12GB"),
}


def check_models() -> dict[str, bool]:
    """Check which model files are available."""
    return {k: (MODEL_DIR / fname).exists() for k, (fname, _) in MODEL_FILES.items()}


def check_iclora_models() -> dict[str, bool]:
    """Check which IC-LoRA model files are available."""
    return {k: (MODEL_DIR / fname).exists() for k, fname in IC_LORA_MAP.items()}


def format_size(size_bytes: int) -> str:
    if size_bytes >= 1024**3:
        return f"{size_bytes / 1024**3:.1f}GB"
    return f"{size_bytes / 1024**2:.0f}MB"


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class PreviewCapture:
    """Captures Stage 1 preview callback data for testing."""
    def __init__(self):
        self.called = False
        self.preview_path = None

    def callback(self, video_frames, audio, frame_rate, num_frames):
        from ltx_core.model.video_vae import get_video_chunks_number
        from ltx_pipelines.utils.media_io import encode_video
        self.called = True
        self.preview_path = str(OUTPUT_DIR / "_stage1_preview.mp4")
        encode_video(
            video=video_frames, fps=frame_rate,
            audio=audio, output_path=self.preview_path,
            video_chunks_number=get_video_chunks_number(num_frames),
        )
        size_kb = Path(self.preview_path).stat().st_size / 1024
        print(f"  [PREVIEW] Stage 1 preview saved: {self.preview_path} ({size_kb:.0f}KB)")


def mem_log(label: str):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  [MEM] {label}: {alloc:.1f}GB alloc / {reserved:.1f}GB reserved")


def _make_dummy_image(width: int = 768, height: int = 512) -> str:
    """Create a dummy PNG image for conditioning tests."""
    from PIL import Image
    img = Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8))
    path = str(OUTPUT_DIR / "test_dummy_image.png")
    img.save(path)
    return path


def _make_dummy_audio(duration: float = 1.0, sr: int = 16000) -> str:
    """Create a dummy stereo WAV file for a2vid test."""
    import wave
    import struct
    path = str(OUTPUT_DIR / "test_dummy_audio.wav")
    n_samples = int(sr * duration)
    with wave.open(path, "w") as wf:
        wf.setnchannels(2)  # stereo — audio encoder expects 2 channels
        wf.setsampwidth(2)
        wf.setframerate(sr)
        # Simple sine wave (L + R)
        data = b""
        for i in range(n_samples):
            sample = int(16000 * np.sin(2 * np.pi * 440 * i / sr))
            data += struct.pack("<hh", sample, sample)  # L, R
        wf.writeframes(data)
    return path


# ---------------------------------------------------------------------------
# Tab 1: Text/Image -> Video (ti2vid)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def test_ti2vid(mgr, frames: int = 9, steps: int = 10) -> bool:
    """Tab 1: TI2VidTwoStagesPipeline (euler sampler)."""
    print("\n" + "=" * 60)
    print("TAB 1: Text/Image -> Video (ti2vid, euler)")
    print("=" * 60)

    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.model.video_vae import get_video_chunks_number
    from ltx_pipelines.utils.media_io import encode_video

    pipeline = mgr.get_ti2vid(sampler="euler")
    mem_log("after pipeline init")

    video_guider = MultiModalGuiderParams(
        cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7,
        modality_scale=3.0, stg_blocks=[28],
    )
    audio_guider = MultiModalGuiderParams(
        cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7,
        modality_scale=3.0, stg_blocks=[28],
    )

    preview = PreviewCapture()
    print(f"Generating... ({frames} frames, {steps} steps, 768x512)")
    t0 = time.time()
    video_frames, audio = pipeline(
        prompt="A golden retriever puppy runs through a sunlit meadow.",
        negative_prompt="low quality, blurry",
        seed=42,
        height=512, width=768,
        num_frames=frames,
        frame_rate=25.0,
        num_inference_steps=steps,
        video_guider_params=video_guider,
        audio_guider_params=audio_guider,
        images=[],
        enhance_prompt=False,
        stage1_preview_callback=preview.callback,
    )
    print(f"  Generated in {time.time() - t0:.1f}s")
    print(f"  Stage 1 preview callback fired: {preview.called}")

    output_path = str(OUTPUT_DIR / "test_ti2vid.mp4")
    encode_video(video=video_frames, fps=25.0, audio=audio, output_path=output_path,
                 video_chunks_number=get_video_chunks_number(frames))
    print(f"  Saved: {output_path} ({Path(output_path).stat().st_size / 1024:.0f}KB)")
    mem_log("after encoding")

    mgr._cleanup()
    cleanup()
    return True


# ---------------------------------------------------------------------------
# Tab 2: Distilled (Fast)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def test_distilled(mgr, frames: int = 9, height: int = 512, width: int = 768) -> bool:
    """Tab 2: DistilledPipeline (8-step fixed)."""
    print("\n" + "=" * 60)
    print("TAB 2: Distilled (Fast)")
    print("=" * 60)

    from ltx_core.model.video_vae import get_video_chunks_number
    from ltx_pipelines.utils.media_io import encode_video

    pipeline = mgr.get_distilled()
    mem_log("after pipeline init")

    preview = PreviewCapture()
    print(f"Generating... ({frames} frames, 8 steps fixed, {height}x{width})")
    t0 = time.time()
    video_frames, audio = pipeline(
        prompt="A golden retriever puppy runs through a sunlit meadow.",
        seed=42,
        height=height, width=width,
        num_frames=frames,
        frame_rate=25.0,
        images=[],
        enhance_prompt=False,
        stage1_preview_callback=preview.callback,
    )
    print(f"  Generated in {time.time() - t0:.1f}s")
    print(f"  Stage 1 preview callback fired: {preview.called}")
    mem_log("after generation")

    output_path = str(OUTPUT_DIR / "test_distilled.mp4")
    encode_video(video=video_frames, fps=25.0, audio=audio, output_path=output_path,
                 video_chunks_number=get_video_chunks_number(frames))
    print(f"  Saved: {output_path} ({Path(output_path).stat().st_size / 1024:.0f}KB)")
    mem_log("after encoding")

    mgr._cleanup()
    cleanup()
    return True


# ---------------------------------------------------------------------------
# Tab 3: IC-LoRA
# ---------------------------------------------------------------------------
@torch.inference_mode()
def test_iclora(mgr, frames: int = 9, lora_choice: str | None = None) -> bool:
    """Tab 3: ICLoraPipeline (requires IC-LoRA model + reference video)."""
    print("\n" + "=" * 60)
    print("TAB 3: IC-LoRA")
    print("=" * 60)

    # Find first available IC-LoRA model
    if lora_choice is None:
        for name, fname in IC_LORA_MAP.items():
            if (MODEL_DIR / fname).exists():
                lora_choice = name
                break
    if lora_choice is None:
        print("  SKIP: No IC-LoRA model files found.")
        return False

    lora_filename = IC_LORA_MAP[lora_choice]
    lora_path = str(MODEL_DIR / lora_filename)
    print(f"  Using LoRA: {lora_choice} ({lora_filename})")

    # Need a reference video — use previous test output or generate one
    ref_video = str(OUTPUT_DIR / "test_distilled.mp4")
    if not Path(ref_video).exists():
        ref_video = str(OUTPUT_DIR / "test_ti2vid.mp4")
    if not Path(ref_video).exists():
        print("  SKIP: No reference video. Run distilled or ti2vid first.")
        return False

    from ltx_core.model.video_vae import get_video_chunks_number
    from ltx_pipelines.utils.media_io import encode_video

    pipeline = mgr.get_iclora(lora_path=lora_path)
    mem_log("after pipeline init")

    preview = PreviewCapture()
    print(f"Generating... ({frames} frames, ref={Path(ref_video).name})")
    t0 = time.time()
    video_frames, audio = pipeline(
        prompt="A golden retriever puppy runs through a sunlit meadow.",
        seed=42,
        height=512, width=768,
        num_frames=frames,
        frame_rate=25.0,
        images=[],
        video_conditioning=[(ref_video, 0.5)],
        conditioning_attention_strength=1.0,
        skip_stage_2=False,
        enhance_prompt=False,
        stage1_preview_callback=preview.callback,
    )
    print(f"  Generated in {time.time() - t0:.1f}s")
    print(f"  Stage 1 preview callback fired: {preview.called}")
    mem_log("after generation")

    output_path = str(OUTPUT_DIR / "test_iclora.mp4")
    encode_video(video=video_frames, fps=25.0, audio=audio, output_path=output_path,
                 video_chunks_number=get_video_chunks_number(frames))
    print(f"  Saved: {output_path} ({Path(output_path).stat().st_size / 1024:.0f}KB)")
    mem_log("after encoding")

    mgr._cleanup()
    cleanup()
    return True


# ---------------------------------------------------------------------------
# Tab 4: Keyframe Interpolation
# ---------------------------------------------------------------------------
@torch.inference_mode()
def test_keyframe(mgr, frames: int = 9, steps: int = 10) -> bool:
    """Tab 4: KeyframeInterpolationPipeline (requires 2+ keyframe images)."""
    print("\n" + "=" * 60)
    print("TAB 4: Keyframe Interpolation")
    print("=" * 60)

    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.model.video_vae import get_video_chunks_number
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.media_io import encode_video

    pipeline = mgr.get_keyframe()
    mem_log("after pipeline init")

    # Create 2 dummy keyframe images
    kf1 = _make_dummy_image(768, 512)
    kf2 = _make_dummy_image(768, 512)
    images = [
        ImageConditioningInput(kf1, 0, 1.0, 23),
        ImageConditioningInput(kf2, frames - 1, 1.0, 23),
    ]

    video_guider = MultiModalGuiderParams(
        cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7,
        modality_scale=3.0, stg_blocks=[28],
    )
    audio_guider = MultiModalGuiderParams(
        cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7,
        modality_scale=3.0, stg_blocks=[28],
    )

    preview = PreviewCapture()
    print(f"Generating... ({frames} frames, {steps} steps, 2 keyframes, 768x512)")
    t0 = time.time()
    video_frames, audio = pipeline(
        prompt="Smooth transition between two scenes in a natural landscape.",
        negative_prompt="low quality, blurry",
        seed=42,
        height=512, width=768,
        num_frames=frames,
        frame_rate=25.0,
        num_inference_steps=steps,
        video_guider_params=video_guider,
        audio_guider_params=audio_guider,
        images=images,
        enhance_prompt=False,
        stage1_preview_callback=preview.callback,
    )
    print(f"  Generated in {time.time() - t0:.1f}s")
    print(f"  Stage 1 preview callback fired: {preview.called}")
    mem_log("after generation")

    output_path = str(OUTPUT_DIR / "test_keyframe.mp4")
    encode_video(video=video_frames, fps=25.0, audio=audio, output_path=output_path,
                 video_chunks_number=get_video_chunks_number(frames))
    print(f"  Saved: {output_path} ({Path(output_path).stat().st_size / 1024:.0f}KB)")
    mem_log("after encoding")

    mgr._cleanup()
    cleanup()
    return True


# ---------------------------------------------------------------------------
# Tab 5: Audio -> Video (a2vid)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def test_a2vid(mgr, frames: int = 9, steps: int = 10) -> bool:
    """Tab 5: A2VidPipelineTwoStage (audio-conditioned video generation)."""
    print("\n" + "=" * 60)
    print("TAB 5: Audio -> Video (a2vid)")
    print("=" * 60)

    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.model.video_vae import get_video_chunks_number
    from ltx_pipelines.utils.media_io import encode_video

    pipeline = mgr.get_a2vid()
    mem_log("after pipeline init")

    # Create dummy audio
    audio_path = _make_dummy_audio(duration=1.0)
    print(f"  Using dummy audio: {audio_path}")

    video_guider = MultiModalGuiderParams(
        cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7,
        modality_scale=3.0, stg_blocks=[28],
    )

    preview = PreviewCapture()
    print(f"Generating... ({frames} frames, {steps} steps, audio-conditioned, 768x512)")
    t0 = time.time()
    video_frames, audio = pipeline(
        prompt="A person playing piano in a concert hall.",
        negative_prompt="low quality, blurry",
        seed=42,
        height=512, width=768,
        num_frames=frames,
        frame_rate=25.0,
        num_inference_steps=steps,
        video_guider_params=video_guider,
        images=[],
        audio_path=audio_path,
        audio_start_time=0,
        audio_max_duration=None,
        enhance_prompt=False,
        stage1_preview_callback=preview.callback,
    )
    print(f"  Generated in {time.time() - t0:.1f}s")
    print(f"  Stage 1 preview callback fired: {preview.called}")
    mem_log("after generation")

    output_path = str(OUTPUT_DIR / "test_a2vid.mp4")
    encode_video(video=video_frames, fps=25.0, audio=audio, output_path=output_path,
                 video_chunks_number=get_video_chunks_number(frames))
    print(f"  Saved: {output_path} ({Path(output_path).stat().st_size / 1024:.0f}KB)")
    mem_log("after encoding")

    mgr._cleanup()
    cleanup()
    return True


# ---------------------------------------------------------------------------
# Tab 6: Retake
# ---------------------------------------------------------------------------
@torch.inference_mode()
def test_retake(mgr, source_video: str, steps: int = 10, distilled: bool = False) -> bool:
    """Tab 6: RetakePipeline (re-generate sections of existing video)."""
    mode = " (distilled)" if distilled else ""
    print("\n" + "=" * 60)
    print(f"TAB 6: Retake{mode}")
    print("=" * 60)

    if not source_video or not Path(source_video).exists():
        print("  SKIP: No source video available. Run ti2vid or distilled first.")
        return False

    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.model.video_vae import get_video_chunks_number
    from ltx_pipelines.utils.media_io import encode_video, get_videostream_metadata

    pipeline = mgr.get_retake(distilled=distilled)
    mem_log("after pipeline init")

    video_guider = MultiModalGuiderParams(
        cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7,
        modality_scale=3.0, stg_blocks=[28],
    )
    audio_guider = MultiModalGuiderParams(
        cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7,
        modality_scale=3.0, stg_blocks=[28],
    )

    gen_kwargs = dict(
        video_path=source_video,
        prompt="A golden retriever puppy runs through a sunlit meadow with butterflies.",
        start_time=0.0,
        end_time=0.2,
        seed=42,
        negative_prompt="low quality",
        num_inference_steps=steps,
        regenerate_video=True,
        regenerate_audio=True,
        enhance_prompt=False,
        distilled=distilled,
    )
    if not distilled:
        gen_kwargs["video_guider_params"] = video_guider
        gen_kwargs["audio_guider_params"] = audio_guider

    print(f"Retaking... (0.0-0.2s, {steps} steps)")
    t0 = time.time()
    video_frames, audio_tensor = pipeline(**gen_kwargs)
    print(f"  Generated in {time.time() - t0:.1f}s")
    mem_log("after generation")

    src_fps, src_num_frames, _, _ = get_videostream_metadata(source_video)
    output_path = str(OUTPUT_DIR / "test_retake.mp4")
    encode_video(video=video_frames, fps=src_fps, audio=audio_tensor, output_path=output_path,
                 video_chunks_number=get_video_chunks_number(src_num_frames))
    print(f"  Saved: {output_path} ({Path(output_path).stat().st_size / 1024:.0f}KB)")
    mem_log("after encoding")

    mgr._cleanup()
    cleanup()
    return True


# ---------------------------------------------------------------------------
# Pipeline registry (matches UI tabs)
# ---------------------------------------------------------------------------
TABS = {
    "distilled":  {"requires": ["distilled", "upscaler", "gemma"]},
    "ti2vid":     {"requires": ["dev", "upscaler", "distilled_lora", "gemma"]},
    "keyframe":   {"requires": ["dev", "upscaler", "distilled_lora", "gemma"]},
    "a2vid":      {"requires": ["dev", "upscaler", "distilled_lora", "gemma"]},
    "retake":     {"requires": ["dev", "gemma"]},
    "iclora":     {"requires": ["distilled", "upscaler", "gemma"]},  # + IC-LoRA file
}


def main() -> None:
    global MODEL_DIR

    tab_names = list(TABS.keys())

    parser = argparse.ArgumentParser(description="LTX-2 Pipeline CLI Test (by UI tab)")
    parser.add_argument("--tab", "-t", choices=tab_names, help="Test specific tab")
    parser.add_argument("--frames", type=int, default=9, help="Number of frames (8k+1, default: 9)")
    parser.add_argument("--steps", type=int, default=10, help="Inference steps (default: 10)")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR), help="Model directory")
    parser.add_argument("--resolution", type=str, default="512x768", help="Resolution HxW (default: 512x768)")
    parser.add_argument("--list", action="store_true", help="List available pipelines")
    args = parser.parse_args()

    MODEL_DIR = Path(args.model_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check models
    avail = check_models()
    iclora_avail = check_iclora_models()
    print(f"Model directory: {MODEL_DIR}")
    print(f"Model status:")
    for key, (fname, expected_size) in MODEL_FILES.items():
        path = MODEL_DIR / fname
        if path.exists():
            size_str = "dir" if path.is_dir() else format_size(path.stat().st_size)
            print(f"  [  OK  ] {fname} ({size_str})")
        else:
            print(f"  [MISSING] {fname} (expected {expected_size})")

    # IC-LoRA models
    any_iclora = False
    for name, fname in IC_LORA_MAP.items():
        path = MODEL_DIR / fname
        if path.exists():
            any_iclora = True
            print(f"  [  OK  ] {fname} (IC-LoRA: {name})")

    if args.list:
        print("\nAvailable tabs:")
        for name, info in TABS.items():
            reqs = info["requires"]
            ok = all(avail.get(r, False) for r in reqs)
            if name == "iclora":
                ok = ok and any_iclora
            missing = [r for r in reqs if not avail.get(r, False)]
            if name == "iclora" and not any_iclora:
                missing.append("ic-lora-model")
            if ok:
                print(f"  {name}: [READY]")
            else:
                print(f"  {name}: [NOT READY] missing: {', '.join(missing)}")
        return

    # Determine which tabs to test
    if args.tab:
        targets = [args.tab]
    else:
        # Default order: distilled first (fastest), then others
        targets = []
        for name, info in TABS.items():
            reqs = info["requires"]
            ok = all(avail.get(r, False) for r in reqs)
            if name == "iclora":
                ok = ok and any_iclora
            if ok:
                targets.append(name)

    if not targets:
        print("\nNo tabs ready to test. Download models first.")
        sys.exit(1)

    h, w = args.resolution.split("x")
    height, width = int(h), int(w)

    print(f"\nTesting: {', '.join(targets)}")
    print(f"Frames: {args.frames}, Steps: {args.steps}, Resolution: {height}x{width}")
    print(f"Output: {OUTPUT_DIR}")

    # Create PipelineManager (same as worker process)
    from pipeline_manager import PipelineManager
    mgr = PipelineManager()
    mgr.model_dir = str(MODEL_DIR)

    results = {}
    source_video = None

    for name in targets:
        print(f"\n{'#' * 60}")
        print(f"# Testing: {name}")
        print(f"{'#' * 60}")
        try:
            if name == "distilled":
                ok = test_distilled(mgr, frames=args.frames, height=height, width=width)
                if ok:
                    source_video = str(OUTPUT_DIR / "test_distilled.mp4")

            elif name == "ti2vid":
                ok = test_ti2vid(mgr, frames=args.frames, steps=args.steps)
                if ok:
                    source_video = str(OUTPUT_DIR / "test_ti2vid.mp4")

            elif name == "keyframe":
                ok = test_keyframe(mgr, frames=args.frames, steps=args.steps)

            elif name == "a2vid":
                ok = test_a2vid(mgr, frames=args.frames, steps=args.steps)

            elif name == "retake":
                if source_video is None:
                    # Try to find existing test output
                    for candidate in ["test_distilled.mp4", "test_ti2vid.mp4"]:
                        p = OUTPUT_DIR / candidate
                        if p.exists():
                            source_video = str(p)
                            break
                ok = test_retake(mgr, source_video, steps=args.steps)

            elif name == "iclora":
                ok = test_iclora(mgr, frames=args.frames)

            else:
                print(f"  Unknown tab: {name}")
                ok = False

            results[name] = "PASS" if ok else "SKIP"

        except Exception as e:
            print(f"\n  FAIL: {e}")
            traceback.print_exc()
            results[name] = "FAIL"
            mgr._cleanup()
            cleanup()

    # Final cleanup
    mgr._cleanup()
    cleanup()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    total = len(results)
    passed = sum(1 for v in results.values() if v == "PASS")
    failed = sum(1 for v in results.values() if v == "FAIL")
    skipped = sum(1 for v in results.values() if v == "SKIP")
    for name, status in results.items():
        icon = {"PASS": "OK", "FAIL": "!!", "SKIP": "--"}[status]
        print(f"  [{icon}] {name}: {status}")
    print(f"\n  Total: {total} | Pass: {passed} | Fail: {failed} | Skip: {skipped}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
