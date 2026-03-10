"""CLI test script for LTX-2 pipelines.

Run after model download to verify pipeline functionality.

Usage:
    python test_pipeline.py                     # Test all available pipelines
    python test_pipeline.py --pipeline ti2vid   # Test specific pipeline
    python test_pipeline.py --pipeline distilled --frames 9  # Quick test (9 frames)
    python test_pipeline.py --list              # List available pipelines
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

from config import MODEL_DIR
OUTPUT_DIR = Path("/tmp/ltx2-test-outputs")


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


def format_size(size_bytes: int) -> str:
    if size_bytes >= 1024**3:
        return f"{size_bytes / 1024**3:.1f}GB"
    return f"{size_bytes / 1024**2:.0f}MB"


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_mem_gb() -> str:
    """Get current memory usage string."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"{alloc:.1f}GB alloc / {reserved:.1f}GB reserved"
    return "N/A"


def print_gpu_mem():
    print(f"  GPU: {get_mem_gb()}")


def mem_log(label: str):
    """Print memory with label for stage tracking."""
    print(f"  [MEM] {label}: {get_mem_gb()}")


@torch.inference_mode()
def test_ti2vid(frames: int = 9, steps: int = 10) -> bool:
    """Test TI2VidTwoStagesPipeline (dev model, 2-stage)."""
    print("\n" + "=" * 60)
    print("TEST: TI2VidTwoStagesPipeline")
    print("=" * 60)

    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_core.quantization import QuantizationPolicy
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
    from ltx_pipelines.utils.media_io import encode_video

    model_dir = str(MODEL_DIR)
    distilled_lora = [LoraPathStrengthAndSDOps(
        str(MODEL_DIR / "ltx-2.3-22b-distilled-lora-384.safetensors"),
        0.8,
        LTXV_LORA_COMFY_RENAMING_MAP,
    )]

    print("Loading pipeline...")
    t0 = time.time()
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=str(MODEL_DIR / "ltx-2.3-22b-dev-fp8.safetensors"),
        distilled_lora=distilled_lora,
        spatial_upsampler_path=str(MODEL_DIR / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
        gemma_root=str(MODEL_DIR / "gemma-3-12b-it-qat-q4_0-unquantized"),
        loras=[],
        quantization=QuantizationPolicy.fp8_cast(),
    )
    print(f"  Pipeline loaded in {time.time() - t0:.1f}s")
    print_gpu_mem()

    video_guider = MultiModalGuiderParams(
        cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7,
        modality_scale=3.0, stg_blocks=[28],
    )
    audio_guider = MultiModalGuiderParams(
        cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7,
        modality_scale=3.0, stg_blocks=[28],
    )

    print(f"Generating... ({frames} frames, {steps} steps, 512x768)")
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
    )
    gen_time = time.time() - t0
    print(f"  Generated in {gen_time:.1f}s")

    output_path = str(OUTPUT_DIR / "test_ti2vid.mp4")
    print("Encoding video...")
    from ltx_core.model.video_vae import get_video_chunks_number
    encode_video(video=video_frames, fps=25.0, audio=audio, output_path=output_path,
                 video_chunks_number=get_video_chunks_number(frames))
    print(f"  Saved: {output_path} ({Path(output_path).stat().st_size / 1024:.0f}KB)")
    print_gpu_mem()

    del pipeline
    cleanup()
    return True


def _q8_wrap_fp8_linears(model: torch.nn.Module) -> int:
    """Wrap existing nn.Linear layers (with FP8 weights) into FP8Linear for native FP8 GEMM.

    Expects weights already in float8_e4m3fn (from pre-converted checkpoint + fp8_cast loading).
    Uses use_hadamard=False since weights are already FP8 without Hadamard transform.
    """
    from q8_kernels.modules.linear import FP8Linear

    count = 0
    replacements = []

    # Collect replacements first (can't modify during iteration)
    for name, module in model.named_modules():
        if "transformer_block" not in name:
            continue
        if not isinstance(module, torch.nn.Linear):
            continue
        if isinstance(module, FP8Linear):
            continue
        replacements.append((name, module))

    for name, module in replacements:
        *parent_path, child_name = name.split(".")
        parent = model
        for part in parent_path:
            parent = getattr(parent, part)

        # Create FP8Linear wrapper — no Hadamard, just native FP8 GEMM
        fp8_layer = FP8Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            use_hadamard=False,
        )
        # Copy weights directly (already FP8 from checkpoint)
        w = module.weight.data
        if w.dtype != torch.float8_e4m3fn:
            w = w.to(torch.float8_e4m3fn)
        fp8_layer.weight.data = w
        if module.bias is not None:
            fp8_layer.bias.data = module.bias.data.to(torch.float32)

        setattr(parent, child_name, fp8_layer)
        count += 1

    gc.collect()
    torch.cuda.empty_cache()
    return count


def _patch_ledger_with_q8(pipeline):
    """Monkey-patch model_ledger.transformer() to wrap FP8 Linear layers with q8_kernels FP8Linear."""
    original_transformer_fn = pipeline.model_ledger.transformer

    def patched_transformer():
        mem_log("before transformer load")
        x0_model = original_transformer_fn()
        mem_log("after transformer load (FP8 via fp8_cast)")
        print("  Wrapping Linear→FP8Linear (q8_kernels, native FP8 GEMM)...")
        t0 = time.time()
        count = _q8_wrap_fp8_linears(x0_model.velocity_model)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        mem_log(f"after FP8Linear wrap ({count} layers, {time.time() - t0:.1f}s)")
        return x0_model

    pipeline.model_ledger.transformer = patched_transformer


@torch.inference_mode()
def test_distilled(frames: int = 9, fp8: bool = False, q8: bool = False, height: int = 512, width: int = 768) -> bool:
    """Test DistilledPipeline (distilled model, 8-step fixed)."""
    mode = " (q8_kernels)" if q8 else (" (FP8)" if fp8 else "")
    print("\n" + "=" * 60)
    print(f"TEST: DistilledPipeline{mode}")
    print("=" * 60)

    from ltx_core.quantization import QuantizationPolicy
    from ltx_pipelines.distilled import DistilledPipeline
    from ltx_pipelines.utils.media_io import encode_video

    # Determine checkpoint and quantization strategy
    ckpt_name = "ltx-2.3-22b-distilled.safetensors"
    quantization = None

    if q8 or fp8:
        # Use pre-converted FP8 checkpoint if available
        fp8_ckpt = MODEL_DIR / "ltx-2.3-22b-distilled-fp8.safetensors"
        if fp8_ckpt.exists():
            ckpt_name = fp8_ckpt.name
        # Full fp8_cast: sd_ops keeps weights FP8 during load, module_ops adds upcast forward
        # (q8 mode will replace upcast forward with native FP8 GEMM later)
        quantization = QuantizationPolicy.fp8_cast()

    print(f"  Checkpoint: {ckpt_name}")
    mem_log("before pipeline init")
    print("Loading pipeline...")
    t0 = time.time()
    pipeline = DistilledPipeline(
        distilled_checkpoint_path=str(MODEL_DIR / ckpt_name),
        gemma_root=str(MODEL_DIR / "gemma-3-12b-it-qat-q4_0-unquantized"),
        spatial_upsampler_path=str(MODEL_DIR / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
        loras=[],
        quantization=quantization,
    )
    print(f"  Pipeline loaded in {time.time() - t0:.1f}s")
    mem_log("after pipeline init")

    if q8:
        _patch_ledger_with_q8(pipeline)

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
    )
    gen_time = time.time() - t0
    print(f"  Generated in {gen_time:.1f}s")
    mem_log("after generation")

    output_path = str(OUTPUT_DIR / "test_distilled.mp4")
    print("Encoding video...")
    from ltx_core.model.video_vae import get_video_chunks_number
    encode_video(video=video_frames, fps=25.0, audio=audio, output_path=output_path,
                 video_chunks_number=get_video_chunks_number(frames))
    print(f"  Saved: {output_path} ({Path(output_path).stat().st_size / 1024:.0f}KB)")
    mem_log("after encoding")

    del pipeline
    cleanup()
    mem_log("after cleanup")
    return True


@torch.inference_mode()
def test_retake(source_video: str, frames: int = 9, steps: int = 10) -> bool:
    """Test RetakePipeline (dev model, 1-stage, requires source video)."""
    print("\n" + "=" * 60)
    print("TEST: RetakePipeline")
    print("=" * 60)

    if not source_video or not Path(source_video).exists():
        print("  SKIP: No source video available. Run ti2vid or distilled first.")
        return False

    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.quantization import QuantizationPolicy
    from ltx_pipelines.retake import RetakePipeline
    from ltx_pipelines.utils.media_io import encode_video

    print("Loading pipeline...")
    t0 = time.time()
    pipeline = RetakePipeline(
        checkpoint_path=str(MODEL_DIR / "ltx-2.3-22b-dev-fp8.safetensors"),
        gemma_root=str(MODEL_DIR / "gemma-3-12b-it-qat-q4_0-unquantized"),
        loras=[],
        quantization=QuantizationPolicy.fp8_cast(),
    )
    print(f"  Pipeline loaded in {time.time() - t0:.1f}s")
    print_gpu_mem()

    video_guider = MultiModalGuiderParams(
        cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7,
        modality_scale=3.0, stg_blocks=[28],
    )
    audio_guider = MultiModalGuiderParams(
        cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7,
        modality_scale=3.0, stg_blocks=[28],
    )

    print(f"Retaking... (0.0-0.2s, {steps} steps)")
    t0 = time.time()
    video_frames, audio_tensor = pipeline(
        video_path=source_video,
        prompt="A golden retriever puppy runs through a sunlit meadow with butterflies.",
        start_time=0.0,
        end_time=0.2,
        seed=42,
        negative_prompt="low quality",
        num_inference_steps=steps,
        video_guider_params=video_guider,
        audio_guider_params=audio_guider,
        regenerate_video=True,
        regenerate_audio=True,
        enhance_prompt=False,
    )
    gen_time = time.time() - t0
    print(f"  Generated in {gen_time:.1f}s")

    output_path = str(OUTPUT_DIR / "test_retake.mp4")
    print("Encoding video...")
    from ltx_core.model.video_vae import get_video_chunks_number
    encode_video(video=video_frames, fps=25.0, audio=audio_tensor, output_path=output_path,
                 video_chunks_number=get_video_chunks_number(frames))
    print(f"  Saved: {output_path} ({Path(output_path).stat().st_size / 1024:.0f}KB)")
    print_gpu_mem()

    del pipeline
    cleanup()
    return True


@torch.inference_mode()
def test_1stage(frames: int = 9, steps: int = 10, fp8: bool = False, q8: bool = False) -> bool:
    """Test TI2VidOneStagePipeline (dev model, 1-stage, no upscaler)."""
    mode = " (q8_kernels)" if q8 else (" (FP8)" if fp8 else "")
    print("\n" + "=" * 60)
    print(f"TEST: TI2VidOneStagePipeline{mode}")
    print("=" * 60)

    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.quantization import QuantizationPolicy
    from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
    from ltx_pipelines.utils.media_io import encode_video

    # Determine checkpoint and quantization strategy
    ckpt_name = "ltx-2.3-22b-dev.safetensors"
    quantization = None

    if q8 or fp8:
        fp8_ckpt = MODEL_DIR / "ltx-2.3-22b-dev-fp8.safetensors"
        if fp8_ckpt.exists():
            ckpt_name = fp8_ckpt.name
        quantization = QuantizationPolicy.fp8_cast()

    print(f"  Checkpoint: {ckpt_name}")
    mem_log("before pipeline init")
    print("Loading pipeline...")
    t0 = time.time()
    pipeline = TI2VidOneStagePipeline(
        checkpoint_path=str(MODEL_DIR / ckpt_name),
        gemma_root=str(MODEL_DIR / "gemma-3-12b-it-qat-q4_0-unquantized"),
        loras=[],
        quantization=quantization,
    )
    print(f"  Pipeline loaded in {time.time() - t0:.1f}s")
    mem_log("after pipeline init")

    if q8:
        _patch_ledger_with_q8(pipeline)

    video_guider = MultiModalGuiderParams(
        cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7,
        modality_scale=3.0, stg_blocks=[28],
    )
    audio_guider = MultiModalGuiderParams(
        cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7,
        modality_scale=3.0, stg_blocks=[28],
    )

    print(f"Generating... ({frames} frames, {steps} steps, 512x768)")
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
    )
    gen_time = time.time() - t0
    print(f"  Generated in {gen_time:.1f}s")
    mem_log("after generation")

    output_path = str(OUTPUT_DIR / "test_1stage.mp4")
    print("Encoding video...")
    from ltx_core.model.video_vae import get_video_chunks_number
    encode_video(video=video_frames, fps=25.0, audio=audio, output_path=output_path,
                 video_chunks_number=get_video_chunks_number(frames))
    print(f"  Saved: {output_path} ({Path(output_path).stat().st_size / 1024:.0f}KB)")
    mem_log("after encoding")

    del pipeline
    cleanup()
    mem_log("after cleanup")
    return True


PIPELINES = {
    "ti2vid": {"fn": test_ti2vid, "requires": ["dev", "upscaler", "distilled_lora", "gemma"]},
    "distilled": {"fn": test_distilled, "requires": ["distilled", "upscaler", "gemma"]},
    "1stage": {"fn": test_1stage, "requires": ["dev", "gemma"]},
    "retake": {"fn": test_retake, "requires": ["dev", "gemma"]},
}


def main() -> None:
    global MODEL_DIR

    parser = argparse.ArgumentParser(description="LTX-2 Pipeline CLI Test")
    parser.add_argument("--pipeline", "-p", choices=["ti2vid", "distilled", "1stage", "retake"], help="Test specific pipeline")
    parser.add_argument("--frames", type=int, default=9, help="Number of frames (8k+1, default: 9)")
    parser.add_argument("--steps", type=int, default=10, help="Inference steps (default: 10)")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR), help="Model directory")
    parser.add_argument("--fp8", action="store_true", help="Use FP8 quantization (halves memory)")
    parser.add_argument("--q8", action="store_true", help="Use q8_kernels (native FP8 GEMM + Hadamard)")
    parser.add_argument("--resolution", type=str, default="512x768", help="Resolution HxW (default: 512x768)")
    parser.add_argument("--list", action="store_true", help="List available pipelines")
    args = parser.parse_args()

    MODEL_DIR = Path(args.model_dir)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check models
    avail = check_models()
    print(f"Model directory: {MODEL_DIR}")
    print(f"Model status:")
    for key, (fname, expected_size) in MODEL_FILES.items():
        path = MODEL_DIR / fname
        if path.exists():
            if path.is_dir():
                size_str = "dir"
            else:
                size_str = format_size(path.stat().st_size)
            print(f"  [  OK  ] {fname} ({size_str})")
        else:
            print(f"  [MISSING] {fname} (expected {expected_size})")

    if args.list:
        print("\nAvailable pipelines:")
        for name, info in PIPELINES.items():
            reqs = info["requires"]
            ok = all(avail.get(r, False) for r in reqs)
            missing = [r for r in reqs if not avail.get(r, False)]
            if ok:
                print(f"  {name}: [READY]")
            else:
                print(f"  {name}: [NOT READY] missing: {', '.join(missing)}")
        return

    # Determine which pipelines to test
    if args.pipeline:
        targets = [args.pipeline]
    else:
        targets = [name for name, info in PIPELINES.items()
                    if all(avail.get(r, False) for r in info["requires"])]

    if not targets:
        print("\nNo pipelines ready to test. Download models first.")
        sys.exit(1)

    print(f"\nTesting: {', '.join(targets)}")
    print(f"Frames: {args.frames}, Steps: {args.steps}")
    print(f"Output: {OUTPUT_DIR}")

    results = {}
    source_video = None

    for name in targets:
        try:
            if name == "retake":
                # Retake needs a source video from a previous test
                if source_video is None:
                    source_video = str(OUTPUT_DIR / "test_ti2vid.mp4")
                    if not Path(source_video).exists():
                        source_video = str(OUTPUT_DIR / "test_distilled.mp4")
                ok = test_retake(source_video, frames=args.frames, steps=args.steps)
            elif name == "distilled":
                h, w = args.resolution.split("x")
                ok = test_distilled(frames=args.frames, fp8=args.fp8, q8=args.q8, height=int(h), width=int(w))
            elif name == "1stage":
                ok = test_1stage(frames=args.frames, steps=args.steps, fp8=args.fp8, q8=args.q8)
            else:
                ok = PIPELINES[name]["fn"](frames=args.frames, steps=args.steps)
                if ok and name == "ti2vid":
                    source_video = str(OUTPUT_DIR / "test_ti2vid.mp4")
            results[name] = "PASS" if ok else "SKIP"
        except Exception as e:
            print(f"\n  FAIL: {e}")
            results[name] = "FAIL"
            cleanup()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, status in results.items():
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
