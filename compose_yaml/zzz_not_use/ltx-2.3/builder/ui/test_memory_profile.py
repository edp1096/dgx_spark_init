"""Memory profiling script for LTX-2 pipeline components.

Loads each component individually and measures memory usage.
Does NOT run inference — just measures loading/unloading memory.

Usage:
    python test_memory_profile.py                    # Profile all components
    python test_memory_profile.py --fp8              # Profile with fp8_cast
    python test_memory_profile.py --fp8 --keep       # Don't unload between components
"""

import argparse
import gc
import time
from pathlib import Path

import torch

from config import MODEL_DIR


def mem_gb() -> tuple[float, float]:
    """Return (allocated, reserved) in GB."""
    return (
        torch.cuda.memory_allocated() / 1024**3,
        torch.cuda.memory_reserved() / 1024**3,
    )


def mem_str() -> str:
    a, r = mem_gb()
    return f"{a:.1f}GB alloc / {r:.1f}GB reserved"


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def section(name: str):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")


def profile_component(name: str, load_fn, unload_after: bool = True):
    """Load a component, measure memory, optionally unload."""
    before_a, _ = mem_gb()
    print(f"\n[LOAD] {name}...")
    t0 = time.time()
    obj = load_fn()
    torch.cuda.synchronize()
    after_a, _ = mem_gb()
    delta = after_a - before_a
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Memory: {mem_str()} (delta: +{delta:.1f}GB)")

    # Check sample param dtype
    if hasattr(obj, 'named_parameters'):
        for pname, param in obj.named_parameters():
            if 'weight' in pname:
                print(f"  Sample: {pname} dtype={param.dtype} device={param.device}")
                break

    if unload_after:
        del obj
        cleanup()
        reclaimed_a, _ = mem_gb()
        print(f"  After unload: {mem_str()} (reclaimed: {after_a - reclaimed_a:.1f}GB)")

    return obj if not unload_after else None


def main():
    global MODEL_DIR

    parser = argparse.ArgumentParser(description="LTX-2 Memory Profiler")
    parser.add_argument("--fp8", action="store_true", help="Use fp8_cast quantization")
    parser.add_argument("--keep", action="store_true", help="Keep components loaded (simulate pipeline)")
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--distilled", action="store_true", help="Use distilled checkpoint")
    args = parser.parse_args()

    MODEL_DIR = Path(args.model_dir)

    print("=" * 50)
    print("  LTX-2 Memory Profiler")
    print("=" * 50)
    print(f"  FP8: {args.fp8}")
    print(f"  Keep loaded: {args.keep}")
    print(f"  Model dir: {MODEL_DIR}")
    print(f"  Baseline: {mem_str()}")

    # Setup
    from ltx_core.quantization import QuantizationPolicy
    from ltx_pipelines.utils import ModelLedger

    if args.distilled:
        ckpt_name = "ltx-2.3-22b-distilled.safetensors"
        fp8_ckpt_name = "ltx-2.3-22b-distilled-fp8.safetensors"
    else:
        ckpt_name = "ltx-2.3-22b-dev.safetensors"
        fp8_ckpt_name = "ltx-2.3-22b-dev-fp8.safetensors"

    quantization = None
    if args.fp8:
        fp8_path = MODEL_DIR / fp8_ckpt_name
        if fp8_path.exists():
            ckpt_name = fp8_ckpt_name
            print(f"  Using FP8 checkpoint: {ckpt_name}")
        else:
            print(f"  FP8 checkpoint not found, will convert at runtime")
        quantization = QuantizationPolicy.fp8_cast()

    print(f"  Checkpoint: {ckpt_name}")

    # ============================================================
    # 1. Gemma Text Encoder
    # ============================================================
    section("1. Gemma Text Encoder")
    gemma_root = str(MODEL_DIR / "gemma-3-12b-it-qat-q4_0-unquantized")

    ledger_gemma = ModelLedger(
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
        checkpoint_path=str(MODEL_DIR / ckpt_name),
        gemma_root_path=gemma_root,
        quantization=quantization,
    )
    print(f"  After ModelLedger init: {mem_str()}")

    text_enc = profile_component(
        "Gemma Text Encoder",
        ledger_gemma.text_encoder,
        unload_after=not args.keep,
    )

    # ============================================================
    # 2. Embeddings Processor
    # ============================================================
    section("2. Embeddings Processor")
    emb_proc = profile_component(
        "Embeddings Processor",
        ledger_gemma.gemma_embeddings_processor,
        unload_after=True,  # Always unload — small and only used once
    )

    # Unload text encoder if kept
    if args.keep and text_enc is not None:
        print("\n[UNLOAD] Gemma Text Encoder (done with text encoding)")
        del text_enc
        cleanup()
        print(f"  After unload: {mem_str()}")

    del ledger_gemma
    cleanup()
    print(f"\n  After Gemma cleanup: {mem_str()}")

    # ============================================================
    # 3. Transformer
    # ============================================================
    section("3. Transformer")
    ledger_main = ModelLedger(
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
        checkpoint_path=str(MODEL_DIR / ckpt_name),
        gemma_root_path=None,
        spatial_upsampler_path=str(MODEL_DIR / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
        quantization=quantization,
    )

    transformer = profile_component(
        "Transformer (X0Model)",
        ledger_main.transformer,
        unload_after=not args.keep,
    )

    # ============================================================
    # 4. Video Encoder
    # ============================================================
    section("4. Video Encoder")
    video_enc = profile_component(
        "Video Encoder",
        ledger_main.video_encoder,
        unload_after=not args.keep,
    )

    # ============================================================
    # 5. Spatial Upsampler
    # ============================================================
    section("5. Spatial Upsampler")
    upsampler = profile_component(
        "Spatial Upsampler",
        ledger_main.spatial_upsampler,
        unload_after=not args.keep,
    )

    # ============================================================
    # 6. Peak: All Stage 2 components together
    # ============================================================
    if args.keep:
        section("6. PEAK — All Stage 2 components loaded")
        print(f"  Transformer + Video Encoder + Upsampler")
        print(f"  Memory: {mem_str()}")
        a, _ = mem_gb()
        print(f"  This is the Stage 2 peak (before activations/latents)")

        # Cleanup everything
        del transformer, video_enc, upsampler
        cleanup()
        print(f"\n  After full cleanup: {mem_str()}")

    # ============================================================
    # 7. Video Decoder
    # ============================================================
    section("7. Video Decoder")
    profile_component("Video Decoder", ledger_main.video_decoder)

    # ============================================================
    # 8. Audio Decoder + Vocoder
    # ============================================================
    section("8. Audio Decoder")
    profile_component("Audio Decoder", ledger_main.audio_decoder)

    section("9. Vocoder")
    profile_component("Vocoder", ledger_main.vocoder)

    # ============================================================
    # Summary
    # ============================================================
    del ledger_main
    cleanup()
    print("\n" + "=" * 50)
    print("  PROFILING COMPLETE")
    print("=" * 50)
    print(f"  Final memory: {mem_str()}")


if __name__ == "__main__":
    main()
