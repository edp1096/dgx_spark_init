#!/usr/bin/env python3
"""One-shot LTX-2.5 Stage-2 Sol-Attn benchmark for DGX Spark."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.types import OffloadMode
from models.ltx25.RTX5090.attention import LTX25Stage2SolAttention
from models.ltx25.RTX5090.exact_adaln import LTX25ExactAdaLN
from models.ltx25.RTX5090.gpu_infer import TimedStage, Timings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--fps", type=float, default=24)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument(
        "--prompt",
        default=(
            "A cinematic tracking shot of a silver robot walking through a rainy "
            "neon-lit Seoul alley at night, natural blue and amber lighting, smooth "
            "deliberate motion, realistic reflections"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("/output/sol.mp4"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--no-exact-adaln", action="store_true")
    return parser.parse_args()


def stage2_tokens(width: int, height: int, frames: int) -> int:
    temporal = (frames - 1) // 8 + 1
    return temporal * (width // 32) * (height // 32)


def build_exact_table(checkpoint: Path, table: Path, tokens: int) -> None:
    if table.is_file():
        return
    table.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "models.ltx25.RTX5090.build_exact_adaln",
            "--checkpoint",
            str(checkpoint),
            "--tokens",
            str(tokens),
            "--output",
            str(table),
        ],
        check=True,
    )


class DenseAttentionControl:
    @contextmanager
    def stage2(self, _enabled: bool):
        yield

    def stats(self) -> dict[str, object]:
        return {"label": "dense"}


def main() -> None:
    args = parse_args()
    model_dir = Path(os.environ.get("LTX_MODEL_DIR", "/models/ltx-2.5"))
    transformer = model_dir / "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"
    text_encoder = model_dir / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    video_vae = model_dir / "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
    audio_vae = model_dir / "vae/ltx-2.5-audio-vae-bf16.safetensors"
    upscaler = model_dir / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    tokens = stage2_tokens(args.width, args.height, args.frames)
    table = (
        model_dir
        / "sol-cache"
        / f"exact-adaln-{args.width}x{args.height}-{args.frames}f-{tokens}t.pt"
    )

    exact_adaln = None
    if not args.no_exact_adaln:
        build_exact_table(transformer, table, tokens)
        exact_adaln = LTX25ExactAdaLN(table, transformer)

    paths = ModelPaths.from_split(
        transformer_path=str(transformer),
        text_encoder_path=str(text_encoder),
        video_vae_path=str(video_vae),
        audio_vae_path=str(audio_vae),
    )
    policy = QuantizationKind.NVFP4_PREQUANT.to_policy(str(transformer))
    load_start = time.perf_counter()
    pipeline = DistilledPipeline(
        model_paths=paths,
        spatial_upsampler_path=str(upscaler),
        loras=[],
        quantization=policy,
        offload_mode=OffloadMode.NONE,
    )
    load_seconds = time.perf_counter() - load_start

    timings = Timings()
    attention = DenseAttentionControl() if args.dense else LTX25Stage2SolAttention()
    instrumented_stage = pipeline.stage if args.dense else pipeline.stage.with_attention(attention)
    pipeline.stage = TimedStage(
        "stage",
        instrumented_stage,
        timings,
        attention,
        exact_adaln,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    generation_seconds = []
    cuda_peaks = []
    for run_index in range(args.repeats):
        pipeline.stage.call_index = 0
        if isinstance(attention, LTX25Stage2SolAttention):
            attention.reset()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        generation_start = time.perf_counter()
        with torch.inference_mode():
            video, audio, actual_frames, tiling = pipeline(
                prompt=args.prompt,
                seed=args.seed,
                height=args.height,
                width=args.width,
                num_frames=args.frames,
                frame_rate=args.fps,
                images=[],
                tiling_config=AUTO_TILING,
            )
            encode_video(
                video=video,
                fps=args.fps,
                audio=audio,
                output_path=str(args.output),
                video_chunks_number=get_video_chunks_number(actual_frames, tiling),
            )
        torch.cuda.synchronize()
        generation_seconds.append(time.perf_counter() - generation_start)
        cuda_peaks.append(torch.cuda.max_memory_allocated() / (1024**3))
        print(
            json.dumps(
                {
                    "run": run_index + 1,
                    "generation_seconds": generation_seconds[-1],
                    "cuda_peak_gib": cuda_peaks[-1],
                }
            ),
            flush=True,
        )

    metrics = {
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "stage2_tokens": tokens,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "cuda_peak_gib": cuda_peaks,
        "timings": dict(timings.values),
        "attention": attention.stats(),
        "exact_adaln": exact_adaln.stats() if exact_adaln is not None else None,
        "output": str(args.output),
    }
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
