"""One isolated music generation process; checkpoints are mounted locally."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import soundfile as sf
import torch


def ace(request, output):
    # Upstream refreshes model-side Python files. Keep verified weights read-only
    # and provide per-request writable code copies alongside weight symlinks.
    checkpoints = output.parent / "checkpoints"
    for source in Path("/models/ace").rglob("*"):
        if not source.is_file() or ".cache" in source.parts:
            continue
        target = checkpoints / source.relative_to("/models/ace")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        if source.suffix == ".py":
            shutil.copy2(source, target)
        else:
            target.symlink_to(source)
    os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(checkpoints)
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    handler, lm = AceStepHandler(), LLMHandler()
    config = "acestep-v15-xl-base" if request["model"] == "ace-xl" else "acestep-v15-turbo"
    message, ready = handler.initialize_service(
        project_root="/opt/ace", config_path=config, device="cuda",
        use_flash_attention=False, compile_model=False, use_mlx_dit=False)
    if not ready:
        raise RuntimeError(message)
    thinking = request.get("thinking", True)
    if thinking:
        message, ready = lm.initialize("/models/ace", "acestep-5Hz-lm-1.7B", backend="pt", device="cuda")
        if not ready:
            raise RuntimeError(message)
    params = GenerationParams(
        caption=request["prompt"], lyrics="[Instrumental]", instrumental=True,
        duration=request["duration_seconds"], bpm=request.get("bpm"), seed=request["seed"],
        inference_steps=50 if request["model"] == "ace-xl" else 8,
        guidance_scale=7.0 if request["model"] == "ace-xl" else 1.0,
        shift=1.0 if request["model"] == "ace-xl" else 3.0,
        thinking=thinking, use_cot_metas=thinking, use_cot_caption=False,
        use_cot_language=False)
    result = generate_music(handler, lm, params,
        GenerationConfig(batch_size=1, use_random_seed=False, seeds=[request["seed"]], audio_format="wav"),
        save_dir=str(output.parent / "raw"))
    if not result.success or not result.audios:
        raise RuntimeError(result.error or result.status_message)
    source = Path(result.audios[0]["path"])
    output.write_bytes(source.read_bytes())
    return {"backend": "ace-step", "config": config, "thinking": thinking,
            "inference_steps": params.inference_steps, "guidance_scale": params.guidance_scale,
            "source_revision": "ca1e85fe9430179831e6bc6be790c332190a3866",
            "weights_revision": "220c1166efbdd9583eafcb12eb160594bbfcb241" if request["model"] == "ace-xl" else "19671f406d603126926c1b7e2adc169acbcade22"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    parser.add_argument("output")
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    if request["model"] not in ("ace-turbo", "ace-xl"):
        raise ValueError("Unsupported music model")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(request["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    details = ace(request, output)
    data, rate = sf.read(output, always_2d=True)
    if not np.isfinite(data).all() or not len(data) or np.max(np.abs(data)) < 1e-5:
        raise ValueError("Generated audio is empty, nonfinite or silent")
    frames = [float(np.sqrt(np.mean(data[i:i+rate//10] ** 2))) for i in range(0, len(data), rate//10)]
    active = [i for i, value in enumerate(frames) if value > .001]
    run = longest = 0
    for value in frames[active[0]:active[-1]+1] if active else frames:
        run = run + 1 if value <= .001 else 0
        longest = max(longest, run)
    metadata = {**request, **details, "generation_seconds": time.monotonic() - start,
                "duration_actual": len(data) / rate, "sample_rate": rate, "channels": data.shape[1],
                "peak": float(np.max(np.abs(data))), "rms": float(np.sqrt(np.mean(data ** 2))),
                "clipping_fraction": float(np.mean(np.abs(data) >= .9999)),
                "longest_internal_silence_seconds": longest / 10,
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
