"""Verify the three qualified checkpoints and link them into ComfyUI."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, try_to_load_from_cache

FILES = [
    ("abenzerps/Qwen-Image-2.1-Uncensored-GGUF", "6b34e59458d3eb7ba6a6f86a116aed5253dc02c3", "qwen-image-2.1-UC-NVFP4.safetensors", "7cdf87660e84aecf740b5b0f65c848549ba8a74d88ac201976931ca4c780d121", "diffusion_models/qwen-image21-uc-nvfp4.safetensors"),
    ("Comfy-Org/Qwen-Image-2.1", "cb504a4090723e43f17ad01cec0359490e2de613", "text_encoders/qwen3vl_8b_w4a8.safetensors", "7754425e55e7bea2bfde4dde59a4cc236cb44e5ee9c215ea66ef8d47012824eb", "text_encoders/qwen3vl_8b_w4a8.safetensors"),
    ("Comfy-Org/Qwen-Image-2.1", "cb504a4090723e43f17ad01cec0359490e2de613", "vae/qwen_image_2.1_vae_bf16.safetensors", "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9", "vae/qwen_image_2.1_vae_bf16.safetensors"),
]


def prepare(download=False, link=True, release_file_cache=False):
    rows = []
    for repo, revision, filename, expected, target in FILES:
        cached = try_to_load_from_cache(repo, filename, revision=revision)
        if not isinstance(cached, str) or not Path(cached).is_file():
            if not download:
                raise RuntimeError("Qualified Qwen Image 2.1 checkpoint missing: " + filename)
            cached = hf_hub_download(repo, filename, revision=revision)
        path = Path(cached)
        with path.open("rb") as file:
            actual = hashlib.file_digest(file, "sha256").hexdigest()
            if actual != expected:
                raise RuntimeError("Qwen Image 2.1 checkpoint SHA-256 mismatch: " + filename)
            if release_file_cache:
                # Cold startup has no image inference yet. SHA verification
                # reads ~10 GiB and would refill the clean cache immediately
                # before CUDA context creation beside the resident QAD engine.
                os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        if link:
            destination = Path("/opt/ComfyUI/models") / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.unlink(missing_ok=True)
            destination.symlink_to(path)
        rows.append({"file": filename, "sha256": actual, "bytes": path.stat().st_size})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--release-file-cache", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(download=args.download, link=not args.check,
                             release_file_cache=args.release_file_cache)), flush=True)
