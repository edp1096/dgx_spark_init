"""Fetch pinned ComfyUI adapters into the shared Hugging Face cache."""
from pathlib import Path
from huggingface_hub import hf_hub_download

ADAPTERS = (
    ("outpaint", "b11770ac6a3cf9325dcf81742c12b1c4e257880f", "LyNiaZ53Tudg0J6sT8Xbx_pytorch_lora_weights_comfy_converted.safetensors"),
    ("object-remove", "0e3f58790356bf1319b263fc56b333c294b42ff7", "kDEkt5q7tDLKOpQJIVMPx_pytorch_lora_weights_comfy_converted.safetensors"),
    ("background-remove", "ebfee4be9a3e431b4832712a0da43c815d6bb127", "sQn8ANj2lbtrfeemXQta0_pytorch_lora_weights_comfy_converted.safetensors"),
)

if __name__ == "__main__":
    for kind, revision, filename in ADAPTERS:
        repo = f"fal/flux-2-klein-4B-{kind}-lora"
        model = Path(hf_hub_download(repo, filename, revision=revision))
        target = Path(f"/opt/ComfyUI/models/loras/fal-flux2-klein-4b-{kind}.safetensors")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        elif target.exists():
            raise RuntimeError(f"refusing to replace non-symlink model path: {target}")
        target.symlink_to(model)
        print(f"LoRA ready: {repo}@{revision}", flush=True)
