""" 
/home/edp1096/workspace/scripts/quantize_fp8.py
From
    Huihui/Qwen3.5-35B-A3B-heretic & Qwen/Qwen3.5-35B-A3B
To
    Qwen3.5-35B-A3B-FP8-heretic
"""

import os, json, re, shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from pathlib import Path
from huggingface_hub import snapshot_download

ORIGINAL_MODEL = os.environ["ORIGINAL_MODEL"]
ABLITERATED_MODEL = os.environ["ABLITERATED_MODEL"]
FP8_MODEL = os.environ["FP8_MODEL"]
OUTPUT_NAME = os.environ["OUTPUT_NAME"]
OUTPUT_MODEL = f"/root/output/{OUTPUT_NAME}"

BLOCK_SIZE = 128


def build_tensor_index(model_path):
    """Build tensor_name -> shard_path mapping."""
    index = {}
    for sf in sorted(model_path.glob("*.safetensors")):
        with safe_open(str(sf), framework="pt") as f:
            for k in f.keys():
                index[k] = sf
    return index


def load_tensor(index, name):
    """Load a single tensor from indexed model."""
    with safe_open(str(index[name]), framework="pt") as f:
        return f.get_tensor(name)


def dequant_blockwise(fp8_tensor, scale_inv, block_size=128):
    """Dequantize FP8 to float32 using block-wise scales."""
    rows, cols = fp8_tensor.shape
    result = fp8_tensor.float()
    scale_expanded = scale_inv.repeat_interleave(block_size, dim=0)[:rows]
    scale_expanded = scale_expanded.repeat_interleave(block_size, dim=1)[:, :cols]
    return result * scale_expanded


def quantize_blockwise(tensor, block_size=128):
    """Quantize float to FP8 with block-wise scaling."""
    rows, cols = tensor.shape
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    pad_rows = (block_size - rows % block_size) % block_size
    pad_cols = (block_size - cols % block_size) % block_size
    if pad_rows > 0 or pad_cols > 0:
        padded = torch.nn.functional.pad(tensor.float(), (0, pad_cols, 0, pad_rows))
    else:
        padded = tensor.float()

    pr, pc = padded.shape
    n_br = pr // block_size
    n_bc = pc // block_size

    blocks = padded.reshape(n_br, block_size, n_bc, block_size).permute(0, 2, 1, 3)
    block_max = blocks.abs().reshape(n_br, n_bc, -1).amax(dim=-1)
    scales = (block_max / fp8_max).clamp(min=1e-12)

    quantized = (blocks / scales[:, :, None, None]).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    quantized = quantized.permute(0, 2, 1, 3).reshape(pr, pc)[:rows, :cols]

    scale_inv = scales.to(torch.float32)
    return quantized, scale_inv


# Step 1: Download all models
print(f"Downloading {ORIGINAL_MODEL}...")
orig_path = Path(snapshot_download(ORIGINAL_MODEL))

print(f"Downloading {ABLITERATED_MODEL}...")
ablit_path = Path(snapshot_download(ABLITERATED_MODEL))

print(f"Downloading {FP8_MODEL}...")
fp8_path = Path(snapshot_download(FP8_MODEL))

# Step 2: Index BF16 models
print("\nBuilding indices...")
orig_index = build_tensor_index(orig_path)
ablit_index = build_tensor_index(ablit_path)

# Step 3: Pre-scan which BF16 tensors actually changed
print("\nScanning for diffs...")
changed_keys = set()
for key in ablit_index:
    if key not in orig_index:
        continue
    orig_t = load_tensor(orig_index, key)
    ablit_t = load_tensor(ablit_index, key)
    if not torch.equal(orig_t, ablit_t):
        changed_keys.add(key)
        print(f"  Changed: {key}")
    del orig_t, ablit_t

print(f"\n{len(changed_keys)} tensors changed by abliteration")

if len(changed_keys) == 0:
    print("ERROR: No diffs found, models might be identical")
    exit(1)

# Step 4: Cache packed expert diffs (to avoid reloading for each of 256 experts)
expert_diff_cache = {}

def get_expert_diff(packed_key):
    """Load and cache packed expert diff."""
    if packed_key not in expert_diff_cache:
        orig_t = load_tensor(orig_index, packed_key)
        ablit_t = load_tensor(ablit_index, packed_key)
        expert_diff_cache[packed_key] = (ablit_t - orig_t).float()
        del orig_t, ablit_t
    return expert_diff_cache[packed_key]


# Step 5: Copy FP8 non-safetensors files
os.makedirs(OUTPUT_MODEL, exist_ok=True)
for f in fp8_path.iterdir():
    if f.suffix != ".safetensors" and f.is_file():
        shutil.copy2(f, OUTPUT_MODEL)
        print(f"Copied {f.name}")

# Step 6: Process each FP8 shard
fp8_shards = sorted(fp8_path.glob("*.safetensors"))
print(f"\nProcessing {len(fp8_shards)} FP8 shards...")

total_modified = 0
total_unchanged = 0

for shard in fp8_shards:
    print(f"\n--- {shard.name} ---")
    expert_diff_cache.clear()

    # Load all tensors from shard
    tensors = {}
    with safe_open(str(shard), framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    new_tensors = {}

    for key in list(tensors.keys()):
        if key.endswith("_scale_inv"):
            continue

        tensor = tensors[key]
        scale_key = key + "_scale_inv"
        has_scale = scale_key in tensors

        # Check if this is an individual expert weight from FP8
        expert_match = re.match(
            r"(.+\.mlp\.experts\.)(\d+)\.(gate_proj|up_proj|down_proj)\.weight",
            key
        )

        if expert_match and has_scale:
            prefix = expert_match.group(1)
            expert_id = int(expert_match.group(2))
            proj_type = expert_match.group(3)

            # Map to packed BF16 key
            if proj_type in ("gate_proj", "up_proj"):
                packed_key = prefix + "gate_up_proj"
            else:
                packed_key = prefix + "down_proj"

            if packed_key in changed_keys:
                diff_packed = get_expert_diff(packed_key)

                if proj_type == "gate_proj":
                    diff = diff_packed[expert_id, :diff_packed.shape[1] // 2, :]
                elif proj_type == "up_proj":
                    diff = diff_packed[expert_id, diff_packed.shape[1] // 2:, :]
                else:
                    diff = diff_packed[expert_id]

                # Dequant -> apply diff -> requant
                scale_inv = tensors[scale_key]
                dequant = dequant_blockwise(tensor, scale_inv, BLOCK_SIZE)
                modified = dequant + diff
                new_fp8, new_scale = quantize_blockwise(modified, BLOCK_SIZE)
                new_tensors[key] = new_fp8
                new_tensors[scale_key] = new_scale
                total_modified += 1
                continue

            # No change for this expert
            new_tensors[key] = tensor
            if has_scale:
                new_tensors[scale_key] = tensors[scale_key]
            total_unchanged += 1
            continue

        # Non-expert quantized weight
        if has_scale and key in changed_keys:
            orig_t = load_tensor(orig_index, key).float()
            ablit_t = load_tensor(ablit_index, key).float()
            diff = ablit_t - orig_t
            del orig_t, ablit_t

            scale_inv = tensors[scale_key]
            dequant = dequant_blockwise(tensor, scale_inv, BLOCK_SIZE)
            modified = dequant + diff
            new_fp8, new_scale = quantize_blockwise(modified, BLOCK_SIZE)
            new_tensors[key] = new_fp8
            new_tensors[scale_key] = new_scale
            total_modified += 1
            print(f"  Modified: {key}")
            continue

        # Non-quantized weight with diff
        if not has_scale and key in changed_keys:
            ablit_t = load_tensor(ablit_index, key)
            new_tensors[key] = ablit_t
            total_modified += 1
            print(f"  Replaced: {key}")
            continue

        # Unchanged - copy as-is
        new_tensors[key] = tensor
        if has_scale:
            new_tensors[scale_key] = tensors[scale_key]
        total_unchanged += 1

    save_file(new_tensors, f"{OUTPUT_MODEL}/{shard.name}")
    del tensors, new_tensors
    print(f"  Saved ({total_modified} modified so far)")

print(f"\nTotal modified: {total_modified}, Unchanged: {total_unchanged}")
print(f"Done: {OUTPUT_MODEL}")
