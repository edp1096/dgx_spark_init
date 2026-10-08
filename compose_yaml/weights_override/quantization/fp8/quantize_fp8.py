"""
/home/edp1096/workspace/scripts/quantize_fp8.py

Universal FP8 weight patcher
Supports: DeepSeek/Qwen (block-wise _scale_inv), vLLM/AutoFP8 (per-tensor weight_scale),
          per-channel scales, and MoE expert packing.

From
    ORIGINAL_MODEL (BF16) & ABLITERATED_MODEL (BF16)
    + FP8_MODEL (quantized)
To
    OUTPUT (FP8 with abliterated diffs applied)
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


# ============================================================
# Scale detection & dequant/requant for all FP8 formats
# ============================================================

SCALE_SUFFIXES = ["_scale_inv", "_scale", ".weight_scale", "_weight_scale"]


def find_scale_key(key, tensors):
    """Find the matching scale tensor key for a given weight key.
    Returns (scale_key, is_inverse) where is_inverse=True means multiply to dequant,
    False means divide to dequant (not currently used but future-proof)."""
    for suffix in SCALE_SUFFIXES:
        candidate = key + suffix
        if candidate in tensors:
            return candidate, True  # All known formats: multiply to dequant
    return None, False


def detect_scale_type(scale_tensor, weight_shape):
    """Detect quantization granularity from scale tensor shape.
    Returns: 'block', 'per_channel', or 'per_tensor'"""
    if scale_tensor.dim() == 0 or (scale_tensor.dim() == 1 and scale_tensor.numel() == 1):
        return "per_tensor"
    if scale_tensor.dim() == 1 and scale_tensor.shape[0] == weight_shape[0]:
        return "per_channel"
    if scale_tensor.dim() == 2:
        return "block"
    # 4D vLLM ModelOpt format: [out_blk, 1, in_blk, 1] -> treat as block
    if scale_tensor.dim() == 4:
        return "block"
    # Fallback: if 1D but doesn't match out_channels, guess per_tensor
    return "per_tensor"


def infer_block_size(weight_shape, scale_shape):
    """Infer block size from weight and scale dimensions."""
    import math
    row_blocks = scale_shape[0]
    col_blocks = scale_shape[1] if len(scale_shape) > 1 else 1
    bs_row = math.ceil(weight_shape[0] / row_blocks)
    bs_col = math.ceil(weight_shape[1] / col_blocks) if col_blocks > 1 else weight_shape[1]
    # Block sizes should match (typically 128), use the row one
    return bs_row


def dequant_fp8(fp8_tensor, scale, scale_type, block_size=128):
    """Universal FP8 dequantization."""
    result = fp8_tensor.float()

    if scale_type == "per_tensor":
        s = scale.float().squeeze()
        return result * s

    if scale_type == "per_channel":
        # scale shape: [out_channels] -> broadcast over columns
        s = scale.float().view(-1, 1)
        return result * s

    if scale_type == "block":
        # Handle 4D ModelOpt format: [out_blk, 1, in_blk, 1] -> squeeze to 2D
        s = scale.float()
        if s.dim() == 4:
            s = s.squeeze(3).squeeze(1)  # -> [out_blk, in_blk]

        rows, cols = fp8_tensor.shape
        s_expanded = s.repeat_interleave(block_size, dim=0)[:rows]
        s_expanded = s_expanded.repeat_interleave(block_size, dim=1)[:, :cols]
        return result * s_expanded

    raise ValueError(f"Unknown scale_type: {scale_type}")


def quantize_fp8(tensor, scale_type, block_size=128, ref_scale_shape=None):
    """Universal FP8 quantization. Returns (fp8_tensor, new_scale).
    ref_scale_shape: original scale shape to preserve format (e.g. 4D ModelOpt)."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    t = tensor.float()

    if scale_type == "per_tensor":
        amax = t.abs().amax()
        scale = (amax / fp8_max).clamp(min=1e-12)
        quantized = (t / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
        return quantized, scale.to(torch.float32).unsqueeze(0)

    if scale_type == "per_channel":
        # Per output channel
        amax = t.abs().amax(dim=1)  # [out_channels]
        scale = (amax / fp8_max).clamp(min=1e-12)
        quantized = (t / scale.unsqueeze(1)).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
        return quantized, scale.to(torch.float32)

    if scale_type == "block":
        rows, cols = t.shape
        pad_rows = (block_size - rows % block_size) % block_size
        pad_cols = (block_size - cols % block_size) % block_size
        if pad_rows > 0 or pad_cols > 0:
            padded = torch.nn.functional.pad(t, (0, pad_cols, 0, pad_rows))
        else:
            padded = t

        pr, pc = padded.shape
        n_br = pr // block_size
        n_bc = pc // block_size

        blocks = padded.reshape(n_br, block_size, n_bc, block_size).permute(0, 2, 1, 3)
        block_max = blocks.abs().reshape(n_br, n_bc, -1).amax(dim=-1)
        scales = (block_max / fp8_max).clamp(min=1e-12)

        quantized = (blocks / scales[:, :, None, None]).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
        quantized = quantized.permute(0, 2, 1, 3).reshape(pr, pc)[:rows, :cols]

        new_scale = scales.to(torch.float32)
        # Restore 4D ModelOpt format if original was 4D
        if ref_scale_shape is not None and len(ref_scale_shape) == 4:
            new_scale = new_scale.unsqueeze(1).unsqueeze(3)  # -> [out_blk, 1, in_blk, 1]

        return quantized, new_scale

    raise ValueError(f"Unknown scale_type: {scale_type}")


# ============================================================
# Tensor index helpers
# ============================================================

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


# ============================================================
# MoE expert packing helpers
# ============================================================

# Patterns for MoE models where FP8 splits packed BF16 experts
MOE_EXPERT_PATTERN = re.compile(
    r"(.+\.mlp\.experts\.)(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
)

expert_diff_cache = {}


def get_expert_diff(packed_key, orig_index, ablit_index):
    """Load and cache packed expert diff."""
    if packed_key not in expert_diff_cache:
        orig_t = load_tensor(orig_index, packed_key)
        ablit_t = load_tensor(ablit_index, packed_key)
        expert_diff_cache[packed_key] = (ablit_t - orig_t).float()
        del orig_t, ablit_t
    return expert_diff_cache[packed_key]


# ============================================================
# Main
# ============================================================

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

# Step 4: Copy FP8 non-safetensors files
os.makedirs(OUTPUT_MODEL, exist_ok=True)
for f in fp8_path.iterdir():
    if f.suffix != ".safetensors" and f.is_file():
        shutil.copy2(f, OUTPUT_MODEL)
        print(f"Copied {f.name}")

# Step 5: Detect FP8 format from first shard
fp8_shards = sorted(fp8_path.glob("*.safetensors"))
print(f"\nDetecting FP8 format...")

detected_format = None
detected_block_size = 128
with safe_open(str(fp8_shards[0]), framework="pt") as f:
    all_keys = list(f.keys())
    for k in all_keys:
        sk, _ = find_scale_key(k, {kk: None for kk in all_keys})
        if sk is not None:
            weight_t = f.get_tensor(k)
            scale_t = f.get_tensor(sk)
            detected_format = detect_scale_type(scale_t, weight_t.shape)
            if detected_format == "block":
                s_shape = scale_t.squeeze().shape if scale_t.dim() == 4 else scale_t.shape
                detected_block_size = infer_block_size(weight_t.shape, s_shape)
            scale_suffix = sk[len(k):]
            print(f"  Format: {detected_format}, block_size: {detected_block_size}")
            print(f"  Scale suffix: '{scale_suffix}', scale shape: {list(scale_t.shape)}")
            print(f"  Sample weight: {k} shape={list(weight_t.shape)}")
            del weight_t, scale_t
            break

if detected_format is None:
    print("WARNING: No scale tensors found, treating as non-quantized model")

# Step 6: Process each FP8 shard
print(f"\nProcessing {len(fp8_shards)} FP8 shards...")

total_modified = 0
total_unchanged = 0

# Collect all scale keys to skip in main loop
all_scale_suffixes = set(SCALE_SUFFIXES)

for shard in fp8_shards:
    print(f"\n--- {shard.name} ---")
    expert_diff_cache.clear()

    # Load all tensors from shard
    tensors = {}
    with safe_open(str(shard), framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    # Build set of scale keys in this shard for fast lookup
    scale_keys_in_shard = set()
    for k in tensors:
        for suffix in SCALE_SUFFIXES:
            if k.endswith(suffix):
                scale_keys_in_shard.add(k)
                break

    new_tensors = {}

    for key in list(tensors.keys()):
        # Skip scale tensors (handled with their weight)
        if key in scale_keys_in_shard:
            continue

        tensor = tensors[key]
        scale_key, _ = find_scale_key(key, tensors)
        has_scale = scale_key is not None

        if has_scale:
            scale_tensor = tensors[scale_key]
            scale_type = detect_scale_type(scale_tensor, tensor.shape)
            block_size = detected_block_size
            if scale_type == "block" and scale_tensor.dim() <= 2:
                s_shape = scale_tensor.shape
                block_size = infer_block_size(tensor.shape, s_shape)
        else:
            scale_type = None
            block_size = detected_block_size

        # --- MoE expert weights (Qwen/DeepSeek packed format) ---
        expert_match = MOE_EXPERT_PATTERN.match(key)

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
                diff_packed = get_expert_diff(packed_key, orig_index, ablit_index)

                if proj_type == "gate_proj":
                    diff = diff_packed[expert_id, :diff_packed.shape[1] // 2, :]
                elif proj_type == "up_proj":
                    diff = diff_packed[expert_id, diff_packed.shape[1] // 2:, :]
                else:
                    diff = diff_packed[expert_id]

                dequant = dequant_fp8(tensor, scale_tensor, scale_type, block_size)
                modified = dequant + diff
                new_fp8, new_scale = quantize_fp8(
                    modified, scale_type, block_size, ref_scale_shape=scale_tensor.shape
                )
                new_tensors[key] = new_fp8
                new_tensors[scale_key] = new_scale
                total_modified += 1
                continue

            # No change for this expert
            new_tensors[key] = tensor
            if has_scale:
                new_tensors[scale_key] = scale_tensor
            total_unchanged += 1
            continue

        # --- Quantized weight with diff ---
        if has_scale and key in changed_keys:
            orig_t = load_tensor(orig_index, key).float()
            ablit_t = load_tensor(ablit_index, key).float()
            diff = ablit_t - orig_t
            del orig_t, ablit_t

            dequant = dequant_fp8(tensor, scale_tensor, scale_type, block_size)
            modified = dequant + diff
            new_fp8, new_scale = quantize_fp8(
                modified, scale_type, block_size, ref_scale_shape=scale_tensor.shape
            )
            new_tensors[key] = new_fp8
            new_tensors[scale_key] = new_scale
            total_modified += 1
            print(f"  Modified ({scale_type}): {key}")
            continue

        # --- Non-quantized weight with diff ---
        if not has_scale and key in changed_keys:
            ablit_t = load_tensor(ablit_index, key)
            new_tensors[key] = ablit_t
            total_modified += 1
            print(f"  Replaced: {key}")
            continue

        # --- Unchanged - copy as-is ---
        new_tensors[key] = tensor
        if has_scale:
            new_tensors[scale_key] = scale_tensor
        total_unchanged += 1

    save_file(new_tensors, f"{OUTPUT_MODEL}/{shard.name}")
    del tensors, new_tensors
    print(f"  Saved ({total_modified} modified so far)")

print(f"\nTotal modified: {total_modified}, Unchanged: {total_unchanged}")
print(f"Done: {OUTPUT_MODEL}")
