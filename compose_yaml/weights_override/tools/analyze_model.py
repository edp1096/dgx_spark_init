"""
Model FP8 format analyzer.

Usage:
  # JSON index only (lightweight, no GPU needed)
  python analyze_model.py index /path/to/model.safetensors.index.json

  # Safetensors file (shows actual dtype, shape, scale values)
  python analyze_model.py safetensors /path/to/model-00001-of-00005.safetensors

  # HuggingFace model (downloads index.json only)
  python analyze_model.py hf meta-llama/Llama-3.1-8B-Instruct-FP8
"""

import sys, json, re
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# JSON index analyzer
# ============================================================

def analyze_index(path):
    data = json.load(open(path))
    wm = data["weight_map"]
    keys = sorted(wm.keys())

    print(f"Total tensors: {len(keys)}")
    print(f"Shards: {len(set(wm.values()))}")
    print()

    # Scale detection
    scale_suffixes = ["_scale_inv", "_scale", ".weight_scale", "_weight_scale",
                      ".weight_packed", ".weight_shape"]
    found_suffixes = Counter()
    scale_keys = []
    for k in keys:
        for suffix in scale_suffixes:
            if k.endswith(suffix):
                found_suffixes[suffix] += 1
                scale_keys.append(k)
                break

    print("=== Scale / Meta tensors ===")
    if found_suffixes:
        for suffix, count in found_suffixes.most_common():
            print(f"  {suffix}: {count} tensors")
    else:
        print("  None found (BF16/FP16 model)")
    print()

    # Expert detection
    expert_pattern = re.compile(r"\.experts\.(\d+)\.")
    expert_keys = [k for k in keys if expert_pattern.search(k)]
    if expert_keys:
        expert_ids = set()
        expert_layers = set()
        for k in expert_keys:
            m = re.search(r"layers\.(\d+)\..*\.experts\.(\d+)\.", k)
            if m:
                expert_layers.add(int(m.group(1)))
                expert_ids.add(int(m.group(2)))

        print("=== MoE Experts ===")
        print(f"  Expert count: {len(expert_ids)} (id: {min(expert_ids)}~{max(expert_ids)})")
        print(f"  MoE layers: {min(expert_layers)}~{max(expert_layers)} ({len(expert_layers)} layers)")

        # Expert naming pattern (show first expert's keys)
        first_expert = [k for k in expert_keys if ".experts.0." in k]
        # Find from first MoE layer
        first_layer = min(expert_layers)
        sample = [k for k in first_expert if f"layers.{first_layer}." in k]
        if sample:
            print(f"  Sample (layer {first_layer}, expert 0):")
            for k in sorted(sample):
                print(f"    {k}")
    else:
        print("=== MoE Experts ===")
        print("  None (Dense model)")
    print()

    # Prefix detection
    prefixes = Counter()
    for k in keys:
        prefix = k.split(".")[0]
        prefixes[prefix] += 1

    print("=== Top-level prefix ===")
    for prefix, count in prefixes.most_common():
        print(f"  {prefix}: {count} tensors")
    print()

    # Layer structure (from layer 0, non-expert)
    layer0_keys = [k for k in keys
                   if re.search(r"layers\.0\.", k)
                   and "expert" not in k
                   and not any(k.endswith(s) for s in scale_suffixes)]
    if layer0_keys:
        print("=== Layer 0 structure (non-expert, non-scale) ===")
        for k in sorted(layer0_keys):
            print(f"  {k}")
    print()

    # FP8 weight <-> scale pairing check
    if found_suffixes:
        print("=== Scale pairing check ===")
        weight_keys = [k for k in keys if k not in scale_keys]
        paired = 0
        unpaired_weights = []
        for wk in weight_keys:
            has_pair = False
            for suffix in ["_scale_inv", "_scale", ".weight_scale", "_weight_scale"]:
                if wk + suffix in wm:
                    has_pair = True
                    break
            if has_pair:
                paired += 1
            else:
                unpaired_weights.append(wk)

        print(f"  Paired (weight+scale): {paired}")
        print(f"  Unpaired (no scale, e.g. embed/norm): {len(unpaired_weights)}")
        if unpaired_weights:
            print(f"  Sample unpaired:")
            for k in unpaired_weights[:10]:
                print(f"    {k}")


# ============================================================
# Safetensors file analyzer
# ============================================================

def analyze_safetensors(path):
    try:
        from safetensors import safe_open
    except ImportError:
        print("ERROR: pip install safetensors")
        sys.exit(1)

    print(f"File: {path}")
    print()

    with safe_open(str(path), framework="pt") as f:
        keys = sorted(f.keys())
        print(f"Total tensors: {len(keys)}")
        print()

        # Group by dtype
        dtype_groups = defaultdict(list)
        scale_tensors = {}

        for k in keys:
            t = f.get_tensor(k)
            dtype_groups[str(t.dtype)].append((k, list(t.shape)))

            # Collect scale tensors for detailed analysis
            if any(s in k for s in ["scale_inv", "weight_scale", "_scale"]):
                scale_tensors[k] = t

        print("=== Dtype distribution ===")
        for dtype, tensors in sorted(dtype_groups.items()):
            print(f"  {dtype}: {len(tensors)} tensors")
            # Show shape examples
            shapes = Counter(str(s) for _, s in tensors)
            for shape, count in shapes.most_common(5):
                sample = next(k for k, s in tensors if str(s) == shape)
                print(f"    {shape} x{count}  e.g. {sample}")
        print()

        # Scale tensor analysis
        if scale_tensors:
            print("=== Scale tensor analysis ===")
            scale_shapes = Counter()
            for k, t in scale_tensors.items():
                scale_shapes[str(list(t.shape))] += 1

            for shape, count in scale_shapes.most_common():
                sample_key = next(k for k, t in scale_tensors.items() if str(list(t.shape)) == shape)
                sample_t = scale_tensors[sample_key]
                print(f"  Shape {shape} x{count}")
                print(f"    dtype: {sample_t.dtype}")
                print(f"    range: [{sample_t.min().item():.6e}, {sample_t.max().item():.6e}]")
                print(f"    mean:  {sample_t.float().mean().item():.6e}")
                print(f"    example: {sample_key}")

                # Infer quantization type
                if sample_t.dim() == 0 or sample_t.numel() == 1:
                    qtype = "per-tensor"
                elif sample_t.dim() == 1:
                    qtype = "per-channel"
                elif sample_t.dim() == 2:
                    qtype = f"block-wise (inferred block_size ≈ first weight_dim / {sample_t.shape[0]})"
                elif sample_t.dim() == 4:
                    qtype = "block-wise (4D ModelOpt format)"
                else:
                    qtype = "unknown"
                print(f"    quantization: {qtype}")
            print()

        # Show first few tensors with details
        print("=== First 20 tensors ===")
        for k in keys[:20]:
            t = f.get_tensor(k)
            print(f"  {k}")
            print(f"    dtype={t.dtype}  shape={list(t.shape)}")


# ============================================================
# HuggingFace model analyzer (index.json only)
# ============================================================

def analyze_hf(model_id):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: pip install huggingface_hub")
        sys.exit(1)

    print(f"Downloading index.json for {model_id}...")
    try:
        path = hf_hub_download(model_id, "model.safetensors.index.json")
        print(f"Downloaded: {path}")
        print()
        analyze_index(path)
    except Exception as e:
        print(f"ERROR: {e}")
        print("Model might not have index.json (single shard?) or requires auth.")
        sys.exit(1)


# ============================================================
# Main
# ============================================================

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1]
    target = sys.argv[2]

    if mode == "index":
        analyze_index(target)
    elif mode == "safetensors":
        analyze_safetensors(target)
    elif mode == "hf":
        analyze_hf(target)
    else:
        print(f"Unknown mode: {mode}")
        print("Use: index, safetensors, or hf")
        sys.exit(1)


if __name__ == "__main__":
    main()
