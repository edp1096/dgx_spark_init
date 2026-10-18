"""Transfer the audited GLM Huihui delta into a separate FP4-Local checkpoint.

Only 44 FP8 projection matrices and their scales may change. Routed NVFP4
experts, their compressed scales, MTP, vision, and all other bytes stay intact.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from weights_core.safetensors_io import Checkpoint, read_header
from download import REPO, REVISION, atomic, sha256, verify
from fetch_deltas import audited
from numeric import DTYPES, mapped, q8_decode, summarize, transform


ROWS = 128
METHOD = ("DQ(NVIDIA FP4-Local QAD) + DQ(Huihui Q8_0) - DQ(Unsloth Q8_0); "
          "BF16 target rounding; MXFP8/FP8 block requantization of 44 projections")
POLICY = ("Per quantization block, choose minimum squared reconstruction error among "
          "unchanged source, original-scale rounding, and fresh ModelOpt-scale rounding; "
          "preserve source bytes on ties")


class Projections(Checkpoint):
    def __init__(self, directory, rows):
        self.root = directory.resolve(strict=True)
        self.tensors = {}
        index = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        wanted = {mapped(row)[0] + suffix for row in rows for suffix in (".weight", ".weight_scale")}
        if not wanted <= index.keys():
            raise ValueError("Source checkpoint is missing audited projection tensors")
        shards = {index[key] for key in wanted}
        for shard in sorted(shards):
            if Path(shard).name != shard:
                raise ValueError("Unexpected projection shard path")
            header, start = read_header(self.root / shard)
            for name in wanted:
                if index[name] == shard:
                    self.tensors[name] = {**header[name], "shard": shard, "start": start}
        if self.tensors.keys() != wanted:
            raise ValueError("Incomplete projection subset")


def _read(cp, descriptor, first, rows, scale=False):
    shape, dtype = descriptor["shape"], descriptor["dtype"]
    if dtype not in DTYPES:
        raise ValueError("Unsupported projection dtype")
    if scale and dtype == "F32":
        if first % 128 or rows % 128 or len(shape) != 4:
            raise ValueError("Misaligned FP8 block-scale row range")
        count = rows // 128 * shape[2]
        offset = first // 128 * shape[2] * 4
        result_shape = [rows // 128, 1, shape[2], 1]
        length = count * 4
    else:
        if len(shape) != 2 or dtype not in ("U8", "F8_E4M3"):
            raise ValueError("Unexpected row-major projection encoding")
        offset, length = first * shape[1], rows * shape[1]
        result_shape = [rows, shape[1]]
    with (cp.root / descriptor["shard"]).open("rb") as stream:
        stream.seek(descriptor["start"] + descriptor["data_offsets"][0] + offset)
        data = stream.read(length)
    if len(data) != length:
        raise ValueError("Truncated projection payload")
    return torch.frombuffer(bytearray(data), dtype=DTYPES[dtype]).reshape(result_shape)


def _bytes(value):
    return value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _write(root, descriptor, first, rows, value, scale=False):
    shape, dtype = descriptor["shape"], descriptor["dtype"]
    if value.dtype != DTYPES[dtype]:
        raise ValueError("Quantizer changed stored dtype")
    if scale and dtype == "F32":
        if list(value.shape) != [rows // 128, 1, shape[2], 1]:
            raise ValueError("Quantizer changed FP8 block-scale geometry")
        offset = first // 128 * shape[2] * 4
    else:
        if list(value.shape) != [rows, shape[1]]:
            raise ValueError("Quantizer changed row-major geometry")
        offset = first * shape[1]
    data = _bytes(value)
    if offset + len(data) > descriptor["data_offsets"][1] - descriptor["data_offsets"][0]:
        raise ValueError("Patch exceeds its tensor payload")
    with (root / descriptor["shard"]).open("r+b") as stream:
        stream.seek(descriptor["start"] + descriptor["data_offsets"][0] + offset)
        stream.write(data)
    return data


def process(root, cp, row, destination=None):
    prefix, shape, expected_kind = mapped(row)
    weight, scale = [cp.tensors[prefix + suffix] for suffix in (".weight", ".weight_scale")]
    if weight["shape"] != shape or weight["dtype"] != "F8_E4M3":
        raise ValueError("Checkpoint/GGUF matrix geometry mismatch")
    expected_scale = [shape[0], shape[1] // 32] if expected_kind == "mxfp8" else [shape[0] // 128, 1, shape[1] // 128, 1]
    if scale["shape"] != expected_scale or scale["dtype"] != ("U8" if expected_kind == "mxfp8" else "F32"):
        raise ValueError("Checkpoint projection has an unexpected scale format")
    files = [root / "ranges" / side / (row["name"] + ".bin") for side in ("original", "huihui")]
    for file, side in zip(files, ("original", "huihui")):
        if file.stat().st_size != row["bytes"] or sha256(file) != row[side + "_sha256"]:
            raise ValueError("Audited GGUF source changed")
    totals = {}
    hashes = [hashlib.sha256(), hashlib.sha256()]
    with files[0].open("rb") as original, files[1].open("rb") as donor:
        for first in range(0, shape[0], ROWS):
            rows = min(ROWS, shape[0] - first)
            length = rows * shape[1] // 32 * 34
            qa = q8_decode(original.read(length), rows, shape[1])
            qb = q8_decode(donor.read(length), rows, shape[1])
            w = _read(cp, weight, first, rows)
            s = _read(cp, scale, first, rows, scale=True)
            qw, qs, kind, stats = transform(qa, qb, w, s)
            if kind != expected_kind:
                raise ValueError("Mixed-precision mapping mismatch")
            for key, value in stats.items():
                totals[key] = totals.get(key, 0) + value
            if destination is not None:
                bw = _write(destination, weight, first, rows, qw)
                bs = _write(destination, scale, first, rows, qs, scale=True)
            else:
                bw, bs = _bytes(qw), _bytes(qs)
            hashes[0].update(bw)
            hashes[1].update(bs)
        if original.read(1) or donor.read(1):
            raise ValueError("Unconsumed GGUF tensor bytes")
    stats = summarize(totals)
    if stats["source_alignment_relative_l2"] > 0.20:
        raise ValueError("FP4-Local and GGUF projection bases do not align: " + row["name"])
    if stats["delta_norm2"] <= 0 or stats["effect_norm2"] <= 0:
        raise ValueError("Ablation delta vanished during requantization")
    if stats["target_relative_error"] > 0.08 or stats["error_relative_to_delta"] > 1.000001:
        raise ValueError("Requantization error exceeded the declared bound")
    if stats["delta_effective_cosine"] < 0.5 or stats["effect_relative_to_delta"] < 0.1:
        raise ValueError("Insufficient preservation of the audited delta direction")
    return {
        "gguf_name": row["name"], "prefix": prefix, "shape": shape,
        "stored_precision": kind, "weight_sha256": hashes[0].hexdigest(),
        "scale_sha256": hashes[1].hexdigest(), **stats,
    }


def regions(path, intervals):
    full, outside = hashlib.sha256(), hashlib.sha256()
    cursor, length = 0, path.stat().st_size
    with path.open("rb") as stream:
        for lo, hi in sorted(intervals) + [(length, length)]:
            if not cursor <= lo <= hi <= length:
                raise ValueError("Invalid or overlapping allowed mutation ranges")
            for amount, include in ((lo - cursor, True), (hi - lo, False)):
                while amount:
                    data = stream.read(min(amount, 8 << 20))
                    if not data:
                        raise ValueError("Truncated checkpoint during byte validation")
                    full.update(data)
                    if include:
                        outside.update(data)
                    amount -= len(data)
            cursor = hi
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return {"full": full.hexdigest(), "outside": outside.hexdigest()}


def card():
    return """---
license: other
license_name: nvidia-open-model-agreement
base_model:
- nvidia/GLM-5.3-Flash-FP4-Local
- huihui-ai/GLM-5.3-Flash-abliterated-GGUF
base_model_relation: finetune
pipeline_tag: image-text-to-text
library_name: Model Optimizer
tags:
- glm5_next
- modelopt
- nvfp4
- mxfp8
- abliterated
---

# Huihui-GLM-5.3-Flash-abliterated-FP4-Local

A locally derived checkpoint; not an official NVIDIA or Huihui release.

The revision-pinned Huihui GGUF ablation delta is transferred into NVIDIA's
FP4-Local QAD checkpoint. Only 44 decoder attention-output and shared-expert
down projections, plus their existing MXFP8/FP8 block scales, can change.
Routed NVFP4 experts, losslessly compressed expert scales, MTP, vision, routers,
normalization parameters, activation scales, and tokenizer files are preserved.

The transfer is `DQ(FP4-Local) + DQ(Huihui Q8) - DQ(Unsloth Q8)`, followed by
BF16 rounding and requantization in the source tensor's format. Each block
selects the lowest squared reconstruction error among unchanged source,
original-scale rounding, and fresh-scale rounding. No new QAD training or
activation calibration was performed. GGUF residuals and requantization errors
remain; equivalence to a Huihui BF16 model and preservation of NVIDIA's benchmark
scores are not claimed.

See `transfer-manifest.json` for pinned sources, per-projection numerical
measurements, exact changed tensor names, and full checkpoint hashes.
`gguf-audit.json` records the earlier complete 1,412-tensor source comparison.
`README.nvidia.md` preserves the upstream model card and its license information.

The upstream special NVFP4 scale format still requires a compatible B12X vLLM
container. Keep the original two-Spark serving requirements. Numerical and byte
verification is separate from runtime qualification; consult any accompanying
runtime qualification report before treating this candidate as tested in use.
"""


def run(args):
    torch.set_num_threads(2)
    root, base = args.root.resolve(strict=True), args.base.resolve(strict=True)
    report, rows = audited(root)
    manifest = json.loads((root / "metadata/base-api.json").read_text())
    if manifest["id"] != REPO or manifest["sha"] != REVISION:
        raise ValueError("Unexpected NVIDIA source revision")
    receipt = json.loads((root / "delta-download-receipt.json").read_text())
    if receipt["status"] != "verified" or receipt["audit_sha256"] != sha256(root / "metadata/gguf-audit.json"):
        raise ValueError("Delta download is not verified against the full audit")
    config = json.loads((base / "config.json").read_text())
    if (config["architectures"] != ["Glm5NextForConditionalGeneration"]
            or [config["text_config"][k] for k in ("hidden_size", "num_hidden_layers", "n_routed_experts")] != [4096, 45, 288]):
        raise ValueError("Unexpected GLM architecture")
    cp = Projections(base, rows)
    by_name = {r["rfilename"]: r for r in manifest["siblings"]}
    started = time.time()
    numeric_sha = sha256(Path(__file__).with_name("numeric.py"))
    converter_sha = sha256(Path(__file__))

    def progress(phase, **extra):
        atomic(root / ("analysis-progress.json" if args.mode == "analyze" else "build-progress.json"), {
            "phase": phase, "started": started, "time": time.time(), **extra,
        })

    for name in ("config.json", "model.safetensors.index.json", "hf_quant_config.json"):
        verify(base / name, by_name[name])
    if args.mode == "analyze":
        for shard in sorted({t["shard"] for t in cp.tensors.values()}):
            progress("verifying_projection_source", shard=shard)
            verify(base / shard, by_name[shard])
        records = []
        for row in rows:
            progress("analyzing", tensor=row["name"], completed=len(records), total=44)
            record = process(root, cp, row)
            records.append(record)
            print("ANALYZED", row["name"], record["stored_precision"],
                  record["delta_effective_cosine"], flush=True)
        atomic(root / "numeric-dry-run.json", {
            "status": "verified", "repo": REPO, "revision": REVISION,
            "audit_sha256": receipt["audit_sha256"], "numeric_sha256": numeric_sha,
            "converter_sha256": converter_sha, "method": METHOD, "scale_policy": POLICY,
            "tensors": records, "started": started, "finished": time.time(),
        })
        progress("verified", completed=44)
        return

    if args.output is None:
        raise ValueError("Build mode requires an output path")
    output = args.output.resolve()
    if any(output == source or source in output.parents or output in source.parents for source in (base, root)):
        raise ValueError("Output must be independent of all inputs")
    work = output.with_name("." + output.name + ".partial")
    if output.exists() or work.exists():
        raise FileExistsError("Candidate or partial already exists; refusing to overwrite it")
    download = json.loads((root / "base-download-receipt.json").read_text())
    if download["status"] != "verified" or download["revision"] != REVISION:
        raise ValueError("Full NVIDIA checkpoint download is not verified")
    dry = json.loads((root / "numeric-dry-run.json").read_text())
    if (dry["status"] != "verified" or dry["revision"] != REVISION
            or dry["audit_sha256"] != receipt["audit_sha256"]
            or dry["numeric_sha256"] != numeric_sha or dry["converter_sha256"] != converter_sha):
        raise ValueError("A matching full numerical dry run must pass before checkpoint creation")
    dry_records = {r["gguf_name"]: r for r in dry["tensors"]}
    full_cp = Checkpoint(base)
    shards = sorted({t["shard"] for t in full_cp.tensors.values()})
    if set(shards) != {r["rfilename"] for r in manifest["siblings"] if r["rfilename"].endswith(".safetensors")}:
        raise ValueError("Source index does not account for every checkpoint weight file")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < sum((base / s).stat().st_size for s in shards) + 5 * 2**30:
        raise ValueError("Insufficient space for an independent output checkpoint")
    work.mkdir()
    state = {
        "status": "building", "runtime_validated": False, "base_repo": REPO,
        "base_revision": REVISION, "method": METHOD, "scale_policy": POLICY,
        "gguf_sources": report["sources"], "audit_sha256": receipt["audit_sha256"],
        "numeric_sha256": numeric_sha, "converter_sha256": converter_sha,
        "activation_scales": "preserved; no new calibration",
        "new_qad_training": False,
        "quantizer_version": importlib.metadata.version("nvidia-modelopt"),
        "torch_version": torch.__version__, "source_shard_hashes": {},
        "tensors": [], "started": started,
    }

    def save():
        atomic(work / "transfer-manifest.json", state)

    for shard in shards:
        progress("copying", shard=shard, completed=len(state["source_shard_hashes"]), total=len(shards))
        digest = hashlib.sha256()
        (work / shard).parent.mkdir(parents=True, exist_ok=True)
        with (base / shard).open("rb") as source, (work / shard).open("xb") as destination:
            while data := source.read(8 << 20):
                digest.update(data)
                destination.write(data)
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        if digest.hexdigest() != by_name[shard]["lfs"]["sha256"]:
            raise ValueError("Source checkpoint changed after download verification")
        state["source_shard_hashes"][shard] = digest.hexdigest()
        save()
        print("COPIED", shard, flush=True)
    metadata = [r for r in manifest["siblings"] if not r["rfilename"].endswith(".safetensors")]
    for entry in metadata:
        name = entry["rfilename"]
        verify(base / name, entry)
        target = work / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base / name, target)
    allowed = set()
    actual_changed = []
    for row in rows:
        progress("patching", tensor=row["name"], completed=len(state["tensors"]), total=44)
        record = process(root, cp, row, work)
        for key in ("weight_sha256", "scale_sha256"):
            if record[key] != dry_records[row["name"]][key]:
                raise ValueError("Conversion differs from the validated numerical dry run")
        for suffix, key in ((".weight", "weight_sha256"), (".weight_scale", "scale_sha256")):
            name = record["prefix"] + suffix
            allowed.add(name)
            if cp.digest(name) != record[key]:
                actual_changed.append(name)
        state["tensors"].append(record)
        save()
        print("PATCHED", row["name"], flush=True)
    # Compare one shard header at a time: the source has nearly 300,000 tensor
    # descriptors, so constructing a second full Checkpoint wastes substantial RAM.
    for shard in shards:
        source_header, source_start = read_header(base / shard)
        output_header, output_start = read_header(work / shard)
        if source_start != output_start or source_header != output_header:
            raise ValueError("Output checkpoint tensor structure changed")
    state["output_shard_hashes"] = {}
    for shard in shards:
        progress("byte_verification", shard=shard, completed=len(state["output_shard_hashes"]), total=len(shards))
        intervals = [(t["start"] + t["data_offsets"][0], t["start"] + t["data_offsets"][1])
                     for name, t in full_cp.tensors.items() if name in allowed and t["shard"] == shard]
        if intervals:
            source_hashes = regions(base / shard, intervals)
            result_hashes = regions(work / shard, intervals)
            if (source_hashes["full"] != state["source_shard_hashes"][shard]
                    or source_hashes["outside"] != result_hashes["outside"]):
                raise ValueError("Non-target bytes or source bytes changed")
            digest = result_hashes["full"]
        else:
            digest = sha256(work / shard)
            if digest != state["source_shard_hashes"][shard]:
                raise ValueError("Untouched shard changed")
        if os.path.samefile(base / shard, work / shard):
            raise ValueError("Output unexpectedly shares a source inode")
        state["output_shard_hashes"][shard] = digest
        save()
        print("BYTE VERIFIED", shard, flush=True)
    for entry in metadata:
        name = entry["rfilename"]
        if (base / name).read_bytes() != (work / name).read_bytes():
            raise ValueError("Source metadata changed during copying")
    shutil.copyfile(work / "README.md", work / "README.nvidia.md")
    (work / "README.md").write_text(card())
    shutil.copyfile(root / "metadata/gguf-audit.json", work / "gguf-audit.json")
    state.update(
        status="candidate_verified", changed_tensors=sorted(actual_changed),
        allowed_projection_tensors=sorted(allowed),
        preserved_tensor_count=len(full_cp.tensors) - len(actual_changed),
        total_tensor_count=len(full_cp.tensors), finished=time.time(),
    )
    save()
    work.rename(output)
    atomic(root / "conversion-result.json", {
        "status": "candidate_verified", "output": str(output),
        "projection_count": 44, "changed_tensor_count": len(actual_changed),
        "shards": len(shards), "runtime_validated": False,
        "transfer_manifest_sha256": sha256(output / "transfer-manifest.json"),
    })
    progress("candidate_verified", output=str(output))
    print("CANDIDATE VERIFIED", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--mode", choices=("analyze", "build"), required=True)
    parser.add_argument("--output", type=Path)
    run(parser.parse_args())
