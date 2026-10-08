#!/usr/bin/env python3
"""Run pinned NVIDIA ModelOpt PTQ and verify the Radix-style precision contract."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


MODELOPT_COMMIT = "87c9f8cf83021957d1a1a575c90c9a4eaaf7ef0c"
DEFAULT_NAME = "Huihui-RadixArk-Qwen3.8-27B-abliterated-NVFP4"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/models/input")
    parser.add_argument("--output-root", default="/models/output")
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--modelopt-root", default="/opt/Model-Optimizer")
    parser.add_argument("--recipe", default="/workspace/qwen38_radix_nvfp4.yaml")
    parser.add_argument("--calib-size", type=int, default=1024)
    parser.add_argument("--calib-seq", type=int, default=512)
    return parser.parse_args()


def validate_paths(args):
    source = Path(args.input).resolve()
    output_root = Path(args.output_root).resolve()
    output = output_root / args.name
    modelopt = Path(args.modelopt_root).resolve()
    recipe = Path(args.recipe).resolve()

    if not (source / "config.json").is_file():
        raise SystemExit(f"Input model is missing config.json: {source}")
    if not (modelopt / ".git").exists():
        raise SystemExit(f"Pinned Model Optimizer checkout is missing: {modelopt}")
    commit = subprocess.check_output(
        ["git", "-C", str(modelopt), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != MODELOPT_COMMIT:
        raise SystemExit(f"ModelOpt commit mismatch: expected {MODELOPT_COMMIT}, got {commit}")
    if not recipe.is_file():
        raise SystemExit(f"Recipe is missing: {recipe}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return source, output, modelopt, recipe


def run_ptq(args, source, output, modelopt, recipe):
    command = [
        sys.executable,
        str(modelopt / "examples/llm_ptq/hf_ptq.py"),
        "--pyt_ckpt_path",
        str(source),
        "--recipe",
        str(recipe),
        "--export_path",
        str(output),
        "--dataset",
        "cnn_dailymail",
        "--calib_size",
        str(args.calib_size),
        "--calib_seq",
        str(args.calib_seq),
        "--batch_size",
        "1",
        "--skip_generate",
        "--trust_remote_code",
    ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})


def verify_output(args, output):
    config_path = output / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"Export did not produce {config_path}")
    config = json.loads(config_path.read_text())
    quant = config.get("quantization_config", {})
    if quant.get("quant_method") != "modelopt":
        raise SystemExit(f"Unexpected quant_method: {quant.get('quant_method')}")
    if quant.get("quant_algo") != "MIXED_PRECISION":
        raise SystemExit(f"Unexpected quant_algo: {quant.get('quant_algo')}")
    kv = quant.get("kv_cache_scheme", {})
    if kv.get("num_bits") != 8 or kv.get("type") != "float":
        raise SystemExit(f"Unexpected KV cache scheme: {kv}")

    groups = quant.get("config_groups", {})
    counts = sorted(len(group.get("targets", [])) for group in groups.values())
    if counts != [193, 208]:
        raise SystemExit(f"Unexpected quantized target counts: {counts}")

    manifest = {
        "name": output.name,
        "source": str(Path("/models/input")),
        "modelopt_commit": MODELOPT_COMMIT,
        "calibration": {
            "dataset": "abisee/cnn_dailymail",
            "samples": args.calib_size,
            "sequence_length": args.calib_seq,
            "batch_size": 1,
        },
        "precision_contract": {
            "attention_and_gdn_fp8_targets": 208,
            "mlp_and_lm_head_nvfp4_targets": 193,
            "kv_cache": "FP8",
            "vision_mtp_and_recurrent_state": "BF16",
        },
    }
    (output / "conversion-manifest.local.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def main():
    args = parse_args()
    source, output, modelopt, recipe = validate_paths(args)
    run_ptq(args, source, output, modelopt, recipe)
    verify_output(args, output)


if __name__ == "__main__":
    main()
