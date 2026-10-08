#!/usr/bin/env python3
"""Create an abliterated quantized candidate from three local checkpoints."""
import argparse
from pathlib import Path
import json
import subprocess
import importlib.util


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'donor', 'base'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--format', choices=['auto','fp8','nvfp4'], default='auto', help='Validate base precision; default: auto')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cache', type=Path, default=Path.home()/'.cache/huggingface')
    parser.add_argument('--source-bf16', type=Path, help='Qwen only: original BF16 safetensors model; expert slices are extracted automatically')
    parser.add_argument('--source-raw', type=Path, help='Qwen only: local original BF16 expert slices')
    parser.add_argument('--helper-image', help='Qwen only: existing offline conversion image')
    gguf_spec = importlib.util.find_spec('gguf')
    gguf_default = Path(gguf_spec.origin).parent if gguf_spec and gguf_spec.origin else Path.home()/'.cache/model-tools/qwen38fn-huihui-venv/lib/python3.12/site-packages/gguf'
    parser.add_argument('--gguf-package', type=Path, default=gguf_default, help='Qwen only: local gguf Python package directory')
    parser.add_argument('--profile', required=True, help='fp8, ornith_15, gemma4_26b, qwen38_huihui, or a local profile JSON path')
    parser.add_argument('--activation-scales', choices=['reject', 'preserve'], default='reject',
                        help='preserve explicitly retains original input scales without fresh calibration')
    parser.add_argument('--max-tensor-mib', type=int, default=128)
    parser.add_argument('--max-relative-error', type=float, default=.3)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    try:
        profile_path = Path(__file__).resolve().parent/'profiles'/f'{args.profile}.json'
        if not profile_path.is_file():
            profile_path = Path(args.profile)
        profile = json.loads(profile_path.read_text())
        if profile.get('pipeline') == 'qwen38_gguf_delta':
            if args.max_tensor_mib != 128 or args.max_relative_error != .3:
                raise ValueError('Qwen adapter uses its fixed, validated chunk sizes and error limits')
            from model_adapters.qwen38_huihui.pipeline import run
            run(args, profile)
            return
        if profile.get('pipeline'):
            raise ValueError('Unsupported profile pipeline')
        if not all((args.original, args.donor, args.base)):
            raise ValueError('This profile requires --original, --donor and --base')
        from weights_core.formats import validate_format
        validate_format(args.base, args.format)
        from weights_core.conversion import convert, prepare
        if args.check_only:
            report, plans = prepare(args.original, args.donor, args.base, args.profile,
                                    args.activation_scales, args.max_tensor_mib*1024*1024)
            print(f'Preflight passed: {len(plans)} changed tensors; no model written')
        else:
            report = convert(args.original, args.donor, args.base, args.profile, args.output,
                             args.activation_scales, args.max_tensor_mib*1024*1024,
                             args.max_relative_error)
            print(f'Candidate saved: {args.output}; {len(report["metrics"])} changed weights; runtime unverified')
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'Conversion refused: {exc}\n')


if __name__ == '__main__':
    main()
