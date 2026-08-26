#!/usr/bin/env python3
"""Compile and validate the experimental SM121 CuTe Sol-Attn backend."""

from __future__ import annotations

import argparse
import json
import time

import torch

from sol_attn import interface
from sol_attn.triton_ref import sol_attn as triton_sol_attn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    return parser.parse_args()


def elapsed_ms(function, iterations: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def main() -> None:
    args = parse_args()
    torch.manual_seed(20260826)
    shape = (1, args.tokens, args.heads, 128)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16).contiguous()
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16).contiguous()
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16).contiguous()

    selected = interface.get_sol_attn_backend()
    if selected != "cute_sm121":
        raise RuntimeError(f"expected cute_sm121, got {selected}")

    def cute_call():
        return interface.sol_attn(q, k, v, tau=args.tau, thresh_type="diag")

    def triton_call():
        return triton_sol_attn(q, k, v, tau=args.tau, thresh_type="diag")

    cute_output = cute_call()
    triton_output = triton_call()
    torch.cuda.synchronize()

    delta = (cute_output.float() - triton_output.float()).flatten()
    reference = triton_output.float().flatten()
    relative_l2 = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference))
    cosine = float(torch.nn.functional.cosine_similarity(cute_output.float().flatten(), reference, dim=0))
    max_abs = float(delta.abs().max())
    finite = bool(torch.isfinite(cute_output).all())

    for _ in range(args.warmup):
        cute_call()
        triton_call()
    cute_ms = elapsed_ms(cute_call, args.iterations)
    triton_ms = elapsed_ms(triton_call, args.iterations)

    result = {
        "capability": torch.cuda.get_device_capability(),
        "backend": selected,
        "shape": shape,
        "finite": finite,
        "relative_l2_vs_triton": relative_l2,
        "cosine_vs_triton": cosine,
        "max_abs_vs_triton": max_abs,
        "cute_ms": cute_ms,
        "triton_ms": triton_ms,
        "speedup_vs_triton": triton_ms / cute_ms,
    }
    print(json.dumps(result, indent=2), flush=True)
    if not finite or relative_l2 > 0.1 or cosine < 0.99:
        raise RuntimeError("SM121 CuTe output failed the Triton parity threshold")


if __name__ == "__main__":
    main()
