"""GPU regression for GEMM -> QAD softmax/top-k, including graph replay.

Run in the QAD image. Optional --fixture accepts the saved router x/w tensors
from a failing request. Requires an SM12x GPU and CUDA PDL support.
"""
import argparse
import json
import torch
import triton
import triton.language as tl
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate


@triton.jit
def delayed_bias_producer(out, size: tl.constexpr):
    # A legal PDL producer can invite dependent launch before completing writes.
    tl.extra.cuda.gdc_launch_dependents()
    zero = tl.inline_asm_elementwise(
        "nanosleep.u32 1000000; mov.u32 $0, 0;", constraints="=r", args=[],
        dtype=tl.int32, is_pure=False, pack=1,
    )
    tl.store(out + tl.arange(0, size), zero.to(tl.float32))


def check(logits, weights, ids):
    assert torch.isfinite(logits).all(), 'GEMM logits are nonfinite'
    assert torch.isfinite(weights).all(), 'Router produced nonfinite weights'
    probabilities = logits.float().softmax(-1)
    expected = probabilities.gather(1, ids.long())
    expected /= expected.sum(-1, keepdim=True)
    torch.testing.assert_close(weights, expected, rtol=2e-5, atol=2e-7)
    selected = logits.gather(1, ids.long()).float()
    torch.testing.assert_close(selected, logits.float().topk(10).values, rtol=0, atol=0)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture')
    args = parser.parse_args()
    torch.manual_seed(38121)
    if args.fixture:
        saved = torch.load(args.fixture, map_location='cpu', weights_only=True)
        source, weight = saved['x'].cuda(), saved['w'].cuda()
    else:
        source = torch.randn(4096, 2560, device='cuda', dtype=torch.bfloat16)
        weight = torch.randn(512, 2560, device='cuda', dtype=torch.bfloat16) * 0.02
    # Deterministic dependency regression: old router reads poisoned bias before
    # waiting; patched router waits, then sees the producer's valid zero values.
    logits = torch.nn.functional.linear(source, weight)
    bias = torch.zeros(512, device='cuda', dtype=torch.float32)
    delayed_bias_producer[(1,)](bias, 512, launch_pdl=True)
    moe_fused_gate(logits, bias, 10, scoring_func='softmax', renormalize=True)
    torch.cuda.synchronize()
    for _ in range(20):
        bias.fill_(float('nan'))
        delayed_bias_producer[(1,)](bias, 512, launch_pdl=True)
        values, ids = moe_fused_gate(logits, bias, 10, scoring_func='softmax', renormalize=True)
        check(logits, values, ids)
    print(json.dumps({'delayed_bias_repetitions': 20, 'passed': True}), flush=True)
    for rows in (1, 2, 16, 137, 4096):
        x = source[:rows]
        for _ in range(30):
            # Old allocation contents must not influence either producer or consumer.
            dirty = torch.empty(rows, 512, device='cuda', dtype=torch.bfloat16)
            dirty.fill_(float('nan'))
            del dirty
            logits = torch.nn.functional.linear(x, weight)
            bias = torch.zeros(512, device='cuda', dtype=torch.float32)
            values, ids = moe_fused_gate(logits, bias, 10, scoring_func='softmax', renormalize=True)
            check(logits, values, ids)
        print(json.dumps({'rows': rows, 'eager_repetitions': 30, 'passed': True}), flush=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits = torch.nn.functional.linear(x, weight)
            bias = torch.zeros(512, device='cuda', dtype=torch.float32)
            values, ids = moe_fused_gate(logits, bias, 10, scoring_func='softmax', renormalize=True)
        for _ in range(10):
            values.fill_(float('nan'))
            graph.replay()
            check(logits, values, ids)
        graph.reset()
        print(json.dumps({'rows': rows, 'graph_replays': 10, 'passed': True}), flush=True)


if __name__ == '__main__':
    main()
