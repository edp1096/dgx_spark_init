"""GPU regression test for the QAD radix overflow repair (no model loading)."""
import json
from pathlib import Path
import torch
from sglang.srt.layers.attention.qsa.kernel import qsa_fast_topk

torch.manual_seed(1031)
checks = []
for width in (4096, 32768, 65536, 262144):
    for kind in ('uniform', 'normal', 'constant', 'negative', 'bf16'):
        scores = torch.randn(4, width, device='cuda')
        if kind == 'uniform':
            scores = torch.rand(4, width, device='cuda')
        elif kind == 'constant':
            scores.fill_(0.25)
        elif kind == 'negative':
            scores = -scores.abs()
        elif kind == 'bf16':
            scores = scores.bfloat16().float()
        starts = torch.tensor([0, 17, 31, 8], device='cuda', dtype=torch.int32)
        lengths = torch.tensor([width, width-17, 99, 0], device='cuda', dtype=torch.int32)
        ends = starts + lengths
        def run():
            return qsa_fast_topk(scores, starts, ends, 512)
        got = run()
        torch.cuda.synchronize()
        # Replay must process changed scores, without host branching on data.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replay = run()
        for attempt in range(2):
            if attempt:
                scores.neg_()
                graph.replay()
                got = replay
            for row, (start, length) in enumerate(zip(starts.tolist(), lengths.tolist())):
                count = min(length, 512)
                ids = got[row, :count].long()
                assert bool(((ids >= 0) & (ids < length)).all())
                assert ids.unique().numel() == count
                values = scores[row, start+ids].sort().values
                expected = scores[row, start:start+length].topk(count).values.sort().values
                assert torch.equal(values, expected), (width, kind, row, attempt)
                assert bool((got[row, count:] == -1).all())
        checks.append(dict(width=width, distribution=kind, rows=4, graph_replay=True, passed=True))
result = dict(cases=checks, passed=len(checks), row_checks=len(checks)*8)
Path('/audit/bench/results/tensorfold-audit/overflow-regression.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
