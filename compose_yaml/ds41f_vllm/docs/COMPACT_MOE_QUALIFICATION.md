# Compact MoE qualification — rejected, 2026-09-13

Stage 2 keeps the serving b12x pin `789bbb3c846565c41f3404af3e0d7c9ce8702f7f`.
The candidate was upstream `323107ff948ca532f1f7c793b4b550c30ba5212b`, after the
previously rejected `081b2359`. Neither candidate image is a serving default.

## Applicability and numerical contract

The restored compact path explicitly requires `intermediate_size % 128 == 64`.
Our TP2 intermediate size is 1152, whose remainder is zero. Selectively importing
that path's dispatch changes would therefore not select it for this model.
Sources: [compact restoration](https://github.com/local-inference-lab/b12x/commit/ac93b1e7bb38)
and [pinned implementation](https://github.com/local-inference-lab/b12x/blob/323107ff948ca532f1f7c793b4b550c30ba5212b/b12x/moe/fused_moe/_impl.py).

Upstream also [removed the deepseek_v41 recipe](https://github.com/local-inference-lab/b12x/commit/15ec4b45f7a0).
The existing FP32 output buffer is rejected: `output must have dtype
torch.bfloat16, got torch.float32`. This is more than a weight-format change.

To quantify the difference, `tools/probe_b12x_default_recipe.py` explicitly
removes the recipe argument and substitutes a BF16 output buffer **only inside
the isolated test process**. It never changes the serving adapter. Comparison
casts outputs to FP32 before checking tolerance, so failure is due to numeric
values rather than a dtype-check mismatch.

## Physical-node checks

Both nodes used kernel 6.17.0-1032, the original checkpoint and packed files,
16 GiB capped standalone containers, fixed random inputs and original target /
draft geometry. Fresh baseline tests passed for M=1/5/6/8/16/512/2048. Repacking
experts 0 and 17 from the original target layer produced identical packed bytes
under the candidate as well.

| Candidate target M=1 | Head | Worker |
|---|---:|---:|
| Execution time | 0.366 ms | 0.366 ms |
| Max absolute difference | 0.001183 | 0.001168 |
| Relative L2 difference | 4.25% | 4.49% |

These are single-layer output differences, **not measured model-quality loss**.
Both nodes then failed at target M=5 with
`CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE`. No valid 5/6-token speed comparison,
draft candidate result or full-model pp/tg measurement exists. The candidate
failed the numerical gate before any serving experiment was warranted.

The useful outcome is exclusion: the new compact dispatch does not fit our
shape, while the broader replacement changes the numerical contract and fails
a common DSpark shape. No projected speedup is assigned to this candidate.

Reproduction: build `Dockerfile.b12x-compact-probe`, then run
`bash tools/run_compact_check.sh <rank> base` followed by `upstream-default`.
The latter is expected to fail; stop full-model service before either command.
Results and error logs are in `results/compact-moe/`; build logs and the initial
FP32-buffer rejection remain in private persistent probe state.
