# Dense prefill kernel selection without resident MoE — 2026-09-13

The first capped-tuning 8192 window improved the two normal fixtures, but a
controlled 54,988-token repetitive input regressed: about 48.44s server TTFT,
versus 38.80s with native4096. Both used 224 seeded slots, zero cached prompt
tokens and an excluded warmup. It is not appropriate to infer universal speed
improvement from the two shorter fixtures.

The capped startup cache contained MXFP8 GEMM profiles only up to M4096.
Large shapes therefore used heuristic fallback. `tools/tune_dense_prefill.py`
calibrates the eight actual K/N geometries at M8192 in separate 32GiB-capped
containers, **without loading any model or MoE weights**. It uses the same
FlashInfer CUTLASS implementation and FP8 scale layout. Gloo averages timings
over the two physical GB10s, as vLLM's normal distributed tuner does.

The head's saved cache is broadcast before tuning. Worker cache files can be
older and lack head-only shapes; using different hit sets would desynchronize
the collective search. Existing serving cache files are read, not overwritten.
The output is a separate candidate cache and per-node timing/error records.

| M8192 GEMM (K, N) | Head heuristic → selected |
|---|---:|
| 4096, 5120 | 23.14 → 3.13 ms |
| 5120, 2304 | 13.55 → 0.98 ms |
| 6144, 25600 | 188.54 → 49.82 ms |
| 15360, 5120 | 94.55 → 20.65 ms |

All eight shapes on both ranks had zero measured output difference. Inputs
remain MXFP8, accumulation/output precision is unchanged, and no model weights
are requantized. These isolated GEMM gains are not full-model throughput claims.
Results: `results/dense-prefill/rank{0,1}.json`.

`dense-prefill-profile.json` contains only the eight M8192 kernel selections and
FlashInfer's environment metadata. `AutoTuner.load_configs` validates GPU and
backend versions, and the library revalidates tactics at dispatch. Incompatible
metadata fails startup when this profile is explicitly enabled. Other shapes
continue using the ordinary runtime cache.

## Loading order matters

The first integration loaded the profile before `kernel_warmup`. A later
`autotune(cache=...)` context cleared the loaded entries and restored the old
160-entry cache. That run's full-model results are **not a treated comparison**.
The helper now loads the profile after all kernel warmups. A regression test
checks this order and hook restoration after errors.

The benchmark-only `ds41_dense_profile_status` RPC checks all eight expected
entries on both ranks and rejects conflicting live-tuned winners. Subsequent
context and throughput qualification requires this check before each request.
With verified late loading, the repetitive-input regression disappeared:
mean38.43s versus native4096's38.80s. Final normal-fixture pp improves52.77% /
47.85% over the original Python4096 configuration. See
[window results](PREFILL_WINDOW_8192.md) for conditions and limitations.

Reproduce using `tools/run_dense_prefill_tune.sh <rank> <relative-cache-path>`
on both stopped model hosts. This is calibration tooling, not an additional
resident model or a per-request optimization process.
