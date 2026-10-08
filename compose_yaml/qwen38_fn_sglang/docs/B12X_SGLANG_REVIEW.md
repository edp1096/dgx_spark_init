# SGLang Qwen3.8 B12X comparison — 2026-09-14

Production Talk/Compose remains on the qualified SGLang BF16-KV TP2 lane.
Experimental files are under `../experiments/b12x/`; API 8014, dist port 29980.
No original checkpoint edits or kernel/driver changes. The engine log shows
online NVFP4 conversion of MTP experts even with the draft unquant flag; this
is an observed engine behavior retained in both comparison arms. The B12X
adapter does not introduce additional weight quantization.

Base image: `dgx-sglang-qwen38-fn:sm121-vocab1`
(`sha256:ecb60e9705433d3cf4172567ff084c586c1b89a158288b8251c93d812d0fca50`).
Pinned B12X source: `ce419b52681b7922bb0972d4b58b590a3fd005b2` (1.3.0).
Torch 2.13.0+cu130, CuTe DSL 4.6.2, FlashInfer 0.6.17 retained.
Do not treat a public image tag as a stable implementation: adapters and APIs
changed substantially since the May SGLang+B12X image.

## Reuse assessment

| Component | Existing implementation | Model-specific integration |
|---|---|---|
| NVFP4 MoE | Public SGLang+B12X image and PR #29190 | Reuse SGLang CUTLASS-packed weights, block scales, routing and static activation scales; adapt to B12X 1.3 plan/prepare/bind/run API |
| NVFP4 dense | Public image has a dense adapter | This checkpoint excludes attention, linear attention, router, shared experts, PLE, MTP and head from NVFP4; no broad dense NVFP4 replacement target |
| QSA | B12X QSA/paged/varlen operators exist | Existing SGLang selection/gather can be preserved to compare the packed BF16 varlen attention step first; replacing storage/selection with B12X QSA is a larger change |
| GDN | B12X gdn_decode/prefill exist | TP2 geometry (8 key heads, 24 value heads, D128, FP32 state) matches caps, but B12X fuses gating/norm and transactional state. SGLang's MTP intermediate-state/rollback contract must be adapted explicitly |

The DeepSeek SGLang Spark image's compressed MLA adapter is not a Qwen QSA
adapter. Generic SGLang+B12X+Spark TP2 support already exists; its existence
must not be confused with every model/operator combination being integrated.

## MoE adapter

Changes only `ModelOptNvFp4FusedMoEMethod.apply` under the opt-in
`SGLANG_B12X_PROBE_MOE=1`; all other SGLang processing is preserved. W4A4 NVFP4
activation precision is retained. Precombined SGLang alphas are converted to
B12X's raw-global-scale convention without changing the original tensors.
Execution scratch is now private to each prepared layer and geometry, allocated
outside first CUDA-graph capture. The initial geometry-global sharing is not
retained after an intermittent full-model output failure.

Real layer-0 TP-rank-0 checkpoint experts are used for the kernel comparison:
512 experts, hidden 2560, intermediate 320, top-k 10, up/gate layout and
original static input/weight scales. Both kernels receive identical synthetic
activations and routing. This is not a model-quality test.

The first eager-only measurement was insufficient: GPU-event timings include
host enqueue gaps. CUDA-graph replay measurement showed about 2.2x speedup for
one row, but no gain for 4/16/128/1024 rows. Current MTP target verification
commonly uses multiple rows, so single-row performance alone does not justify
adoption. Output relative L2 difference was approximately 2.5–2.7%; all values
finite, cosine >0.9996. Full-model response tests are required.

## Full-model comparison

Same TP2, original weights, BF16 KV, SSD PLE, MTP NEXTN 3/1/4, chunk 1024,
max running 1, context 262144, and guarded memory limits for both modes.
Exact prompts 512/4096/16384 tokens, warm request before each length, three
measured repeats with prefix cache flushed before every trial. Client TTFT
and generated-token counts determine PP/TG, not the server's logging interval.
Start-of-answer arithmetic check guards gross output failures; it is not a
broad quality benchmark. MoE comparison completed; see `b12x-sglang-comparison.json`. The private-scratch
arm passed 3 repeats at 512, 10 at 4096, and 3 at 16384 tokens. CUTLASS baseline
had 3 repeats at each size. MoE PP improved only 1.2–1.3% at 4K/16K while TG
fell 1.6–2%; at 512, TG improved 9.4% but TTFT grew 4.2%. This is not a broad
PP improvement and the MoE adapter is not adopted.

Sources:
- https://github.com/local-inference-lab/b12x/tree/ce419b52681b7922bb0972d4b58b590a3fd005b2
- https://hub.docker.com/r/lukealonso/sglang-cuda13-b12x
- https://github.com/sgl-project/sglang/pull/29190
- https://raw.githubusercontent.com/sgl-project/sglang/0d95a9c1ff14773b722009a6c6fd6ad66c7ce395/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx


Initial full-model B12X attempt: 512-token and two 4096-token measured trials
passed, but the third 4096-token request emitted only `!` for all 128 output
tokens. The harness stopped both servers. That trial's apparent speed is
invalid; no adoption or 1M escalation from it. The next attempt uses private
per-layer scratch. This is a suspected ownership issue, not a proven root cause.


## Additional kernel measurements

All measured after both full-model servers stopped, with CUDA graphs, same
input tensors and output precision. Source/results are in the experiment folder.

- QSA's packed selected-KV dot/softmax stage (D256, 12:1 GQA, 2051 selected
  keys): B12X generic varlen was 2.7–6.6x slower for 1/4/16/128 queries.
  Relative L2 ~0.0029. This rejects that narrow replacement, not every possible
  B12X QSA/paged integration; selection and gather were deliberately retained.
- GDN normal decode plus sigmoid gated RMSNorm (TP2 8/24 heads, FP32 state):
  both SGLang and B12X passed B12X's independent output/state oracle tolerances.
  B12X was about 17% slower for one sequence and 10% slower for four. This
  does not qualify or measure MTP rollback, or the research-only GDN prefill.
- BF16/FP32 dense: output projection and FP32 router replacements were slower.
  BF16 LM-head geometry (124160 x 2560 local shard) with one row improved from
  3.625 ms to 2.587 ms (~29%). Four-row benefit was only ~3%; 128 rows regressed.
  Thus the next full-model arm changes only BF16 one-row LM-head multiplication.
  TP gather, sampling, dtype and weights stay in the original SGLang path.
  QKVZ and narrow gate projections also had some favorable small-row cases,
  but no integration/adoption claim is made from synthetic matrices alone.

Dense tests used representative geometry and synthetic weights, not a full
model. MoE used actual checkpoint weights. Neither alone proves a quality or
end-to-end throughput gain.


The BF16 one-row LM-head arm passed all 16 measured requests (3/10/3), with
identical generated text to the CUTLASS baseline at each input length.
TG improved 3.55–3.80%; PP/TTFT were essentially unchanged (512-token PP was
1.9% lower, 4K/16K PP +0.1–0.2%). Detailed values: `b12x-head-comparison.json`.

A head-only image was built without the experimental MoE patch. Its logits
processor AST equals the tested implementation after normalizing the log text,
and its ModelOpt MoE source equals the original image. See
`b12x-head-image-check.json`. The 1M validation passed. TP2 standalone and the embedded Talk recipe now
select the head-only image; the TP1 image remains unchanged.


## Adopted result

Only the BF16 single-row LM-head path is retained. The 1M run processed exactly
1,046,528 tokens, recovered all three values, and emitted the same output IDs
as the original SGLang baseline. TTFT 1101.479 s / PP 950.112 tok/s, versus
1098.874 s / 952.364 tok/s: no prefill gain. Short JSON TG estimate 60.612 tok/s,
but the repeated short-request TG improvement (3.5–3.8%) is the stronger evidence.
No OOM or reboot observed. See `b12x-head-1m-result.json` for guards and final states.

The retained image is `dgx-sglang-qwen38-fn:sm121-b12x-head-v1`; build it with
`Dockerfile.b12x-head`. Standalone Compose/manager and Talk's embedded TP2
recipe use this default. An explicit image override still takes precedence.
B12X source/toolchain are pinned; the source patch fails if the expected
SGLang LM-head structure changes. Original weight files are unchanged.

The scratch-sharing MoE failure and its private-scratch correction remain as
experimental evidence only. MoE, QSA, GDN and wider/synthetic dense candidates
were not promoted. Full alternative paged-QSA and GDN MTP/prefill integrations
were not claimed as qualified by these narrower measurements.


Deployment finished: head/worker image IDs match, canonical worker source files
were synchronized, the embedded Talk recipe passed parity/materialization tests,
and the rebuilt Talk process returned HTTP 200 on `/api/health`. Model services
were left stopped. The final 1M container exit codes 137/143 were produced by
explicit post-test stop; OOMKilled=false and boot IDs did not change.

Validation: `go test ./internal/orchestrator ./internal/config ./internal/server`
and the subsequent embedded-asset recheck passed; the ARM64 Talk build succeeded.
The new default is used at the next TP2 model start. Explicit custom image
overrides are preserved.
