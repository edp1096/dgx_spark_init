# GB10 TP1 validation — 2026-09-18

Host: `spark-f05a` (gx10-2), NVIDIA GB10. Engine: SGLang, TP1.
Checkpoint: `local-inference-lab/Qwen3.8-Flash-Next-NVFP4`, revision
`7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd`.

Image: `dgx-sglang-qwen38-qad:tp1-trial1`.
Image ID: `sha256:9a27ab327ce8238399036d55e29f2107b7f2c3d11ce1eaccb8e8bfa2721d1038`.

All 36 checkpoint files passed indexed-tensor/header and file-size checks.
Both download workers completed with exit 0 and no OOM. The final scalar-heavy
shard has a legitimate 9,375,048-byte JSON header.

## Full-checkpoint result

The server became healthy and passed all three API probes at approximately
23:39 KST (14:39 UTC). Configuration: text only, 4,096-token context,
8,192-token KV capacity, one request, no CUDA graphs, no MTP, static fraction
0.80, explicit FlashInfer CUTLASS FP4 and MoE runners.

| Probe | Observed result | End-to-end seconds |
| --- | --- | ---: |
| Plain text, thinking disabled | `25` for 17 + 8; normal stop | 2.043 |
| Thinking enabled, low effort | Final `17`; separate `reasoning_content`, 43 reasoning tokens | 5.441 |
| Tool call | `get_weather` with `{"city":"Seoul"}`; `tool_calls` finish reason | 3.011 |

These are three short functional probes, not a throughput benchmark or a model
quality evaluation. No weather tool was executed.

- SGLang reported weight loading in 553.95 seconds; healthy after 632.8 seconds
  from the validation runner's start, including initialization/warmup.
- PLE readiness: 320,001,536 rows × 160 dimensions, all 128 shards loaded.
- Missing-parameter warnings: **0**. The loader intentionally skipped 470 vision
  tensors in text-only mode.
- SGLang reported 34.82 GiB available after weight loading and 29.46 GiB after
  memory-pool initialization. These are unified-memory runtime readings, not
  a standalone VRAM size measurement.
- Minimum host `MemAvailable` sampled every five seconds: **26.61 GiB**.
- The validation runner stopped the trial after success. Existing Talk services
  and production TP1/TP2 recipes were not replaced.

Raw local evidence: `/tmp/qwen-qad-validation-5/{server.log,api-results.json,api-probes.log,memory.json}`
and `/tmp/qwen-qad-validation-5-run.log`.

## Component evidence

- Packed NVFP4 PLE: exact BF16 comparison on GPU, pinned CPU and file-backed
  memory, including file-page eviction and CUDA Graph replay.
- Actual PLE checkpoint sample: exact BF16 agreement with NVIDIA ModelOpt.
  Another 4,096 random rows × five global scales agreed exactly.
- Missing/duplicate PLE pieces rejected; TP2 rejected by this trial path.
- Real SGLang file-based mixed-config loader and fused b/a scale loader passed.
- MXFP8 2560→96: synthetic relative L2 0.027956; actual checkpoint layer 0 b/a
  relative L2 0.025486 versus FP32 reference matmul. The runtime includes MXFP8
  activation quantization; these figures are not model-quality scores.

## Stage 1 remaining work (subsequently addressed below)

- Vision quantization propagation and numerical/runtime checks.
- MTP W4A16 expert execution and acceptance/quality checks.
- B12X integration and performance comparison. **No B12X optimization was used
  for the successful full-model test.**
- Longer-context/resource tests and broader model-quality evaluation.
- Talk TP1 migration, including the explicit thinking-off template toggle.
  The pinned template rejects legacy top-level `reasoning_effort: none`.

## Stage 2 component results — 2026-09-19 KST

The isolated `tp1-trial2` image adds independently written SGLang adapters to
B12X commit `ce419b52681b7922bb0972d4b58b590a3fd005b2` (Apache-2.0).
These results do not by themselves qualify full-model serving:

- All 470 real vision tensors loaded. The vision encoder produced a finite
  `(4, 2560)` output. Its W4A16 FC2 relative L2 against ModelOpt-dequantized
  BF16 matmul was 0.00196–0.00218 for M=1/17/64. K=4304 is padded with zero
  weights/scales/activations to 4320 for the B12X kernel.
- Actual MTP W4A16 experts versus BF16-dequantized PyTorch SiLU reference:
  relative L2 0.00578–0.00686 for M=1/4/17/64. The complete 4,637-tensor MTP
  loaded; dispatch/combine with 512 experts and top-10 routing produced finite
  output. MTP activations remain BF16.
- Actual text NVFP4 experts versus SGLang's native CUTLASS implementation:
  relative L2 0.0173–0.0370. Against a separate BF16-activation reference,
  CUTLASS error was 0.1379–0.1563 and B12X 0.1386–0.1567. Both paths apply
  the checkpoint's NVFP4 activation calibration; different fused kernels are
  not bitwise equivalent. These numbers are arithmetic checks, not model
  accuracy scores.
- Real LM head: B12X BF16 GEMV relative L2 0.00000632 (M=1) and 0.00011881
  (M=4) against BF16 PyTorch matmul.
- MXFP8 attention/shared/vision projections explicitly retain dynamic MXFP8
  activations (`mode='quantized'`). Actual GDN b/a relative L2 remains 0.025486.

The B12X expert adapter must swizzle block scales and describe physical
`[gate, up]` as B12X `w31`; SGLang uses a different naming convention.
Numerical reference tests caught both layout errors before full-model testing.

Raw component logs: `/tmp/qad-stage2-components4.log`,
`/tmp/qad-experts-reference.log`, `/tmp/qad-mtp-full2.log`,
`/tmp/qad-mtp-full3.log`. Full stage-2 serving and graph qualification are
recorded separately when complete.

## Full stage 2 and stage 3 results — 2026-09-19 KST

Stage 2 (4K, eager, full-vocabulary MTP) became healthy in 734.9 seconds.
All eight API probes passed: text, parsed reasoning, tool call, legacy thinking
off, two vision inputs, tool-result round trip, and counting. Average speculative
acceptance length was 3.3; minimum sampled host available memory was 25.35 GiB.
The trial was stopped normally after completion.

For graph capture, the adapter now provides a PyTorch-owned output buffer to
B12X. W4A16 and NVFP4 changed-input/changed-route-weight graph replay matched
eager output exactly in component tests (`/tmp/qad-stage2-graph-components2.log`).

Stage 3 used a 65,536 configured context, 131,072 BF16 KV tokens, two concurrent
requests, `ko64k` draft vocabulary and full target-verify/draft CUDA graphs.
It became healthy after 707.8 seconds; minimum host available memory was
21.28 GiB. All short API probes passed again, with average acceptance length
3.35 before the operational probes. The logs confirm two simultaneous requests
with `cuda graph: True`.

| Additional probe | Result | End-to-end seconds |
| --- | --- | ---: |
| Concurrent integers 1–60 | Exact sequence, normal stop | 5.684 |
| Concurrent integers 101–160 | Exact sequence, normal stop | 6.957 |
| SSE | `16`, proper `[DONE]` | 0.419 |
| Actual 14,463-token document | Both embedded fields recovered correctly | 16.923 |

This verifies real sparse attention beyond 8K, **not** full 64K input or 1M.
It is a functional check, not a broad quality or throughput benchmark.
The inherited TileLang QSA indexer emitted a static race-check warning whose
counterexample uses negative group/context indices; capture and the real
positive-length sparse-attention probes passed. The warning was not suppressed.

The fresh reproducible deployment image is `dgx-sglang-qwen38-qad:sm121-v1`,
ID `sha256:c418b045a5653223d67be29d4d817d9c011fd54957190616df80c9c9897645d0`.
Its complete SGLang Python-source digest matches the graph-tested trial image:
`1ac0b9caecd27748e230b32d36be7aa559c4f9d22f2e11301411b2c61bb2a69f`.
It is built by the explicit `qad-tp1` Docker target; the default target remains
the legacy runtime so TP2 base-image builds are preserved.

Raw evidence: `/tmp/qad-stage2-validation-1/`, `/tmp/qad-stage3-validation-1/`.
The latter includes the immutable image ID/effective command in `runtime.json`.

## Talk integration

TP1 Compose and embedded Talk assets use the pinned QAD snapshot and mixed
quantization, with a separate PLE/compiler cache. TP2 model/image/recipe files
are unchanged. Configuration revision 12 migrates only the old built-in TP1
model identity, preserving custom model IDs, endpoints, contexts, runtime
options and the active TP2 model. New QAD progress uses its actual 36-shard
ETA; the old 206-shard heuristic remains restricted to 206-shard loads.

Config tests and the QAD/FlashNext/TP2/build-asset orchestrator tests passed.
The full orchestrator suite still fails three unrelated tests:
`TestSharedExtraResolvesPerSet`, `TestGLMEmbeddedRecipeMatchesIndependentSources`,
`TestPackagedModelPatchesMatchStandalone`. The same failures were reproduced
from unchanged HEAD in `/tmp/qad-talk-baseline`; logs are
`/tmp/qad-talk-baseline-tests.log` and `/tmp/qad-talk-tests.log`.

The Talk executable also builds successfully at `/tmp/sparktalk-qad-tp1`.
The running Talk process was not restarted by these tests.

## Initial long-context experiment (not a 256K pass)

Using the deployment image, YaRN factor 4, configured context 1,048,576,
BF16 KV, MTP, one request and memory fraction 0.85:

- SGLang clamped the requested KV pool to **308,096 tokens**; its reported
  request input limit was **308,090**. A 1M configuration was not a 1M capacity.
- An actual **65,391-token** prompt recovered both fields correctly in 65.9s.
- During a **261,999-token** request, host available memory fell to 7.68 GiB.
  The validation runner's 8 GiB guard stopped the trial. This request did not
  finish and must not be counted as a successful 256K test.
- The trial omitted production's `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  Variable-length prefill workspaces can fragment the caching allocator; the
  observed growth is consistent with that, but the first run alone does not
  prove the attribution. The allocator setting has been aligned for retries.
- Automatic Mamba sizing reserved 25 persistent slots for one running request.
  The next experiment caps this at eight (the inherited overlap path needs
  five per active request), preserving the FP32 SSM state precision.

Evidence: `/tmp/qad-context-validation-1/`; selected raw results and memory
samples are retained under `results/context1-*` in this directory.

## 512K BF16 KV retry — passed

Image `sm121-v1`, YaRN factor 2, configured/allocated KV 524,288 tokens,
one request, eight Mamba slots, memory fraction 0.90, full CUDA graphs and
`expandable_segments:True`. Weight/state precision was unchanged.

| Actual input tokens | Both fields recovered | End-to-end seconds |
| ---: | :---: | ---: |
| 65,391 | Yes | 64.4 |
| 261,999 | Yes | 299.9 |
| 524,151 | Yes | 745.0 |

Minimum host available memory was 10.93 GiB. The trial completed and stopped
normally. These prompts use repeated benign filler with two named fields at
separated positions. They test real token processing and retrieval, not general
long-document quality or the worst-case PLE working set.

Raw evidence: `/tmp/qad-context512-validation-1/`, with selected results,
runtime metadata and memory samples under `results/context512-*`.

## Canonical B12X storage follow-up

The adapter now makes each module's existing weight/scale parameter reference
B12X's canonical storage, preserving the parameter class, loader attributes,
logical shape and dtype. This removes raw-scale retention beside the swizzled
scales (7.03 GiB across the 48 target layers), plus any superseded MTP repack
source buffers. The checkpoint files remain unchanged.

The new `sm121-v2` component tests passed with the same numerical errors as
before, exact changed-input graph replay, shared-storage assertions, and a
complete 512-expert MTP dispatch/combine forward. Evidence:
`/tmp/qad-v2-components.log`. Full-model memory savings and 1M capacity are
being measured separately; they are not implied by these component results.

## 1M v2 initialization with a 6 GiB guard — incomplete

The v2 image loaded target weights in 573.19 seconds, reporting 74.36 GiB
weight memory, versus approximately 79.8 GiB before canonical-storage reuse.
The observed reduction is about 5.5 GiB; the raw duplicate-scale size alone
is not an observed runtime saving. MTP weight loading reported 4.25 GiB.

With YaRN factor 4, static fraction 0.95 and eight Mamba slots, the runtime
allocated the requested 1,048,576 BF16 KV tokens (target K/V: 12 + 12 GiB;
draft K/V: 1 + 1 GiB). The host-memory guard stopped initialization below
6 GiB available, before graph capture or API qualification. This was a
validation-guard stop, **not an OOM or a successful 1M input test**.

Evidence: `/tmp/qad-context1m-v2-validation-1/`; runtime and memory samples
are retained as `results/context1m-v2-6g-*`. The same configuration is being
retried with the user's explicitly requested 4 GiB available-memory guard.

## 1M v2 with the requested 4 GiB guard — boots, full input incomplete

The identical `sm121-v2` configuration was retried with the host available
memory guard explicitly lowered to 4 GiB. Image ID:
`sha256:dde0b8ed90a10b4280a810f3a19dd6f70a209af10295a7f836886462925bbac4`.

- Healthy after **672.5 seconds**, with full target/draft CUDA graphs.
- Confirmed allocated KV **1,048,576** and input limit **1,048,570**.
- Text, reasoning, tools, vision, legacy thinking-off, MTP, SSE and operational
  retrieval checks passed. Two submitted requests also passed, but this profile
  has `max-running-requests=1`: they are queued, not simultaneous GPU execution.
- A diverse **130,925-token** input recovered both fields in **255.5 seconds**.
- The next **1,048,431-token** input did not finish. The memory guard sampled
  **3.93 GiB** available after 1,300.7 seconds from trial start and stopped
  the trial. The captured log had processed **283,136** tokens and had
  **765,295** pending. Additional chunks ran during Docker's stop grace period.
- Docker reports `OOMKilled=false`. Exit 137 after the stop grace period does
  not establish an OOM. This is a **4 GiB validation-guard stop**.

Thus 6 GiB was too conservative even for startup, but lowering the guard to
4 GiB did not qualify the complete 1M request in this configuration. The
largest completed input remains the earlier **524,151-token** v1 test; that
is a verified result, not a proven exact upper bound of the port. No BF16 KV
precision was reduced. Production defaults remain 64K/two requests rather
than exposing the experimental 1M memory settings.

The v2 adapter passed component numerical/graph checks and the full-model
API checks above. Standalone TP1 Compose and Talk embedded assets now select
`sm121-v2`. Relevant config/orchestrator tests, source synchronization checks
and the final Talk binary build passed. Running Talk was not restarted.

Evidence: `/tmp/qad-context1m-v2-validation-2/`; retained metadata, outcomes,
API results and memory samples: `results/context1m-v2-4g-*`.

## 1M v2 with the requested 3 GiB guard — full input incomplete

The user explicitly authorized one further test down to **3 GiB**, but no
lower. The immutable image, command and recorded environment exactly matched
the preceding 4 GiB run; only the validation guard changed. BF16 KV, MTP,
YaRN factor 4, memory fraction 0.95 and the 1,048,576-token pool were retained.

- Healthy after **652.5 seconds**. Short text, reasoning, tools, vision, MTP,
  SSE and operational probes passed again. The one-running-request profile
  queues the two submitted operational requests rather than executing them
  simultaneously on the GPU.
- Diverse **130,925-token** retrieval passed in **250.0 seconds**.
- During the subsequent **1,048,431-token** request, the memory guard sampled
  **2.98 GiB** available, **1,798.1 seconds** after trial start.
- The log captured at the guard had processed **565,760** tokens, with
  **482,671** pending. More chunks ran during the Docker stop grace period.
- The trial was stopped by the **3 GiB guard**, not by an observed OOM. Docker
  reports `OOMKilled=false`; exit 137 followed the explicit stop grace period.
  No final answer was generated for this 1M request.

Lowering the guard to 3 GiB therefore did **not** establish 1M completion in
this configuration. Processing 565,760 tokens of an unfinished request does
not qualify that length as a completed context test. The largest fully
completed retrieval remains the earlier **524,151-token** v1 test. This is
not proof that 1M is physically impossible or that 512K is the exact limit.
No test below the user's 3 GiB boundary was launched. Production settings
were not changed by this retry.

Evidence: `/tmp/qad-context1m-v2-validation-3/`; retained outcome, runtime,
API results and five-second memory samples: `results/context1m-v2-3g-*`.

## FP8 KV follow-up — full-model functional checks passed

Backported the runtime hunks of SGLang PR36644 (head
`3df8e1e7dbc5807696622afe2929b6c33c185ca3`) into an isolated FP8 image.
The existing SM121 KDA decode remains the attention arithmetic backend;
selected FP8 K/V are correctly descaled into BF16 scratch. No native FP8
attention arithmetic or B12X GDN integration is claimed by this stage.

Seven upstream FP8 tests and three added GB10 TP1 tests passed: D256 /
24 query heads / two KV heads, scaled real cache writes, compact gather,
non-unit descales and changed-input CUDA graph replay. Full-model startup
reported **FP8 target and MTP KV pools, 131,072 tokens**, and became healthy
after 687.8 seconds. Text, reasoning, tools, vision, MTP, concurrency, SSE
and 14K retrieval passed. Minimum available memory: **27.90 GiB**.

Evidence: `/tmp/qad-sgl-fp8-validation-1/`, `/tmp/qad-fp8-upstream-tests.log`,
`/tmp/qad-fp8-sm121-tests.log`; retained results `results/fp8-stage3-*`.
This stage did not yet test a complete 1M request.

## B12X GDN with FP8 KV — full-model functional checks passed

The isolated `gdn-trial` image upgrades B12X to Apache-2.0 commit
`9043b448622764a598969518d413b3fd8b3c0c07` and CuTe DSL to 4.7.0. It adds
independent SGLang adapters for prefill, decode and linear-chain MTP verify.
Persistent and tentative recurrent state share one allocation, with distinct
slots: native SGLang acceptance commits only the selected tentative state.
The native scatter entry check now permits the outer layer stride already
supported by its kernel; inner state entries must still be contiguous.

Four GB10 component checks passed: changed-input CUDA graph replay, decode,
all verify checkpoints with persistent-state rollback, chunk-prefix state,
and actual multilayer native acceptance/track scatter. B12X rounds verify
beta to BF16 while native SGLang keeps FP32 beta: measured state relative L2
was 0.001576 against native SGLang, with a separate tight comparison against
the B12X reference. This is a documented precision-policy difference, not
bitwise equivalence to the prior recurrence.

Upgraded-library regressions also passed for main NVFP4 experts, MTP W4A16
experts, changed-input graphs and all 470 vision tensors. The full model
became healthy in 639.8 seconds; text, reasoning, tools, vision, MTP,
two concurrent requests, SSE, 14K retrieval, cold/cached prefix branching,
and two-turn conversation all passed. Minimum host available memory was
27.75 GiB. Trial stopped normally; production was not restarted.

Evidence: `/tmp/qad-gdn-component-tests3.log`,
`/tmp/qad-gdn-quant-regression.log`, `/tmp/qad-sgl-gdn-validation-1/`;
retained API/runtime/memory results: `results/gdn-stage3-*`.

## Checkpoint-backed PLE — component qualification

`SGLANG_QAD_PLE_IO_URING=1` keeps immutable CPU safetensors mappings instead
of rewriting 26.8 GiB of packed PLE files. Eager batches over 256 lookups use
B12X `DiskRowCache` and actual io_uring submissions. Small decode/verify
batches and CUDA capture use an independent direct-checkpoint UVA kernel.
SGLang computes PLE IDs inside its graphs; CPU I/O must not be captured and
then incorrectly assumed to execute on replay. This is intentionally a
hybrid integration, not the vLLM graph partitioning implementation.

The component check covers file offsets, uneven tail shards, invalid IDs,
duplicate rows, bounded cache growth/reuse, all NVFP4 codes and changing
IDs on graph replay. Both paths exactly matched the independent BF16
reference. `submit_calls > 0` verifies io_uring actually ran. Required image
dependency is `liburing-dev`; runtime needs unlocked memlock and io_uring
permission (the isolated profile uses `seccomp=unconfined`).

Evidence: `/tmp/qad-ple-component-tests2.log`. Full-model / 1M qualification
is tracked separately; component success alone does not establish it.

## Long-input regression found before qualification

The first combined FP8/GDN/PLE 1M-capacity trial passed the short operational
checks but returned 64 exclamation marks (token ID 0) for the 130,925-token
diverse probe. The harness correctly failed it and did not submit the 1M
probe. A fresh diagnostic run with eager finite-output hooks returned
`MAPLE,COMET` in 88.0 seconds. That different outcome is retained, not treated
as proof that the original problem did not exist.

Three focused tests then reproduced **all-NaN output** in the old GDN
adapter: padded physical capacity beyond live `cu_seqlens`, a non-16-aligned
final checkpoint, and an aligned track destination equal to the final state
slot. B12X's transactional metadata validation deliberately rejects these
inputs. The adapter now passes the live count from `cu_seqlens`, zeroes padded
output rows, and performs native-style aligned final-state copying after the
kernel instead of requesting an invalid/duplicate internal checkpoint. Zero
offset checkpoints are copied before the kernel without a CPU synchronization.
An asynchronous error-code assertion prevents silent NaN propagation from
future rejected metadata.

All seven GDN component tests pass after the fix, including the three tests
that failed before it. The original full sequence of API/prefix/diverse/1M
checks must still pass; this unit result alone is not long-context approval.

Evidence: `/tmp/qad-sgl-fp8-context1m-1/`,
`/tmp/qad-sgl-finite-diagnostic-1/`, `/tmp/qad-gdn-tail-before.log`,
`/tmp/qad-gdn-tail-after.log`; retained results `results/fp8-context1m-first-*`
and `results/finite-diagnostic-*`.

The candidate also filters Qwen4 quantized MTP file reads through the exact
checkpoint index. Its loader consumes only `mtp.*` tensors, so selecting
their shard(s) avoids scanning unrelated target/PLE files. Other models and
the target loader are unaffected; a missing indexed MTP shard is an error.
The source-filter unit test passed. Full-model loading is checked separately.

## Corrected FP8/GDN/PLE — actual 1M input completed

The full original probe sequence was rerun with the corrected GDN adapter:
all API, vision, MTP, streaming and prefix/multiturn checks passed first,
then diverse retrieval, then the uncached full-context request.

- Healthy after **462.3 seconds**. Target load: **385.02 seconds**; MTP load:
  **9.40 seconds**, with `QAD_MTP_SHARD_FILTER selected=1 total=36`.
- Diverse **130,925 input tokens**: `MAPLE,COMET`, **83.4 seconds**.
- **1,048,431 input tokens**: `MAPLE,COMET`, **1,045.054 seconds**
  (17 minutes 25 seconds), six output tokens and normal stop.
- Reported prompt count exactly matched the submitted token IDs;
  **cached_tokens=0**. The input was not truncated.
- The entire trial, including the subsequent serving benchmark, stayed above
  **14.21 GiB host MemAvailable**. The user-requested 3 GiB guard did not fire.
- FP8 target/MTP KV capacity: **1,048,576 tokens**; YaRN factor 4; 4,096-token
  prefill chunks; one running request, eight Mamba cache slots, memory fraction
  0.85. ASR/TTS were not GPU-loaded. Two submitted short requests in this
  profile were queued; true two-request execution was tested separately.
- The benchmark generated 256 tokens per response with exact prompt bytes
  shared with the vLLM benchmark, zero temperature and thinking disabled.
  Every response preserved ascending integer order. At 64,058 input tokens,
  cold total latency was **40.560 seconds**, repeated-prefix latency
  **5.050 seconds**. These are individual configured-runtime measurements.

For the same 1,048,431-token retrieval workload, the preserved vLLM result is
**1,229.367 seconds** (20 minutes 29 seconds). This SGLang run took about
15% less elapsed time. vLLM allocated 1,318,640 KV tokens and uses different
MTP/head policies; the memory numbers are not an equal-pool comparison.
See `PARITY.md` for the remaining engine differences.

Evidence: `/tmp/qad-sgl-fp8-context1m-2/`; retained JSON/probe output:
`results/fp8-context1m-corrected/`. The final packaging follow-up adds only
checkpoint mapping residency maintenance and explicit PLE-mode logging;
its numerical/graph and default-profile qualification are tracked separately.

## Final v3 packaging and default-profile qualification

Release image: `dgx-sglang-qwen38-qad:sm121-v3`, immutable ID
`sha256:5d6d286cba0451f4bb912127cae94c8c2ecb4e1bd33d284ba867a3c9595049cb`.
The release tag was assigned only after checking it is the exact image used
by the final integration trial.

The added PLE residency worker holds its source tensors alive, accepts only
verified immutable checkpoint file mappings, and never applies DONTNEED to
anonymous allocations. It accounts for each overlapping file VMA once and
trims in 64 MiB pieces. This covers decode CUDA graphs, which do not call
Python on each replay. The final component suite passed **13 tests** covering
GDN, QSA FP8, PLE I/O/changed graph replay/forced mapping eviction and the MTP
file filter. Actual first/middle/last checkpoint PLE mappings also passed
the file-mapping ownership check.

The final integration profile used 64K context, two running requests,
131,072 FP8 KV tokens, 512-token prefill chunks and an intentionally aggressive
**0.001 GiB** PLE mapping budget. Five real mapping trims were observed.
Text, reasoning, vision, tools/round trip, MTP, genuine concurrent requests,
SSE, 14K retrieval, cold/cached prefix branching and two-turn conversation
all passed. Diverse **65,398-token** retrieval returned `MAPLE,COMET` in
**62.0 seconds**. Healthy after **487.3 seconds**; minimum host available
memory **26.68 GiB**. Production uses the less aggressive 4 GiB mapping budget.

Standalone and Talk embedded runtime sources are synchronized, both default
Compose profiles select v3 and FP8/GDN/PLE, and the explicit standalone 1M
override resolves correctly. Relevant config, catalog, deployment, progress,
TP2-preservation and embedded-source tests passed. The Talk binary was built
at `/tmp/sparktalk-qad-tp1`. Running Talk was not restarted, no other GPU
service was started, and both isolated trial containers were stopped.

Evidence: `/tmp/qad-v3-final-component-tests.log`,
`/tmp/qad-ple-trimmer-cpu.log`, `/tmp/qad-ple-real-mappings.log`,
`/tmp/qad-sgl-v3-release-validation/`, `/tmp/qad-v3-talk-final-tests.log`,
`/tmp/qad-v3-talk-build.log`; retained final runtime/API/memory/trim records:
`results/v3-release/`.

### Talk TP1 1M deployment (2026-09-19)

User requested the Talk TP1 set use 1,048,576 context tokens with extra services,
ASR and TTS, excluding FLUX. Embedded Talk Compose now uses the qualified YaRN4,
FP8 KV, 1M token pool, one running request, Mamba8 and 0.5GiB PLE RSS profile.
Standalone default Compose remains 64K. Config revision13 upgrades built-in64K
TP1 and removes its FLUX membership; ASR/TTS start after LLM readiness. Custom
context limits and TP2 profiles are preserved. Missing-service legacy catalogs
filter absent bindings during fallback/migration.

Built and deployed Talk binary SHA256:
`a51753aaa7f3b05c40f2aebee470bca7df69b1b307c4882085d9f081959e07fb`.
Live image is `dgx-sglang-qwen38-qad:sm121-v3`. `/get_server_info` confirmed
context_length=max_total_tokens=1048576, max_running_requests=1, FP8 KV and TP1.
Talk selected_bundle=flash-next and context_tokens=1048576. FLUX is stopped;
existing extra services stayed running. ASR/TTS `/ready` both returned200/true.
A short chat returned the correct answer42 with normal stop (9.91s,29input,
3output tokens). MemAvailable after startup was approximately13GiB.
This deployment check did not repeat a full1M prompt with speech services loaded;
the earlier full1M benchmark above was LLM-only.

Validation: complete config package tests passed; focused FlashNext/QwenTP2/QAD,
embedded source and catalog orchestrator tests passed; build and diff checks
passed. Logs: `/tmp/talk-1m-config-tests.log`,
`/tmp/talk-1m-orchestrator-tests.log`, `/tmp/talk-1m-chat-check.json`,
`/tmp/talk-1m-server-info.json`. Previous binary and config retained under
`/tmp/sparktalk-before-1m*`.

### FLUX restored to Talk QAD set (2026-09-19)

At the user's subsequent request, config revision14 adds FLUX to the QAD TP1
set, with start_after_llm=true. Context1M, ASR/TTS and extras remain enabled.
Talk was rebuilt/restarted; the existing LLM and speech containers stayed up.
FLUX /health returned200/statusok and Talk reported the start operation complete.
LLM health and both speech readiness endpoints returned200. Idle MemAvailable
was approximately11GiB. This validates service startup, not peak memory during
simultaneous image generation and a full1M inference. Complete config tests
passed (`/tmp/talk-qad-flux-tests.log`); missing-service and TP2 preservation
checks remain covered.
