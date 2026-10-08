# Final decoder routed-expert selection: measured, default OFF

The experimental `DSV41_FINAL_DECODER_ROWS=1` path preserves all target SWA and
compressed KV writes, every DSpark auxiliary hidden row, and the existing
128-token prefix retention. It selects **only routed-expert computation in final
target layer 39** for actual logits rows. Attention, its output projection,
router, shared experts, hyperconnections, collective shapes and normalization
retain the original full-row execution. Unused final output rows omit routed
expert contributions; no consumer or future KV depends on those rows. The
complete DSpark aux list consists of outputs through layer 38.

This is a limited final-layer optimization, not decoder-half tail-only prefill.
Dense prompt logprobs, multimodal input, decode, profiling, MTP-only speculation,
sequence parallel, and incompatible auxiliary-output layouts use the original
path. The launch flag remains default OFF because the TG regression outweighs the small PP gain. Benchmark control can
select it per reset epoch. An optional untimed full-layer comparison checks the
selected states against a full final-layer replay on the identical input.

### Rejected first variant

The first variant also selected final attention queries and output projection
rows. On 6308 input tokens, its actual full-layer comparison exceeded the chosen
state tolerance (max absolute differences up to 1.0 in FFN output, 0.25 in the
residual). This is not a qualified correctness result. Its measured state logs
are retained in `results/decoder-rows-validation-retry.log`; no speed result or deployment claim
is made for that variant. The current candidate preserves those dense operation
shapes and only prunes the streamed experts. Syntax checks alone do not qualify
it; the accepted narrower path and its measurements are described below.

## Architecture constraint

Every decoder layer also owns local SWA KV. Keeping only 128 input rows through
all decoder layers truncates attention dependencies. The official
[LMSYS September 10 description](https://www.lmsys.org/blog/2026-09-10-deepseek-v41)
explicitly calls decoder bounded replay an approximation. Original weight
precision alone does not make that execution numerically equivalent.
Larger exact pruning would require accounting for each layer's receptive field
and all reusable checkpoints. Existing dense 128-token checkpoints constrain
that pruning; silently publishing incomplete checkpoints is not acceptable.

## September 12 startup interruption

Both model containers were stopped normally, then the monitored baseline launch
started head container at 17:56:30 KST with scheduler 4096, BENCH_CONTROL=1 and
FINAL_DECODER_ROWS=0. No measurement or validation request was submitted.
The last persisted API log was configuration setup at 17:56:49, before engine
readiness or recorded expert-slot allocations. Head telemetry at 17:56:55 had
123,427,876,864 bytes available and cgroup current/peak 2,435,207,168 bytes,
with zero OOM events. Those are last observations, not guarantees about the
unobserved interval before reset.

Head boot history then contains starts at 17:57:15, 17:57:42 and 17:58:14.
Its previous boot ended without a persisted OOM/Xid/panic cause. Worker boot ID
was unchanged; the peer recorder deliberately killed the labelled worker after
losing the head monitor. Exit 137 on that worker is not evidence of worker OOM.
No host reboot command was issued by this experiment. Exact cause is unknown.

Raw head evidence and independent worker mirror are preserved under
`~/.local/state/ds41-probes/20260912-decoder-test{,-worker}/`.
Machine-readable summary: `results/decoder-rows-startup-interrupted.json`.
Watchdogs have been stopped. Both model containers remain stopped, with the
candidate flag OFF. Talk UI was restarted with existing external-runtime config.
Python syntax and shell checks pass; GPU correctness and speed measurements
remain outstanding. Do not describe this candidate as validated or faster.

## Completed retry: correctness and measurements

The user authorized continuation after the host-reset incident. Subsequent
monitored model starts and measurements kept head boot ID
`309849e3-8913-4921-8711-058c0d6ad092`; worker boot ID stayed unchanged.
This does not establish the cause of the earlier reset.

The routed-expert-only candidate compared five selected final-layer state
components against an actual full-layer execution on the same input. For 6308
and 12934 input tokens, all 12 rank/chunk checks had max absolute difference
**0.0**. Two additional 3399-token code tests also had zero differences on both
ranks. All target cache-producing work and the full DSpark aux list execute
before the omission; the generic final MTP buffer is not consumed by DSpark
when that aux list is present. MTP-only/dense-output configurations fall back.

### PP and TTFT

Same frozen Talk/tool and mixed-document fixtures, scheduler 4096, native expert
capacity 2048, shared I/O, scratch 384 MiB and uniform 224/128 expert slots.
Each timed request used a unique KV namespace and reset expert maps on both
ranks. Three trials per input and mode, alternating order. Four earlier
same-process warmups were excluded. A user-reported concurrent request caused
the initial timing run to be excluded and every timed trial to be repeated
once the server was idle. The clean harness checks idle status before and after
each request. No validation double-execution is included in timings.

| Input tokens | Original PP | Selected PP | PP change | Original median TTFT | Selected median TTFT |
|---|---:|---:|---:|---:|---:|
| 6308 | 191.92 | 196.05 | +2.15% | 32.879 s | 32.181 s |
| 12934 | 241.52 | 246.09 | +1.89% | 53.557 s | 52.453 s |

All timed responses returned the correct marker and stopped normally.

### Cache reuse, output behavior and TG

Original -> candidate -> original prefix reuse returned the correct changed
markers, including a junction inside the old prompt. Cache hits were 3200 and
2560 tokens. Prompt-logprob requests correctly disabled selection.
Generated binary-search functions passed 402 cases each; both explanatory
answers correctly distinguished concurrency and parallelism.

Literal free-generation equality is not a sound gate for this pinned runtime:
two original-path requests with the same seed also produced different wording.
`decoder-repeat.json` retains those controls and candidate internal-state checks.
Free-generation TG was therefore not treated as the decisive matched-output
speed comparison.

The additional TG test copied fixed code and prose after the same long context.
Each mode ran both texts twice in alternating order. All eight responses matched
the required output exactly, with matching token counts. Each mode generated
368 tokens after first tokens:

- Original: **26.8248 tok/s** (13.7186 seconds).
- Selected experts: **24.6894 tok/s** (14.9052 seconds).
- Change: **-7.96%**.

Decision: keep `DSV41_FINAL_DECODER_ROWS=0` as the default. The roughly 2% PP
improvement is not worth this TG loss. The experimental implementation remains
available for explicit tests, with original checkpoint precision unchanged.
Reduced final-layer expert-cache coverage after prefill is a plausible tradeoff;
this experiment did not isolate per-layer decode stalls to prove attribution.
Broader decoder pruning or phase-dependent expert residency is not implemented.

### Artifacts and rerun

- `results/decoder-rows-correctness.json`: full vs selected states.
- `results/decoder-rows-ab.json`: clean PP A/B, hashes and settings.
- `results/decoder-rows-continuation.json`: cache/fallback/behavior checks.
- `results/decoder-repeat.json`: repeated controls and code-state validation.
- `results/decoder-fixed-output.json`: matched-output TG test.
- `results/decoder-rows-summary.json`: combined decision and summary.
- `results/interrupted-user-query-*`: excluded initial timings.
- `results/decoder-experts-complete-rank*.log`: complete model logs.

`tools/check_decoder_rows.py --fixtures PRIVATE_JSON` validates the selected state;
`tools/bench_decoder_rows.py --fixtures PRIVATE_JSON` measures PP;
`tools/check_decoder_continuation.py` checks cache and behavior;
`tools/bench_decoder_fixed_output.py` performs the matched-output TG follow-up.
These require a dedicated BENCH_CONTROL=1 serving session. Normal serving must
use BENCH_CONTROL=0 and remove the temporary control files afterward.


## Final serving state

Restored and validated with scheduler 4096, kernel 2048, shared I/O, DSpark 5,
65536 context and 8192 output allowance. Both ranks have BENCH_CONTROL=0 and
FINAL_DECODER_ROWS=0; temporary control files are removed. Both APIs return HTTP
200 and both model containers are running without OOM. The external probe
watchdogs were stopped after validation; model serving remains active.

Real SparkTalk checks returned both requested markers, stored metrics matched
SSE metrics, and the temporary session was deleted. First request TTFT was
35.4328 seconds; follow-up TTFT was 8.7907 seconds with 6144 cached tokens.
These are restored-baseline connection checks, not candidate speed results.
`results/decoder-final-runtime.json` verifies settings and source hashes on both
hosts; `results/decoder-final-sparktalk.json` records the application checks.
Memory summaries are in `decoder-probe-memory.json` and
`decoder-final-memory.json`. Python syntax, shell syntax and diff checks pass.
