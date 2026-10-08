# Phase-aware V4.1 prefill: applicability to the current TP2 engine

The user supplied the full continuation of [Antirez's September 11 post](https://x.com/antirez/status/2098422997719650788).
It describes three choices based on pending prefill size: existing expert-cache
execution for small inputs, layer-major execution that overlaps loading layer
N+1 with computing layer N for larger inputs, and a fully resident encoder for
very large inputs. The reported Q2 M5 Max 128 GB example spends about eight
seconds switching residency and then reaches about 800 prompt tokens/second.
These are the author's measurements, not predictions for this TP2 setup. The
post also warns that residency changes displace the useful decode cache and
that full residency can fail to beat layer-major execution on some hardware.

The currently fetched public ds4 main revision was
`6289c516273979173abbc062209a81dd3706b804`, committed September 8. Its code and
streaming documentation predate this post; the new three-mode implementation
was not established from that revision. The design discussion below therefore
uses the supplied post, the actual installed vLLM code, and the
[official V4.1 reference](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py).

## What is already implemented here

The scheduler currently runs a token chunk through the model layers, then
advances to the next chunk. The native MoE helper keeps a group of actual
routed experts resident while computing all tokens of that chunk. Its normal
read overlap is with the same layer's shared expert. The optional routed
staging experiment overlaps a separate expert group of the same layer.
Neither is the author's layer-N+1 loading / layer-N computation schedule over
an entire large prefill.

The installed `patches/target_model.py` forward loop passes the whole token
array through all 40 target layers. The checkpoint's shared-KV sources are
layers 2/8/14/20, while its DSpark targets are 37/38/39. The DSpark implementation
projects those late-layer hidden states and inserts them into its own
128-token sliding-window context caches. The reference also seeds DSpark's
window during prefill. This makes a wholesale removal of decoder work unsafe:
the remaining required query rows, draft context and prefix-cache checkpoint
validity must be handled together.

A promising implementation would compute the encoder/cache-producing work for
all new tokens, retain only the decoder rows actually needed by logits and
DSpark's context, and preserve valid cache checkpoints for reuse and rollback.
The exact boundary includes source projections and cannot be implemented by
merely dropping half the layers. First validate logits, greedy continuations,
DSpark acceptance, changed-prefix reuse and long contexts. Only then evaluate
residency redistribution and layer-major execution.

## Memory arithmetic, not an implemented allocation

The existing packed format uses 9,400,320 bytes per expert per rank. Twenty
complete 384-expert layers would use 67.236 GiB per rank. Twenty other layers
with 64 slots would use 11.206 GiB, and the three 128-expert draft layers use
3.362 GiB. Together that is 81.804 GiB, exactly the current expert-slot budget
of 40 × 224 + 3 × 128 slots.

This demonstrates a possible redistribution without extra weight quantization;
it does not prove that encoder prefill needs exactly those twenty whole-layer
expert stores, that a 64-slot decoder cache is fast enough, or that the larger
execution plans fit the present scratch arena. Dense weights, KV, activation
buffers and scratch remain additional. A phase switch must safely migrate or
evict cached entries and must never let captured kernels read repurposed
storage. Switching back for decode costs reads too.

Applying that static redistribution while still running all tokens through all
40 layers could make the decoder half read substantially more from SSD. The
execution change and the cache allocation need to be evaluated together.

## Selection policy

Use newly uncached tokens as the size input, not total conversation length.
A conversation with six thousand cached prompt tokens and two hundred new
tokens should usually retain its decode-friendly cache. Compare complete
request latency, including encoder loading and the decoder-cache refill, rather
than displaying the resident kernel's pp alone.

For a rough break-even estimate, let L be the residency-switch time, D the
extra first-generation/cache-recovery time, P0/P1 the measured old/new prefill
rates, and N the uncached tokens. The candidate helps only when
`L + D + N/P1 < N/P0`. Those values must be measured on these two GB10s at the
original weight precision. The tweet's eight seconds and 800 tok/s cannot be
substituted as local measurements.


A further source to audit is the [4× RTX PRO 6000 SGLang deployment](https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000/blob/main/boot.py),
which enables `--enable-decoder-swa-bounded-replay` with DSpark. Its underlying
implementation and compatibility with this custom vLLM TP2 engine have not been
validated; this is a research lead, not a vLLM option to turn on here.


## Follow-up qualification

The official [LMSYS V4.1 description](https://www.lmsys.org/blog/2026-09-10-deepseek-v41)
clarifies that decoder bounded replay truncates local attention at the retained
tail boundary and is an approximation. It cannot be assumed equivalent merely
because the checkpoint precision is unchanged.

An exact-dependency final-layer routed-expert reduction was implemented and
measured in [DECODER_ROWS_EXPERIMENT.md](DECODER_ROWS_EXPERIMENT.md). It preserves
all cache-producing work and DSpark aux rows, but gained only about 2% PP and
lost about 8% matched-output TG. It remains default OFF. This is not the broad
encoder/decoder phase-aware implementation proposed above.
