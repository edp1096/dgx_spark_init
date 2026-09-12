# SparkTalk integration: tool support and context budgets

Two serving-configuration omissions caused consecutive HTTP400 errors.

1. SparkTalk sends tools plus `tool_choice: auto` whenever tools are enabled.
   The original launch omitted automatic tool selection and a tool parser, so
   even an ordinary identity question was rejected before generation.
2. The model still used the 2,048-token benchmark context. Actual SparkTalk
   configuration was a 65,536-token window, 8,192 output reserve and 2,048 safety
   margin. Its next request was rejected because the output allowance alone
   exceeded the model window. The real input also exceeded 2,048 tokens.

The first fix was tested with small API fixtures using a 256-token output
allowance. Those fixtures verified parsing but did not verify the real
SparkTalk input/output budgets. They were insufficient grounds to claim that
the full application connection was ready.

## Serving changes

`launch.sh` now uses:

```text
--enable-auto-tool-choice
--tool-call-parser deepseek_v41
--reasoning-parser deepseek_v41
--max-model-len 65536
--kv-cache-memory-bytes 2147483648
```

The parsers are present in the pinned vLLM image; no custom parser was added.
The model still defaults to thinking off, with separate reasoning when the
request enables it. `DSV41_MAX_MODEL_LEN` and `DSV41_KV_CACHE_BYTES` are forwarded
to both ranks by `manage.sh`. The memory guard adds KV allocations above the
original 512 MiB allowance to its required host headroom.

KV allocation rises by 1.5 GiB per rank. Expert payload stays exactly
87,836,590,080 bytes, uniform224 target slots, original weight precision and
DSpark5. No SparkTalk configuration or application source change was needed;
its existing 65,536-token window and 8,192-token output reserve are preserved.
No other models or support services were restarted.

Historical performance results retain their original context 2,048 / KV512 MiB
conditions. Use explicit overrides to reproduce those trials; they are not
measurements of long-context throughput.

## Validation

`tools/check_tools.py` passed live streamed auto/no-call, automatic function name,
arguments and ID, an arithmetic tool-result round trip, and separated reasoning
with `tool_choice: none`. The initial reasoning fixture incorrectly assumed an
auto-capable model would never use addition tools for multiplication; the model
legitimately selected them. That raw result is retained separately. The final
fixture explicitly disables tools for the reasoning-only check.

`tools/check_sparktalk.py` uses the running application's `/api/chat` handler, without
replacing its prompts, tool registry, reasoning preference or output budget.
It creates an isolated temporary session, sends `넌 누구냐?` with tools enabled,
records the SSE stream and deletes only that temporary session afterward.

The actual application run completed with a nonempty Korean answer identifying
itself as SparkTalk and a `done` event, with no error event:

- Actual input: **6,384 tokens**, versus preflight estimate 6,627.
- Configured output allowance: **8,192 tokens**, preserved.
- Context window 65,536; input budget 55,296.
- HTTP 200; elapsed 95.330 seconds, including prefill, reasoning and response.
- Test session cleanup succeeded; user conversation history was not modified.

These measurements describe one fresh application request, not an optimized
latency guarantee. Evidence: `results/sparktalk-context-validation.json` and log.

`tools/check_context_capacity.py` additionally requests an 8,192-token output allowance
with roughly 55,000 input tokens and retrieves an exact marker from the middle.
Its repetitive input checks extended positions, chunked prefill and KV storage;
it is not a broad long-context language-quality benchmark. Raw results are in
`results/context-capacity-validation.json` and log.

Completed capacity result: 54,988 input tokens plus an 8,192-token requested
output allowance, exact marker `spark-amethyst-642`, normal stop, HTTP 200.
First content arrived at 52.217 s; total 52.690 s. A subsequent short request
using the same 8,192-token output allowance returned exactly `OK`, confirming
that the engine also handles the transition back from the long request.
`results/context64k-runtime-validation.json` records final options, unchanged
expert payload, host memory and health; both engine logs are retained.
