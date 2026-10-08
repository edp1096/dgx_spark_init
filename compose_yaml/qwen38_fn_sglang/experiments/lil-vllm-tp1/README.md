# LIL/B12X vLLM TP1 1M qualification

Host: gx10-2 (`spark-f05a`), GB10. Existing production services are not replaced.
Model: `local-inference-lab/Qwen3.8-Flash-Next-NVFP4`, pinned snapshot
`7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd`.

Reference: https://github.com/eugr/spark-vllm-docker/blob/main/recipes/qwen3.8-flash-next-nvfp4-solo.yaml

Base image digest: `eugr/spark-vllm-b12x@sha256:8e7e062186f841453ef0ec6f713043c5b65447decc3835206685128c18e42262`.
vLLM `b40673cd006bf3496fdd70361dad2ad29eff54e7`; B12X
`9043b448622764a598969518d413b3fd8b3c0c07`.

## Configuration

FP8 KV; BF16 model dtype; mixed ModelOpt weights; B12X MoE, MXFP8 linear
and GDN; MTP four draft tokens; disk PLE (B12X io_uring); AOT and CUDA
graphs; memory utilization 0.8; chunked prefill 4096.
The image selects Marlin for unsupported B12X NVFP4 dense layers.

Changes from the reference recipe: one running request, max context
1,048,576, YaRN factor 4 with original context 262,144.
Host available-memory floor: 3 GiB, sampled every two seconds.
Only localhost port 8017 is exposed. The developer cache-reset endpoint
is enabled for uncached probes. Unlimited container memlock is necessary
for the B12X weight allocator. The io_uring path uses an unconfined seccomp
profile in this isolated trial. Model files are mounted read-only and caches
are separate from production.

## YaRN compatibility

The pinned vLLM image applies dictionary `hf_overrides` only to the target,
not to its MTP draft. A read-only test config overlay therefore supplies the
same YaRN parameters to both. The original checkpoint is not edited.

Loading this YaRN config exposed a constructor compatibility bug: the
enclosing Qwen config lacked `max_position_embeddings` before Transformers
validates its inherited rope parameters. The derived image's Dockerfile
adds exactly one assignment from `text_config.max_position_embeddings`
before that validation. It changes no weights, kernels or native context
value. Both target and draft then report maximum length 1,048,576.

## Procedure

`validate.py` starts only the named trial, monitors host memory, launches
`probe.py` when healthy, and saves image/arguments, memory samples, logs
and an explicit outcome. It stops the trial on completion, guard violation
or failure. Existing GPU work prevents trial startup.

The probe checks short text, then 130,925 tokens of diverse synthetic text,
then about 1,048,431 tokens of repeated filler with two named values at
separated positions. Prefix cache is reset before each long request.
API usage must agree with the tokenized input; answer correctness, normal
stop and SSE completion are checked. This is a capacity/retrieval check,
not broad long-document quality or eight-request concurrency qualification.

```sh
docker build -t dgx-qwen38-lil-vllm:context1m-v1 .
python3 validate.py --output-dir /tmp/qad-vllm-new-run
```

## Initial attempts

Attempts 1–4 exposed trial setup problems (memlock and YaRN config
construction), before long-input testing. Attempt 5 booted and passed short
text, but the test script called the disabled cache-reset endpoint and
received HTTP 404. These do not establish a long-context capacity failure.
Attempt 6 enables that endpoint and runs the full input probes.

## Completed result — 2026-09-19 KST

**Passed on one GB10, TP1, with FP8 KV and MTP enabled.**

| Actual input tokens | Answer | End-to-end time | Finish |
| ---: | --- | ---: | --- |
| 130,925 (diverse words) | MAPLE, COMET | 75.36 s | stop + SSE DONE |
| 1,048,431 (repeated filler) | MAPLE, COMET | 1,229.37 s (20m 29s) | stop + SSE DONE |

Each returned six completion tokens. API `usage.prompt_tokens` exactly
matched the submitted token IDs, with no truncation. The usage response
does not expose cached-token details in this image; uncached execution is
supported by the successful engine cache-reset logs before each request
and the final 0.0% prefix-hit rate.

The successful run allocated **1,318,640 KV tokens / 18.24 GiB**.
The minimum host available memory across initialization and both requests
was **11.15 GiB**, above the 3 GiB guard. No swap growth from the starting
sample, no guard trip, and `OOMKilled=false`. Trial stopped normally with
exit 0 after success. Host available memory returned to about 117 GiB;
existing support services remained running.

API became healthy after 178.0 seconds with compiler caches from earlier
startup attempts. Model loading reported 74.12 GiB and 57.34 seconds.
Neither figure is a controlled speed/memory A/B against SGLang or the old
Huihui checkpoint. The API logger's interval throughput is not the measured
end-to-end prefill rate; use the request timings above.

Scope: one long request on the isolated GPU, no concurrent ASR/TTS load.
The short completion and two-value synthetic retrieval establish capacity
and this retrieval result, not broad 1M accuracy, video/image quality, long
answer generation or concurrent request stability. Production Talk/Compose
was not switched to vLLM. The one-line YaRN constructor fix described above
is part of the tested image; an unmodified base-image 1M pass is not claimed.

Evidence: [summary](results/context1m-20260919/summary.json),
[raw API results](results/context1m-20260919/api-results.json),
[server log](results/context1m-20260919/server.log),
[memory samples](results/context1m-20260919/memory.jsonl).
Full original run directory: `/tmp/qad-vllm-context1m-6/`.
