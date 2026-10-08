# GLM native adaptive-depth experiment: not promoted

The fixed-depth Entrpi v2.3-tier1 baseline completed 24 requests. The adaptive
candidate failed during engine initialization, before loading weights or generating
any responses:

```
DFlash2 drafts at a fixed physical depth; num_speculative_tokens_per_batch_size
and adaptive_speculative_tokens_window are not supported.
```

The generic scheduler/configuration has an acceptance-length controller, but the
DFlash2-specific speculator explicitly rejects it. Removing that check is not an
implementation of dynamic-depth support. Production Compose, entrypoint and defaults
are unchanged. The attempted changes are preserved in `adaptive-attempt.patch`;
`native-capabilities.json` records the source identities and the runtime finding.

MiaAI main 9c0794b also uses a different scheduler and EXL3 implementation: its
adaptive patch fails the first exact-source anchor against this image, and its E3
kernel does not plug into the installed B12X planned Trellis prefill path. Neither
MiaAI's E3 nor its FP8 dense projection changes were installed. A backend migration
requires a separate candidate image and parity tests; no speedup is claimed here.

## Historical protocol

Both requested profiles: 128K maximum context, two sequence slots, batch 4096,
8 GiB KV pool, fixed DFlash physical depth 7, temperature 0. Candidate requested
native adaptation window 8. Six exact-answer/tool prompts and two prose prompts
ran three times sequentially. Existing Lovesenko o_proj transplant and MXFP8
DFlash2 weights were retained. Six baseline prose responses reached the 768-token
limit, so their timings are bounded generation measurements, not completed-answer
quality validation. Slot count 2 is a configuration, not a concurrent-load test.

`run_update_trial.py` retains the trial orchestration. It requires applying the
archived patch to the repository first, and the pinned runtime is known to reject
the adaptive arm. Do not expose it in SparkTalk as a working feature. The private
profiles stay under ignored `results/`; public reports must not copy `.runtime`.
