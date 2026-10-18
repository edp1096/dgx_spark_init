# Qwen 3.8 Flash-Next FP4-Local

Serves `edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-FP4-Local` with the
GPU-tested B12X runtime, factor-4 YaRN (1,048,576 tokens), FP8 KV, disk-backed
PLE, and four MTP draft tokens. The original checkpoint config stays read-only;
`runtime-config.json` supplies the tested YaRN overlay.

The checkpoint must already exist under
`$HOME/.cache/huggingface/edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-FP4-Local`.
The checkpoint is published on [Hugging Face](https://huggingface.co/edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-FP4-Local)
at revision `d1ec8b01dd859336d7905db2ad2ceb725146008c`. Talk currently checks
the local artifact; download it into the path above before starting. Override
`SPARKTALK_HF_CACHE` when using another cache root.

```bash
docker compose build
docker compose up -d
docker compose logs -f runtime
docker compose down
```

The default endpoint is `http://127.0.0.1:18004`. Build inputs are also embedded
in Talk, so its managed launcher does not depend on this checkout. Talk exposes
this as a separate `Qwen 3.8 Flash-Next FP4-Local` set with ASR, TTS and Extra services. Auxiliary models load on demand after
the LLM is ready, so their startup allocations do not overlap its cold load.
Existing set selections and the startup default are preserved when it is added.
The request profile `qwen3.8-vllm` explicitly passes Thinking off/Low/Medium/XHigh
to the model template; its default is Medium.

Measured model allocation was 70.69 GiB, with about 100 GiB of additional host
memory while idle at the 1M setting. Cold loading reached about 115.4 GiB of
additional host memory and used swap. Talk reserves 116 GiB for startup and
retains the user's configured memory reserve. For this fixed recipe, Talk can
automatically reclaim file caches and start under live memory monitoring when
the conservative estimate misses by at most 2 GiB. It checks available memory
every 250 ms, stops only the starting model if the configured reserve is crossed,
and retries once after cleanup. Larger shortages and custom memory budgets
retain the strict preflight check. The native PyTorch allocator uses a 0.7
garbage collection threshold. Kernel compilation may take about 20 minutes
on a first start. Compiler and checkpoint caches are isolated from other models.

Both the NVIDIA baseline and this checkpoint retrieved two fields from an
actual 1,048,056-token prompt. One candidate smoke check failed an integer-only
output instruction despite calculating the correct answer. This integration
does not upgrade that result to an all-checks pass. See the checkpoint's
`runtime-qualification.json` for the full measurements and limitations.
