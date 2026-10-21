# Qwen FP4-Local: B12X integration experiment

Optional research adapter. The default Talk and standalone TensorFold recipes do not enable it.

The original FP4-Local weights remain read-only. This adapter replaces routed experts in the 48 main layers; shared experts, MTP and other operations remain in TensorFold. It uses a lossless MMA-packed layout and FP32 router-weight combination. `TF_B12X_WHOLE_K=1` tests full-K tile scheduling to avoid row-dependent split-K accumulation.

- `patches/`: changes against the qualified TensorFold source and the B12X implementation identified in `manifest.json`.
- `adapter/`: Python/C ABI graph bridge. Paths target the isolated `/bridge` test mount.
- `tests/`: row-position, replay, FP32-reference and full-model cached/MTP checks. Run them in the matching isolated test image.
- `control/`: isolated vLLM startup cache-trim diagnostic; never applied to production.
- `manifest.json`: tested images, model revision, controls and evidence directory. `Dockerfile.whole-k` expects the identified original B12X source tree plus the kernel patch in its build context.

The adapter's retained graph and per-shape scratch caches are experimental and unbounded. Production adoption requires memory-lifecycle qualification as well as passing full-model correctness and performance checks.

The tested B12X candidates were slower than native TensorFold. The revised candidate passed 6/8 full-model cached/MTP output-equality checks; native passed 8/8. The remaining failures are Medium output/token differences, not evidence of an incorrect arithmetic answer. B12X was not adopted. The new 1M vLLM control crossed the startup memory floor and produced no usable speed measurements, including with the diagnostic cache trim; this does not establish that 1M vLLM is impossible.

Results: [comparison report](../../docs/qwen38-fp4-local-b12x-followup-20261020.json). Single-layer equality does not establish full-model equality or broad model-quality parity.

Source: [TensorFold](https://github.com/ashhart/TensorFold/tree/db281878ddb836fd0df510d8771ecb7e0fe47d26), [Mia FP4 recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/tree/656879fd752749bbd11a6e1a065a326c00940911), [B12X](https://github.com/local-inference-lab/b12x), [FlashInfer B12X](https://github.com/flashinfer-ai/flashinfer/tree/601cb12713fd73f1744b15fdea6f9ddaaf2b4ec3/flashinfer/experimental/b12x). The tested B12X image contains additional changes; its file hashes and upstream comparison are preserved in the evidence directory.
