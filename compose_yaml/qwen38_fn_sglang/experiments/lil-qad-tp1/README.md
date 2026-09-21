# SGLang-first QAD TP1 compatibility work

Status: **FP8 KV + B12X GDN + checkpoint-backed PLE completed an actual
1,048,431-token request and correct answer in 1,045.054 seconds**.
Minimum available host memory over that trial was 14.21 GiB, with ASR/TTS
not GPU-loaded. CUDA graphs, vision, tools, MTP, concurrency and prefix reuse
have their own recorded checks. Default TP1 Compose/Talk remains 64K/two requests;
the explicit standalone 1M override is `../../compose.context1m.yaml`.
Historical stage Dockerfiles/profiles below are investigation artifacts;
the reproducible release build is `../../Dockerfile --target qad-tp1`.
Runtime adapters now live in `../../qad/`; this folder contains reproducible probes.
See [VALIDATION.md](VALIDATION.md) for the full-checkpoint boot and API results.
See [PARITY.md](PARITY.md) for engine differences and matched serving measurements.

Target: `local-inference-lab/Qwen3.8-Flash-Next-NVFP4`, revision
`7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd`.
Baseline image: `dgx-sglang-qwen38-fn:sm121-vocab1`.

## Findings

- The checkpoint's ModelOpt mixed configuration parses in the existing SGLang
  image. Using the model's actual `model.language_model.*` prefixes, attention
  resolves to MXFP8, routed experts to NVFP4, and shared experts to MXFP8.
  This is configuration dispatch validation, not kernel or full-load validation.
- Its PLE stores 128 packed U8 shards, E4M3 block scales and a shared
  `weight_scale_2`. Shard 0 has weight shape `[2500012, 80]` and scale shape
  `[2500012, 10]`, representing 160 columns with 16-element scale blocks.
- The unmodified baseline Qwen4 PLE file offload accepts BF16/FP8, not packed NVFP4. Loading
  U8 as numerical BF16 values or dropping block/global scales is invalid.
- `../../qad/nvfp4_ple.py` decodes the E2M1 representation.
  It decodes selected rows without expanding the complete table. Global FP32
  scale multiplication precedes BF16 output rounding.

## Verified on GB10, 2026-09-18

`test_nvfp4_ple.py` passed exact BF16 comparison against a separate PyTorch
lookup-table reference for all 16 E2M1 codes, varying FP8 scales, repeated IDs,
partition boundaries and out-of-partition zero output. The same test and CUDA
Graph replay passed on GPU tensors, pinned CPU tensors and the existing
SGLang file-backed host allocator, including file gather after fsync,
MADV_DONTNEED and POSIX_FADV_DONTNEED. These are synthetic component tests, not
full-checkpoint accuracy or performance results.

From repository root:

```sh
docker run --rm --gpus all --network none --memory 3g \
  --ulimit memlock=-1:-1 \
  -v "$PWD/compose_yaml/qwen38_fn_sglang/experiments/lil-qad-tp1:/probe:ro" \
  --entrypoint python3 dgx-sglang-qwen38-qad:sm121-v1 \
  /probe/test_nvfp4_ple.py
```

## Stage 1 history (superseded by stage 2 evidence in VALIDATION.md)

1. DONE in the isolated image: packed PLE construction, shard/scale/global-scale
   loading, prefetch and file residency trimming. TP1/file mode is required.
2. DONE for sampled actual checkpoint rows (0, 1, 100, final row of shard 0):
   exact BF16 agreement with the installed NVIDIA ModelOpt NVFP4 decoder.
3. DONE for full text-model loading and basic inference: mixed MXFP8/NVFP4
   projection/scale loading. Vision and MTP remain unqualified. Fail explicitly
   instead of falling back to incorrect unquantized weights.
   Static inspection found additional work for vision/MTP: the inherited
   Qwen3-VL constructor passes `quant_config=None` to its vision tower, while
   this checkpoint quantizes vision with MXFP8 and W4A16 NVFP4. MTP's W4A16
   expert dispatch also needs independent numerical/backend validation; the
   text model's global CUTLASS W4A4 selection is not proof of W4A16 correctness.
4. DONE with existing SGLang kernels, 4K context and one request: generated
   text, parsed tools and reasoning on/off. Images, longer context and B12X
   acceleration remain. Measure prefill, decode and MTP acceptance separately;
   the functional probes are not performance or model-quality benchmarks.
5. Only then migrate Compose/Talk TP1 identities, caches and progress handling.
   Keep the existing SGLang base image needed by TP2; do not overwrite it.
   The pinned checkpoint template rejects `reasoning_effort: none`; disabling
   thinking requires `chat_template_kwargs: {enable_thinking: false}`. Direct
   tokenizer rendering confirmed the rejection and the accepted toggle/low
   forms. Talk's current `qwen3.8` adapter sends top-level effort, so the TP1
   integration must verify and adapt this path without altering cloud/TP2
   model behavior.

## Source boundaries

- Checkpoint metadata: https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4
- SGLang runtime/interfaces: https://github.com/sgl-project/sglang (Apache-2.0).
- Optional B12X library: https://github.com/local-inference-lab/b12x (Apache-2.0).

## Integration checks, 2026-09-18

Image `dgx-sglang-qwen38-qad:tp1-trial1` is built from `Dockerfile` in this
folder. It modifies only the isolated image, not the production source/image.
`compose.yaml` is a short-context, eager, text-only, no-MTP trial on localhost
port 8016. This text-only validation is not yet a qualified production replacement.
The trial also adds the two Qwen architecture aliases to the CLI's
`language_model_only` allowlist; the inherited Qwen model and loader already
implement that mode, but the pinned CLI originally allowed only Muse.
Run `check_checkpoint.py <snapshot>` before startup; all 36 complete shards
must exist. Download is pinned to the revision above.
The trial uses a 0.80 static memory fraction because this budget includes model
weights, not only KV cache; 0.65 failed after a successful full load. KV capacity
is independently capped at 8,192 tokens with one running request. Both FP4 GEMM
and MoE runners are explicitly `flashinfer_cutlass`: mixed-format auto selection
otherwise chose the unsupported TRT-LLM NVFP4 MoE path at warmup.

- `test_ple_loader.py <sample>` uses samples from `fetch_ple_sample.py <dir>`.
  ModelOpt comparison passes, as do missing/duplicate tensor rejection checks.
  A further 4,096 randomized rows across five global scales match ModelOpt
  exactly, including all finite nonnegative E4M3 block-scale codes. FP32 scale
  reconstruction follows the same multiplication order as ModelOpt.
- `test_sglang_constructor.py` verifies that the patched NGram constructor
  selects packed U8/E4M3 file storage and rejects TP2 for the trial path.
- `test_quant_config.py` exercises the real file-based quantization loader.
  ModelOpt `config_groups` must reach the mixed parser directly instead of the
  older `quantization.quant_algo` discriminator used for single-format exports.
- `test_scale_loading.py` exercises the real Qwen4 load_weights method with
  a tiny fused GDN projection: MXFP8 weight_scale -> weight_scale_inv mapping
  is needed and both a/b partitions must load exactly.
- `test_mxfp8_linear.py` uses `/metadata/hf_quant_config.json` and a synthetic
  TP1 2560->96 MXFP8 layer. The baseline CUTLASS path rejected N<128.
  Zero-padding weight rows to 128, then cropping output, enables the operation
  without changing checkpoint values. Relative L2 versus unquantized-activation
  matmul was 0.027956; this measures the synthetic MXFP8 arithmetic path,
  not model quality. Native MXFP8 activation quantization remains enabled.
  Passing the complete snapshot directory as an optional argument tests actual
  checkpoint layer 0 fused b/a weights and E8M0 scales. That test passed with
  relative L2 0.025486 versus FP32 matmul of the dequantized checkpoint weights
  and BF16 input (the runtime also quantizes activations to MXFP8).

For a complete isolated boot/probe cycle after download:

```sh
python3 compose_yaml/qwen38_fn_sglang/experiments/lil-qad-tp1/validate_trial.py \
  --output-dir /tmp/qwen-qad-validation-1
```

The output directory must be new. This checks checkpoint completeness, requires
an idle GPU and 95 GiB available host memory before starting, monitors host
memory, and stops only the trial service after success or failure. It saves
server logs, API responses and memory samples. Startup has a 30-minute limit;
available host memory below 8 GiB stops the trial.

For an already running isolated Compose service, wait for `/health`, then run:

```sh
python3 compose_yaml/qwen38_fn_sglang/experiments/lil-qad-tp1/smoke_test.py \
  --output /tmp/qwen-qad-api-results.json
```

This checks deterministic arithmetic text, parsed reasoning, and a parsed
function call. It writes request/response evidence before assertions. It does
not test vision, MTP, long context or benchmark model quality.
