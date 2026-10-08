# Official Huihui BF16 delta → step-5500 QAD NVFP4/MXFP8

This adapter transfers the changes in the official Huihui BF16 checkpoint into
the newer LIL QAD hybrid checkpoint. It uses native Hugging Face tensor order.

Pinned inputs:

| Role | Repository | Revision |
| --- | --- | --- |
| QAD base | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` | `6909a5bed089a48fa07e956d3915af2537de9368` |
| Abliterated BF16 | `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated` | `298f94632b784e26a7fe576114f82066689d5baa` |
| Original BF16 | `Qwen/Qwen3.8-Flash-Next` | `de4b8e4d43b917e7706784d8bb445c9af86a3540` |

The source audit covers all 1,658 BF16 tensors. There are 144 changed tensors:
48 attention output projections, 48 shared-expert down projections, and 48
fused routed-expert down projections. Each fused tensor contains 512 experts,
so the transfer covers 24,672 matrices. The other 1,514 tensors are unchanged.
These source revisions differ from the older GGUF-based adapter's inputs.

## Download and identity checks

The audit binds Xet identities to the pinned Hugging Face LFS hashes. Equal
interior chunk hashes and exact boundary bytes establish unchanged tensors
without downloading both full 335 GiB BF16 checkpoints. Unresolved tensors are
compared using complete tensor bytes. `refine_xet.py` resolves chunk-boundary
ambiguities before selection; absence of a matching chunk alone does not prove
a weight changed.

`fetch_selected.py` downloads the selected ranges from both BF16 sources. Its
lossless Xet transport verifies chunk hashes and reconstructs the exact bytes.
It reuses earlier local data and writes a hash receipt for each complete pair.
The download plan must fit the explicit `--max-network-gib` allowance. Raw
decoded bytes and compressed payload transfer are recorded separately.

## Conversion

For each changed matrix, `build.py` computes in FP32:

`target = DQ(QAD5500) + Huihui_BF16 - Qwen_BF16`

It rounds the target to BF16 and encodes with the vendored NVIDIA ModelOpt
revision `87c9f8cf83021957d1a1a575c90c9a4eaaf7ef0c`. MXFP8 scales are refreshed.
For NVFP4, the encoder compares the original and refreshed scales, retaining
the smaller squared reconstruction error. This is a transfer of the exact
BF16 source delta followed by lossy quantization; it does not imply lossless
preservation of the delta or the source models' benchmark scores. There is no
fresh activation calibration or QAD training.

The candidate is an independent copy. All bytes outside approved weight and
weight-scale ranges must match the QAD source. Activation scales, PLE, vision,
MTP and tokenizer remain unchanged. The manifest records source and output
shard hashes, source tensor hashes, numerical errors and converter identity.
The original source is rehashed after conversion. A `.partial` directory is
renamed only when these checks pass.

Two metadata corrections are recorded separately: the missing
`text_config.ple_embedding_dtype = "nvfp4"` tag is restored after verifying the
128 packed PLE shards, and index `metadata.total_size` is recomputed from actual
tensor payloads. The tensor map, shapes, dtypes and shard headers are preserved.

`--stream-inputs` permits conversion while selected downloads finish. It waits
for verified per-tensor receipts and reconciles every consumed hash with the
completed audit before completing the candidate. Existing output directories
are rejected rather than overwritten.

## Validation

`test_numeric.py` checks decoder equivalence against ModelOpt, scale selection,
untouched byte ranges and the fused-expert mapping. `test_xet_audit.py` and
`test_xet_ranges.py` exercise the selective transfer path. `probe_numeric.py`
checks real source matrices; `probe_alignment.py` checks native HF alignment
against the QAD base, including a contrasting GGUF column permutation.

`launch_validation.py` requires the pinned production runtime image
`sha256:95b4633b7224deaad847a553c6550891fd3bdad97ad9d8e0c70c843239d26e72`.
It launches an isolated worker container with FP8 KV, MTP, file-backed PLE, and
the explicit PLE loader tag. The 1M configuration uses the existing factor-4
YaRN override. `validate_runtime.py` waits for the verified candidate, runs the
existing language/tool/vision smoke cases, and tests actual diverse long-input
retrieval at 64K and 1M. It records responses and stops its validation containers.
These are functional checks, not a comprehensive model-quality or refusal-rate
benchmark. Runtime qualification and production promotion are separate actions.

## Qualified artifact (2026-10-04)

The completed worker artifact is:

`192.168.100.60:/home/edp1096/.cache/model-download-jobs/qwen38-qad5500-bf16/outputs/Huihui-Qwen3.8-Flash-Next-NVFP4-QAD5500-BF16Delta`

All 43 weight shards (99.0317 GiB), source hashes and
untargeted byte regions passed verification. The 24,672 matrices were converted;
18 output shards differ from the pinned QAD base. Relative L2 reconstruction
errors across source tensors range from 1.5704% to 3.6655%. Original NVFP4 scales
were selected for all 24,576 routed-expert matrices.

| Qualification | Actual input tokens | Retrieval | Elapsed |
| --- | ---: | --- | ---: |
| Candidate 64K | 65,398 | MAPLE, COMET | 38.93 s |
| Candidate 1M | 1,048,435 | MAPLE, COMET | 1132.24 s |
| Pinned QAD base 1M | 1,048,435 | MAPLE, COMET | 1128.03 s |

Both candidate configurations passed language, arithmetic, JSON, sorting, code,
fiction, assembled tool-call, vision and short retrieval probes. The generated
code passed four behavior cases. Parsed reasoning was verified separately at
64K. Short generation after each full-context request also passed. Validation
used TP1, concurrency 1, FP8 KV, MTP3 and file-backed PLE; 1M used factor-4 YaRN.
The two validation containers exited cleanly with no OOM kill.

See [conversion summary](docs/conversion-summary.json),
[source audit](docs/source-audit-summary.json), and
[runtime results](docs/runtime-summary.json). These checks establish the tested
functional behavior; they do not measure broad benchmark quality or refusal rate.
