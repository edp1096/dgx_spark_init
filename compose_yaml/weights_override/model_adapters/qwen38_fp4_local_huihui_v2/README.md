# Qwen3.8 FP4-Local + Huihui v2

Transfers the audited Huihui v2 Q8_0 delta onto NVIDIA FP4-Local. This is a
quantized delta transfer, with no new QAD training or activation calibration.
The output is an independent checkpoint, never a hard-linked modification of
the source.

| Input | Pinned revision |
| --- | --- |
| `nvidia/Qwen3.8-Flash-Next-FP4-Local` | `b9ce18a6887e18c089fdea8509a33e63429a3300` |
| `unsloth/Qwen3.8-Flash-Next-GGUF`, `Q8_0/` | `766911a6b7369840a91dbcd95f9f997acaab6cd6` |
| `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF`, `unsloth-Q8_0/` | `43b4c1a8871afcc27e37bf996a431528b3d957e5` |

The source audit covers all 1,224 text tensors and all 334 vision tensors.
The 96 changed projections are 36 GDN output projections, 12 attention output
projections and 48 shared-expert down projections. The routed experts and
vision weights match. The shared-expert changes must be included even though
the donor description says experts are unchanged.

For each changed projection, the builder computes
`MXFP8(BF16(DQ(NVIDIA) + restore_HF(DQ(Huihui) - DQ(Unsloth))))`.
GDN's GGUF column permutation is reversed before applying the delta. The
builder checks numeric alignment and quantization errors, then hashes every
output shard and every byte outside the permitted weight/scale intervals.
Routed NVFP4 weights, CSF scales, PLE, MTP, vision and runtime metadata retain
the NVIDIA checkpoint bytes.

## Pipeline

Use an empty job directory with pinned Hugging Face API responses in
`metadata/{base,original,huihui}-api.json` (include `blobs=true` so file sizes
and LFS hashes are available). Run these scripts from this directory:

```bash
python3 download.py --root "$FP4_JOB" --workers 2
python3 audit.py --root "$FP4_JOB" --workers 2
python3 build.py --root "$FP4_JOB" --probe
python3 build.py --root "$FP4_JOB" --output "$FP4_OUTPUT"
```

`download.py` verifies all NVIDIA file hashes. `audit.py` uses matching LFS
hashes, Xet content trees and exact boundary checks for unchanged text, full
Q8 ranges for the changed projections, and a full payload hash for vision.
It does not store the two full Q8 checkpoints. `build.py` needs PyTorch and
ModelOpt's `MXFP8QTensor`; its manifest records the actual quantizer version
and implementation hash. It writes to a new `.partial` directory and renames
only after verification; an existing output or partial directory is refused.

The qualification run uses the local ModelOpt image recorded in
`runtime/modelopt-image.json`, with CPU-only conversion, a read-only source
mount and an 8 GiB memory limit. The exact invocation is retained in
`reports/build-invocation.json`. Allow about 195 GiB for inputs and output,
plus runtime caches and container images; no complete donor Q8 download is
needed.

## GPU qualification

Build the pinned local runtime from `Dockerfile.runtime`. It includes two
checked patches needed for YaRN: expose the composite config's text context
length before Transformers normalizes RoPE, and separate MRoPE's video cache
capacity from YaRN's original training length. `test_runtime_rope.py` compares
native and extended positions with the reference implementations. The
launcher requires that receipt for the exact image/config before a 1M run.

```bash
docker build -f Dockerfile.runtime -t local/qwen38-fp4-local:csf-yarn-20261016 .
python3 launch.py --root "$FP4_JOB" --model "$FP4_JOB/base" --label base --dry-run
FP4_RUNTIME_ID=$(docker image inspect --format '{{.Id}}' local/qwen38-fp4-local:csf-yarn-20261016)
docker run --rm --network none --entrypoint python3 \
  -e CUDA_VISIBLE_DEVICES= -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$PWD/test_runtime_rope.py:/check.py:ro" \
  -v "$FP4_JOB/base:/model:ro" \
  -v "$FP4_JOB/runtime/base/runtime-config.json:/model/config.json:ro" \
  -v "$FP4_JOB/runtime:/results" \
  "$FP4_RUNTIME_ID" /check.py --image-id "$FP4_RUNTIME_ID" \
  --output /results/rope-qualification.json
python3 qualify.py --root "$FP4_JOB" --model "$FP4_JOB/base" --label base --long
python3 qualify.py --root "$FP4_JOB" --model "$FP4_OUTPUT" --label candidate --long
```

These sequential runs require the GPU to be free. They use a localhost-only
server, TP1, FP8 KV, MTP with four draft tokens, disk-backed PLE, and a separate
packing cache per checkpoint. The native checkpoint config is unchanged;
the default qualification overlays factor-4 YaRN at 1,048,576 tokens.
`--context 262144` selects native context. A server accepting a context setting
does not establish long-context quality: `--long` tokenizes and submits the
actual near-limit retrieval prompt and checks the result.

Compiler artifacts in `cache/b12x/compile` and `cache/triton` may be copied
between runs using the same runtime image and GPU to avoid recompilation.
Do not copy `b12x/loader` or other checkpoint/packing caches. Record any compiler
cache seeding when comparing startup times.

The harness records generated text, code execution results, tool calls,
vision, prompt/generation timing, speculative decoding counters and host UMA
memory/swap samples. Korean prose, translation and fiction also require
semantic review. It stops its own server when finished and leaves logs and
the stopped container for inspection. Archive/rename a previous qualification
container and results before retrying with the same label.

Build integrity and runtime qualification are separate results. Consult
`transfer-manifest.json`, `source-audit.json`, and the completed runtime reports
before describing the checkpoint as qualified. Runtime success does not
establish broad benchmark parity or refusal-rate equivalence to the donor.
