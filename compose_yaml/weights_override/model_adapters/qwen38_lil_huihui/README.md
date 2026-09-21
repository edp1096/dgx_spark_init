# Huihui GGUF delta → LIL QAD mixed NVFP4/MXFP8

This separate adapter follows the existing `qwen38_huihui` full GGUF audit.
It does **not** run the RadixArk BF16 transfer on an incompatible checkpoint.

Pinned inputs:

- LIL `local-inference-lab/Qwen3.8-Flash-Next-NVFP4`, `7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd`
- Huihui GGUF `7e3bfc316b880fefeb049596f11c49d6a18e05fb`
- Original Unsloth GGUF `38bb39ee97821de2c9009abb7e93950eec396e66`
- Existing 1,558-tensor audit: `../qwen38_huihui/docs/audit.json`
- NVIDIA ModelOpt vendor commit `87c9f8cf83021957d1a1a575c90c9a4eaaf7ef0c`

The 101 changed Q8 tensors comprise 96 output/shared projections plus routed
expert down projections in five layers. Original Q8 ranges are downloaded with
HTTP 206 range and SHA256 checks against the earlier full audit. The full Huihui
checkpoint is downloaded again. Each changed donor range is also hash checked.
No MiaAI code is used.

LIL's high-precision QAD training weights are unavailable. The candidate uses:

`DQ(LIL checkpoint) + DQ(Huihui Q8) - DQ(Unsloth Q8)`

The result is rounded to BF16 and quantized with ModelOpt back into the original
MXFP8 or NVFP4 tensor layout. GDN output columns are restored from GGUF to HF order.
This retains LIL's reconstructed QAD base instead of substituting teacher weights,
but adds requantization error and does not guarantee QAD benchmark preservation.
It is not equivalent to an unavailable Huihui BF16 checkpoint.

All non-target bytes, activation scales, PLE, vision, MTP, tokenizer and configuration
are preserved. No fresh activation calibration is performed. Candidate shards are
independent copies. Output/partial collisions are rejected. The entire source is
hashed before/after, every non-target tensor compared, headers/index checked and
all output shards hashed before promotion from `.partial` to the candidate path.
Runtime qualification and promotion to Talk are separate steps.

`test_numeric.py` compares both checkpoint decoders with ModelOpt's decoder and
checks the inverse GDN permutation. Run in the existing CPU-only helper with
this root and vendored ModelOpt on PYTHONPATH, plus the existing local gguf package.
`build.py` requires `--base`, `--donor`, `--original` (verified Q8 range directory),
`--audit`, and `--output`. No source or existing output is overwritten.

Before runtime qualification, run `refine_scales.py` with the same `--base`,
`--original`, `--donor`, `--audit`, and `--candidate` set to the unqualified build.
It compares refreshed versus original LIL NVFP4 scales for every changed expert
matrix and retains the smaller squared reconstruction error. MXFP8 stays unchanged.
It requires a verified independent candidate with no runtime qualification, marks
it `refining` during mutation, and verifies all bytes outside the permitted expert
ranges before marking it verified again. The initial manifest is retained, and
final shard hashes/numerical metrics replace the initial values. This local error
minimization is not an accuracy or abliteration-quality benchmark.

For the two-stage workflow, supply Q8 **range directories** for both `--original`
and `--donor`. `fetch_delta.py --side original` and `--side huihui` create them
using the pinned audit; `--workers` controls independent tensor downloads.
The initial builder also accepts a full donor GGUF snapshot, but the refinement
step expects the verified donor range directory. Do not refine a model being served.
