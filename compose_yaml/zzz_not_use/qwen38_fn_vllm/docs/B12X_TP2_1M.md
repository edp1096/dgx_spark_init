# B12X TP2 / YaRN 1M experiment (2026-09-14)

Separate from the qualified SGLang BF16 KV / Talk model set. API port 8013.
No original checkpoint weights or config are modified. Actual near-1M retrieval
passed, but this tested configuration is slower than the qualified SGLang lane;
it has not been adopted into Talk or the production Compose configuration.

Pinned image: `eugr/spark-vllm-b12x@sha256:8e7e062186f841453ef0ec6f713043c5b65447decc3835206685128c18e42262`.
Image versions: vLLM `0.1.dev20759+gb40673cd0.d20260913`, B12X 1.3.0
(source `9043b448622764a598969518d413b3fd8b3c0c07`), Torch 2.13.0+cu130.

Configuration: original dealignai NVFP4 checkpoint, native two-node TP2,
B12X linear/MoE/GDN/QSA, FP8 KV storage, disk-backed PLE, MTP 3,
YaRN factor 4, original context 262144, maximum context 1048576.
Preserves M-RoPE sections 11/11/10, interleaving, theta, and partial rotary factor.
Runtime target MXFP8 and MTP NVFP4 LM-head quantization are explicitly disabled;
the image enables these by default. Original checkpoint quantization remains.
B12X's selected QSA path converts FP8 KV to BF16 for its matrix calculations.

Safety: each container 104 GiB RAM+swap limit, no additional swap allowance;
watchdog stops at host MemAvailable <8 GiB or cgroup usage >98 GiB.
GPU fraction 0.70, one concurrent request, prefill chunks 1024. No kernel,
clock/governor, driver, or production Talk configuration changes.

Startup issues found and addressed in the isolated experiment:

- Docker default seccomp denies io_uring used by the B12X O_DIRECT loader.
  The experimental containers use seccomp=unconfined. This is a container
  permission change, not a kernel tuning or model modification.
- Dictionary `--hf-overrides` deliberately do not propagate to MTP in this
  fork. Target initialized with 1M but MTP remained 256K and graph capture
  failed with `QSA sequence length 1048576 exceeds ... 262144`.
  Both now read an ephemeral config directory whose weights/tokenizer files
  are symlinks to the read-only checkpoint.
- Qwen4Exp's outer config forwards text YaRN fields to Transformers before
  exposing max_position_embeddings. A guarded one-line container-local patch
  initializes that attribute from text_config before calling the base class.

Run from the project root, with other model services stopped:

```sh
python3 compose_yaml/qwen38_fn_vllm/b12x_tp2/manage.py start
python3 compose_yaml/qwen38_fn_vllm/b12x_tp2/run_probe.py
python3 compose_yaml/qwen38_fn_vllm/b12x_tp2/manage.py stop
```

The manager verifies identical images and available memory on both nodes.
The probe checks API readiness, short arithmetic, then sends exactly 1046528
input tokens with registry values at 10%, 50%, and 90%. It requires exact usage
accounting and all three values, so KV capacity alone cannot count as success.
This fixture is a limited retrieval check, not a general long-context quality
benchmark. Output speed on a short JSON is not a general decoding benchmark.
Results and guard logs are under `../b12x_tp2/results/<run-token>/`.

The first fully initialized run (`b12x-20260914-135746`) passed short arithmetic,
allocated 5,251,509 FP8 KV token slots and captured target/MTP CUDA graphs.
Its actual 1046528-token request was interrupted at 159.65 seconds after
suspected lack of progress. Async-off run `b12x-20260914-140439` was interrupted
at 119.60 seconds for the same suspicion. These are **inconclusive interrupted
runs**, not proof of a deadlock or failed 1M support. GPU synchronization stacks
alone do not establish a deadlock.

Correction during the third run (`b12x-20260914-141047`): request metrics retain
prompt_tokens_total=25 from the short request while the long prefill proceeds.
KV pool usage rose from 0.9% to 4.6% and worker stacks changed, proving progress
rather than a fixed stall. Async scheduling and projection overlap remain off
in that run, but no causal improvement from either flag has been established.

Outcome: third run **passed**. Exactly 1,046,528 input tokens, three of three
registry values recovered at 10%/50%/90%, 56 output tokens, no prefix-cache hit.
TTFT 1315.1148 s (21m55s), input/TTFT 795.769 tok/s, short JSON output estimate
59.424 tok/s. The same fixture on SGLang BF16 was TTFT 1098.8740 s (18m19s),
952.364 tok/s and 58.055 tok/s respectively. B12X TTFT was 19.68% longer,
PP 16.44% lower. This compares whole configurations (different engines, KV,
schedulers and kernels), not an isolated FP8-KV or B12X-kernel speed experiment.
Only one completed request per configuration; no broad quality/performance claim.

Minimum host MemAvailable: head 22.15 GiB, worker 26.39 GiB; no observed OOM,
watchdog trip or boot-id change. Both experimental model containers were stopped
after evidence collection. Cgroup usage does not account for all GPU/shared
allocations, so host MemAvailable is the relevant whole-system margin here.

Do not interpret a zero PP interval, GPU event wait, or unchanged completed-request
token counters as a hang. Final server log reported 104645 tok/s for its last
10-second reporting interval because it attributed the entire prompt at completion;
that is not actual prefill throughput. Use the client TTFT-based value above.

Machine-readable comparison: `b12x-tp2-1m-result.json`; raw result:
`../b12x_tp2/results/b12x-20260914-141047/retrieval.json`.

SGLang B12X correction (user supplied PR #29190): a real NVFP4 MoE
integration existed; saying SGLang had no B12X implementation was too broad.
GitHub API confirms closed 2026-09-13, merged=false. The author force-pushed
the branch, making the current diff empty; historical commit 6be5c56 still
contains `moe_runner/flashinfer_cutedsl_sm120.py`. It calls
`b12x.integration.tp_moe.b12x_moe_fp4`, `quant_mode="nvfp4"`, ModelOpt source,
with static activation scales. It does not implement QSA/KV or YaRN.
The installed qualified SGLang layer/server-arg sources contain no B12X runner.
Completion and W4A16 integration issues remain open at inspection time:

- https://github.com/sgl-project/sglang/pull/29190
- https://github.com/sgl-project/sglang/issues/33709
- https://github.com/sgl-project/sglang/issues/33710
- https://github.com/hsr1234563/sglang/blob/6be5c56/python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl_sm120.py

The reason for this vLLM experiment is availability of its complete Qwen QSA,
GDN and other B12X adapters, not a claim that SGLang cannot use B12X.

Further corrections supplied by the user:

1. `lukealonso/sglang-cuda13-b12x` is a real public SGLang+B12X image.
   Registry config confirms latest was built 2026-05-12 with B12X 0.13.6.
   The latest/w4a16/noloop tags inspected are amd64, but that restriction must
   not be generalized to all SGLang+B12X images. Extracted source includes
   B12X MoE, linear, paged attention, NSA and other integration, so this is
   broader than PR #29190. Its 1952 Python files have no Qwen3.8/Qwen4Exp
   references; that specific old image is not a current Qwen3.8 recipe.
2. The official DeepSeek V4 cookbook explicitly documents Spark TP2 B12X:
   `lmsysorg/sglang:dev-v4f-2dgx-v2`, branch b12x-vision @452239a74f,
   with SM12x MoE W4A8 and compressed MLA, vision fixes and GB10 pins.
   Env includes `SGLANG_SM120_FLASHMLA_BACKEND=b12x` and
   `SGLANG_B12X_MAX_TOKENS` matching chunked-prefill-size. Its NVFP4 target
   uses flashinfer_cutlass MoE while MXFP4 DSpark uses B12X.
   Thus SGLang+B12X+Spark TP2 already exists. The remaining model-specific
   question is Qwen3.8 QSA/GDN/FP8 KV/YaRN 1M, not generic B12X availability.

Sources:
- https://hub.docker.com/r/lukealonso/sglang-cuda13-b12x/tags
- https://raw.githubusercontent.com/sgl-project/sglang/0d95a9c1ff14773b722009a6c6fd6ad66c7ce395/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx (lines 219–227)
