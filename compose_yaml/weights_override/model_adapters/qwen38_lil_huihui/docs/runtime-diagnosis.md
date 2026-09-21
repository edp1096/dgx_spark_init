# Qualified and deployed: Huihui/LIL TP1

Talk's `flash-next` TP1 bundle now uses the local `edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD` checkpoint and `dgx-sglang-qwen38-qad:sm121-v4`. Context and allocated KV are both 1,048,576 tokens, MTP is disabled, and Qwen/Flux/ASR/TTS run together. TP2's saved bundle was compared before/after and is unchanged. No diagnostic hooks remain.

## Root cause and fix

The initial candidate returned repeated token 0 (`!`) during a long request. Fresh original LIL and candidate short tests both passed, and additional synchronized diagnostics sometimes hid the failure. Passive operand capture eventually found NaN routing weights with finite model input and shared-expert output.

SGLang's Triton router loaded `bias_ptr` **before** `gdc_wait()`. Qwen passes a freshly produced zero bias; with programmatic dependent launch (PDL), that read could observe the allocation before its producer finished. The resulting NaNs propagated through softmax/top-k and the expert outputs. This was an execution-order defect, not a required change to the converted weights.

`qad/patch_router_pdl.py` moves the bias load after the producer wait. PDL, quantization, routing arithmetic and checkpoint bytes are preserved. The patch is installed only in the QAD TP1 Docker target; the legacy/TP2 target is unchanged. Talk's embedded build assets include the same patch.

## Evidence

| Check | Result |
| --- | --- |
| Original LIL and candidate, identical fresh-process smoke and repeated retrieval | Both passed |
| Captured GEMM/router input, PDL enabled vs disabled | NaNs reproduced only with early dependent launch |
| Controlled delayed bias producer, 20 launches | Old ordering: 15,360 NaN rows; fixed ordering: zero |
| `qad/test_router_ordering.py` on v3 / v4 | v3 fails; v4 passes |
| Fixed GPU regression using captured input | 20 delayed-producer cases, 150 eager executions, 50 graph replays; Torch reference agreement |
| Final 130,925-token retrieval | Passed |
| Final 1,048,435-token retrieval, no diagnostic hooks | `MAPLE,COMET`, 1,107.23 seconds |
| Fresh image generation/edit/object removal plus ASR/TTS | Nine early overlapping requests passed |
| Flux restart at 802,816 tokens, cold edit plus ASR/TTS at 851,968 | Three late overlapping requests passed |
| Minimum available unified memory during final test | 5.229 GiB; no memory guard or OOM |
| Post-1M arithmetic, JSON, sorting, vision, retrieval and tool call | Passed |
| Talk's actual model preparation API | Full checkpoint and metadata hash validation completed |
| Talk bundle start and final live request | Complete; correct model/1M KV; arithmetic `3973` |

Detailed evidence: `router-ordering-regression.json`, `final-qualification.json`, `deployment.json`. The checkpoint's `runtime-qualification.json` binds the final transfer manifest hash to the tested image and configuration. The prior Huginnfork checkpoint/container is retained for rollback. No checkpoint was published remotely.

## Scope of checks

Related QAD/Qwen, TP2 preservation and embedded Qwen build-asset tests passed. Server/config tests and the new local-variant preparation API scope regression passed. The broader orchestrator suite still reports pre-existing unrelated expectations for DS4 collector placement, cluster TTS placement, and GLM/DS4 packaged-source drift; these were not changed as part of this TP1 fix. A pre-existing root README whitespace change was also left intact.

Checkpoint construction details and requantization caveats remain in the adapter README and conversion manifest. Disk size is 98.571 GiB versus 125.910 GiB for the previous Radix derivative (27.340 GiB / 21.714% smaller); this is a checkpoint-size comparison, not a measured runtime-memory saving or a claim that retained backups were deleted.
