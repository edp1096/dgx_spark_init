# GLM 5.3 Flash FP4-Local TensorFold trial

Archived experiment, not a production Compose recipe. No Talk GLM profile was changed by this trial.

Two DGX Sparks ran the Huihui FP4-Local checkpoint with a 1,048,576-token window, native adaptive MTP (up to three drafts), BF16 KV and one active request. The checkpoint's 42 shards passed full SHA256 verification after transfer.

Korean explanation TG was 36.9 t/s with thinking off and 39.7 t/s with thinking enabled (low requested). Interval-merge code TG was 47.1/48.5 t/s. The unchanged strict synthetic suite passed 7/10 cases in each mode; failures include output-format violations and an incorrect interpretation of the uniqueness task. These small samples do not establish general model quality.

JSON-schema arithmetic and conversation recall passed. The image digits were correctly recognized as `5829`, but returned in an object instead of the requested JSON string. `qualification.json` preserves the raw strict result and separately identifies the correct image content.

Full 1M input retrieval and MTP/plain-token exactness were **not completed**. A configured 1M window must not be read as a 1M qualification. The runtime admitted no default prompt cache: this implementation copies the entire 13.08 GiB state arena for each prefix snapshot, exceeding the remaining memory observed in the trial.

The executable used TensorFold's native CUDA kernels. The SGLang image supplied CUDA/NCCL libraries; its serving engine and B12X kernels were not combined into TensorFold. Both nodes used the same executable, base image and grammar-library hashes. Their separately built Docker images had different build metadata.

The local patch exchanges the root's NCCL unique ID over the existing startup connection so NCCL honors the selected interface. It changes no model math or CUDA kernels. Five bootstrap/liveness unit tests passed. The source pin and hashes are in `build-proof.json`; the patch is retained here. The prototype is not approved for production rollout.

Source: [TensorFold PR 1050](https://github.com/ashhart/TensorFold/pull/1050), source revision `c595d643019b5590c97746463a08fbbb18844f4b`, [NVIDIA checkpoint](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local), and the local synthetic results and launch manifest in this directory. The parent comparison record includes the matched NVFP4/SGLang control and the vLLM candidate.
