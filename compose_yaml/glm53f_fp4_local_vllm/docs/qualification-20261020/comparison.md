# GLM 5.3 Flash FP4-Local comparison

Status: FP4-Local vLLM/B12X MTP 5 selected after matched comparison and full 1M input qualification. Final deployment lifecycle checks are recorded separately.

Measured on two DGX Sparks on 2026-10-20. This compares complete checkpoints and serving stacks, including their native templates, KV formats and draft methods. It does not isolate quantization or engine effects.

Synthetic cases use temperature 0. Thinking-off and thinking-on (low requested) are separate. Two repeated Korean explanation and interval-merge tasks supply the TG medians; these are small samples, not a general quality benchmark.

| Stack | Thinking | Korean TG (t/s) | Code TG (t/s) | Strict checks |
|---|---|---:|---:|---:|
| sgl-nvfp4 | off | 22.1 | 47.4 | 8/10 |
| sgl-nvfp4 | on-low | 25.9 | 50.1 | 7/10 |
| tf-fp4 | off | 36.9 | 47.1 | 7/10 |
| tf-fp4 | on-low | 39.7 | 48.5 | 7/10 |
| vllm-fp4 | off | 34.9 | 36.1 | 8/10 |
| vllm-fp4 | on-low | 34.9 | 37.7 | 7/10 |
| vllm-fp4-mtp5 | off | 30.0 | 48.4 | 8/10 |
| vllm-fp4-mtp5 | on-low | 31.5 | 49.3 | 8/10 |

TensorFold TG uses its backend decode timer. Other stacks use the first-to-last visible SSE span and completion counts. Tool-call TG is excluded because buffered tool output can make a client-only rate invalid. Raw client rates are retained for diagnosis.

PP is an approximation: prompt tokens divided by the complete one-token request duration. This also handles a hidden first thinking token. It includes request/template overhead and one decode step; it is not a kernel-only prefill measurement. SGLang input shapes were warmed with a different prefix before measurement. Native TensorFold uses prebuilt CUDA kernels.

| Stack | Thinking | Prompt tokens | PP approximation (t/s) |
|---|---|---:|---:|
| sgl-nvfp4 | off | 5182 | 1285 |
| sgl-nvfp4 | off | 20540 | 1900 |
| sgl-nvfp4 | off | 81977 | 1929 |
| sgl-nvfp4 | on-low | 5176 | 1731 |
| sgl-nvfp4 | on-low | 20543 | 1895 |
| sgl-nvfp4 | on-low | 81979 | 1938 |
| tf-fp4 | off | 5182 | 1282 |
| tf-fp4 | off | 20540 | 1365 |
| tf-fp4 | off | 81977 | 1369 |
| tf-fp4 | on-low | 5176 | 1279 |
| tf-fp4 | on-low | 20543 | 1362 |
| tf-fp4 | on-low | 81979 | 1370 |
| vllm-fp4 | off | 5182 | 1975 |
| vllm-fp4 | off | 20540 | 2154 |
| vllm-fp4 | off | 81977 | 2189 |
| vllm-fp4 | on-low | 5176 | 1969 |
| vllm-fp4 | on-low | 20543 | 2155 |
| vllm-fp4 | on-low | 81979 | 2194 |
| vllm-fp4-mtp5 | off | 81977 | 2203 |

The NVFP4 control uses the production SGLang image, DFlash2 with 5 draft tokens, FP8 KV, one request and a 1M context. Its production Compose permits swap; earlier isolated trials with a no-swap container limit or missing launch settings are excluded. FP4-Local TensorFold uses the pinned PR 1050 source, native adaptive MTP (up to 3 drafts), BF16 KV and a 1M window. vLLM FP4-Local uses the pinned B12X image with MTP 1 or MTP 5 and FP8 KV. MTP 5 is the balanced default: it preserves code TG while improving Korean TG and prefill relative to the NVFP4 control. MTP 1 favors Korean TG but slows this code workload.

TensorFold used a local NCCL root-ID exchange fix; CUDA kernels and model math were unchanged. Both nodes used the same executable, base image and grammar library. Separate Docker build metadata produced different image IDs.

The TensorFold GLM prefix snapshot currently copies the full 13.08 GiB state arena at a 1M window. No prompt cache was admitted by its default memory budget in this trial. A configured 1M window alone does not prove successful 1M input processing.

Sources: [NVIDIA FP4-Local checkpoint](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local), [TensorFold FP4-Local PR](https://github.com/ashhart/TensorFold/pull/1050), [NCCL bootstrap source](https://github.com/NVIDIA/nccl/blob/master/src/bootstrap.cc). Local raw results, launch manifests and memory samples accompany the experiment record.


## Selected profile qualification

The tested checkpoint is Huihui abliterated FP4-Local. The original NVIDIA checkpoint is selectable and revision-pinned, but was not separately benchmarked here.

MTP 5 accepted an actual 1,047,005-token prompt and returned all five exact codes placed at 2%, 25%, 50%, 75% and 98% of the input. Total request time was 628.53 seconds. This verifies full input processing and this retrieval task; it does not establish general long-context quality.

The vLLM MTP 5 small synthetic suite passed 16/20 strict checks versus 15/20 for NVFP4. Repeated performance cases are included in that count; there are eight distinct tasks. These samples do not establish broad model superiority.

Native TensorFold was not selected: its approximately 1,370 t/s long-prompt processing was slower than vLLM at approximately 2,203 t/s, and its default 1M prefix snapshots did not fit the available cache budget. Its faster Korean TG does not offset those limits for this deployment.

The selected production Compose allows CPU swap during loading (112 GiB memory limit, 128 GiB total memory+swap limit); the KV pool remains on GPU. Independent launcher and Talk lifecycle results, including cold-start memory measurements, accompany the final deployment record.

## Final launcher checks

The standalone recipe was copied outside the checkout and started both ranks successfully. Full first-cache startup took 602 seconds, including 4,494 kernel measurements and 1,401 compilations. CPU swap was permitted. Minimum host-available RAM was 1.93 GiB on head and 2.50 GiB on worker; head swap-out was 2,279 MiB. These are measured samples, not guaranteed minimum reserves.

The final profile passed forced tool calls with thinking off/on, ordinary thinking off/on, native MP4 video OCR, image OCR, strict JSON schema and multi-turn recall. Forced tool selection can return a correct tool call without reasoning text even when thinking is enabled; ordinary thinking was checked separately.

A 17,365-token repeated prompt took 8.73 seconds cold and 0.335 seconds on repeat. The backend explicitly reported all 17,365 tokens cached. One API call returning correct content is not used as proof of broad model quality.

Talk deployment lifecycle and worker TTS results are recorded in the completed records below. The original NVFP4 checkpoint remains on worker as a rollback source; only the temporary head comparison copy was removed.

## Completed Talk deployment

The complete managed bundle started successfully. GLM API readiness took 311.70 seconds from the Talk request; the complete bundle took 1449.00 seconds because the worker built its missing Qwen3-TTS image for the first time. That image is now retained. Automatic tool choice and a second turn consuming the tool result both passed. Worker CUDA TTS produced 4.56 seconds of 24 kHz WAV audio in 1.16 seconds.

Talk stopped both GLM ranks and worker TTS successfully. Both GLM ranks exited with code 0 and OOMKilled=false. The prior Qwen FP4-Local selection was restored with all model containers stopped and automatic startup disabled. Shared Extra services retained their previous running state.

The cached Talk restart sampled head RAM as low as 0.773 GiB for a short interval (less than 1.5 GiB for approximately 1.0 seconds), then recovered; worker minimum was 4.504 GiB. No OOM or memory-guard termination occurred. These measurements document the tight checkpoint conversion peak; they are not a promise of sufficient headroom under unrelated additional workloads. CPU swap remains enabled during startup.

Final source, packaging, unit/integration and model-list browser checks passed. Full details are in deployment-summary.json, final-audit.json and the accompanying lifecycle records.

Verification commands: `go test -p 1 ./...`, `go run ./cmd/package-recipes --check`, `go run ./cmd/package-support --check`, both first-install and recovery Python unittest suites, `npm test`, `npm run build`, and `npx playwright test e2e/exl3-quantization-sets.spec.js` (from the corresponding Talk/web directories).
