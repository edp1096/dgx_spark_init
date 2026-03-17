"""Pipeline lifecycle manager for ZIT — Z-Image-Turbo + ControlNet Union.

Single model family on GPU. ZIT + ControlNet co-resident (128GB sufficient).
No family switching needed.
"""

import gc
import logging
import os
from pathlib import Path

import torch

from zit_config import (
    CONTROLNET_CONFIG,
    CONTROLNET_DIR,
    CONTROLNET_FILENAME,
    LORAS_DIR,
    MODEL_DIR as DEFAULT_MODEL_DIR,
    OUTPUT_DIR,
    PREPROCESSORS_DIR,
    ZIMAGE_TURBO_DIR,
)

logger = logging.getLogger("zit-ui")

os.makedirs(str(OUTPUT_DIR), exist_ok=True)
os.makedirs(str(DEFAULT_MODEL_DIR), exist_ok=True)


# ---------------------------------------------------------------------------
# q8_kernels FP8 — only need FP8Linear class for weight container
# ---------------------------------------------------------------------------
_Q8_AVAILABLE = False
try:
    from q8_kernels.modules.linear import FP8Linear as _Q8FP8Linear
    _Q8_AVAILABLE = True
except ImportError:
    pass


def _scaled_fp8_forward(self, x, x_scales=None, out_dtype=None):
    """Corrected FP8 forward: per-tensor dynamic scaling via torch._scaled_mm.

    q8_kernels' FP8LinearFunc has two bugs:
      1. Applies Hadamard rotation without inverse/scale compensation (cos_sim ≈ 0)
      2. fp8_gemm kernel overflows to inf on certain weight rows (accumulation bug)

    Uses torch._scaled_mm which handles scaling and accumulation correctly.
    """
    if x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        x = x.to(torch.bfloat16)

    # Always output BF16 — FP8 output is not useful for downstream ops
    out_dtype = torch.bfloat16

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])

    # Per-tensor dynamic quantization
    x_abs_max = x_2d.abs().amax().clamp(min=1e-12)
    scale_a = (x_abs_max / 448.0).float()  # must be float32 for _scaled_mm

    x_fp8 = (x_2d / scale_a).to(torch.float8_e4m3fn)

    # Cache transposed weight to avoid repeated .t().contiguous()
    if not hasattr(self, '_weight_t'):
        self._weight_t = self.weight.data.t().contiguous()

    # Use stored weight_scale if available (normalized FP8 quantization),
    # otherwise 1.0 (simple BF16→FP8 cast)
    if hasattr(self, '_weight_scale'):
        scale_b = self._weight_scale
    else:
        scale_b = torch.tensor(1.0, device=x.device, dtype=torch.float32)
    result = torch._scaled_mm(
        x_fp8, self._weight_t,
        scale_a, scale_b,
        out_dtype=out_dtype, use_fast_accum=True,
    )
    if isinstance(result, tuple):
        result = result[0]

    if self.bias is not None:
        bias = self.bias.to(result.dtype) if self.bias.dtype != result.dtype else self.bias
        result = result + bias

    return result.reshape(orig_shape[:-1] + (self.weight.shape[0],))


def _load_fp8_weight_scales(filepath):
    """Load weight_scale values from an FP8 checkpoint (if present).

    Returns a dict mapping layer name → weight_scale tensor, e.g.
    {"double_blocks.0.img_attn.proj": tensor(0.0006)}.
    """
    from safetensors.torch import load_file
    sd = load_file(filepath, device="cpu")
    scales = {}
    for key in sd:
        if key.endswith(".weight_scale"):
            layer_name = key[: -len(".weight_scale")]
            scales[layer_name] = sd[key]
    del sd
    return scales


def _patch_transformer_q8(model, weight_scales=None):
    """Replace FP8 nn.Linear layers with q8_kernels FP8Linear for native FP8 GEMM.

    Uses a corrected forward (per-tensor dynamic scaling, no Hadamard rotation)
    instead of q8_kernels' broken default which applies Hadamard without inverse.
    After patching, casts ALL non-FP8Linear floating-point params to BF16 so that
    every dtype matches fp8_mm's BF16 output (index_put, add, etc. require matching).

    Args:
        weight_scales: optional dict of layer_name → weight_scale tensor, for
            FP8 checkpoints that use normalized quantization (weight/scale → FP8).
    """
    import types

    if weight_scales is None:
        weight_scales = {}

    count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear) or isinstance(module, _Q8FP8Linear):
            continue
        if module.weight.dtype != torch.float8_e4m3fn:
            continue
        fp8 = _Q8FP8Linear(
            module.in_features, module.out_features,
            bias=module.bias is not None, device=module.weight.device,
        )
        fp8.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            fp8.bias = torch.nn.Parameter(
                module.bias.data.to(torch.bfloat16), requires_grad=False,
            )
        # Store weight_scale for normalized FP8 quantization
        if name in weight_scales:
            fp8._weight_scale = weight_scales[name].to(
                device=module.weight.device, dtype=torch.float32,
            )
        # Replace broken default forward with corrected scaled version
        fp8.forward = types.MethodType(_scaled_fp8_forward, fp8)
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], fp8)
        count += 1

    # Cast ALL non-FP8Linear-weight floating-point params to BF16.
    # fp8_mm returns BF16, so everything must be BF16 for consistency
    # (index_put, element-wise ops, etc. require matching dtypes).
    # FP8Linear weights must stay FP8 for GEMM.
    fp8_linear_names = {n for n, m in model.named_modules() if isinstance(m, _Q8FP8Linear)}
    cast_count = 0
    for name, param in model.named_parameters():
        if any(name.startswith(p + ".") for p in fp8_linear_names) and name.endswith(".weight"):
            continue
        if param.is_floating_point() and param.dtype != torch.bfloat16:
            param.data = param.data.to(torch.bfloat16)
            cast_count += 1

    if count > 0 or cast_count > 0:
        logger.info("q8_kernels: %d Linear → FP8Linear, %d params cast to BF16", count, cast_count)


# ---------------------------------------------------------------------------
# Required models per generation type
# ---------------------------------------------------------------------------
REQUIRED_MODELS = {
    "zit_t2i": [ZIMAGE_TURBO_DIR],
    "controlnet": [ZIMAGE_TURBO_DIR, f"{CONTROLNET_DIR}/{CONTROLNET_FILENAME}"],
    "inpaint": [ZIMAGE_TURBO_DIR, f"{CONTROLNET_DIR}/{CONTROLNET_FILENAME}"],
    "outpaint": [ZIMAGE_TURBO_DIR, f"{CONTROLNET_DIR}/{CONTROLNET_FILENAME}"],
}


def scan_lora_files() -> list[str]:
    """Scan LoRA directory for .safetensors files."""
    loras_dir = Path(DEFAULT_MODEL_DIR) / LORAS_DIR
    if not loras_dir.exists():
        loras_dir.mkdir(parents=True, exist_ok=True)
        return []
    return sorted(f.name for f in loras_dir.glob("*.safetensors"))


class PipelineManager:
    """Manages ZIT + ControlNet pipeline — co-resident on GPU."""

    def __init__(self, progress_queue=None) -> None:
        # ZIT components (always resident once loaded)
        self.zit_components: dict | None = None

        # ControlNet (co-resident with ZIT)
        self.controlnet_loaded: bool = False
        self._need_controlnet: bool = True  # whether current load includes CN adapter

        # LoRA state
        self._current_lora: str | None = None

        # Precision state
        self.use_fp8: bool = True  # default: FP8 if available
        self._loaded_fp8: bool | None = None  # track what was actually loaded

        # Preprocessors (lazy-loaded)
        self._preprocessors: dict = {}

        # State
        self.model_dir: str = str(DEFAULT_MODEL_DIR)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_backend: str = "native"

        # Progress tracking
        self._is_generating: bool = False
        self._current_task_id: str | None = None
        self.progress_queue = progress_queue

    # -------------------------------------------------------------------
    # Progress IPC
    # -------------------------------------------------------------------
    def _send_progress(self, msg_type: str, data: dict):
        if self.progress_queue is not None and self._current_task_id is not None:
            try:
                self.progress_queue.put_nowait({
                    "task_id": self._current_task_id,
                    "type": msg_type,
                    "data": data,
                })
            except Exception:
                pass

    # -------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------
    def _gpu_cleanup(self):
        for _ in range(3):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @staticmethod
    def _release_module(module):
        """Move module to meta device to immediately release CUDA tensor storage."""
        if isinstance(module, torch.nn.Module):
            try:
                module.to("meta")
            except Exception:
                pass

    def cleanup_all(self):
        if self.zit_components is not None:
            logger.info("Cleaning up ZIT components")
            for key, val in list(self.zit_components.items()):
                self._release_module(val)
            self.zit_components = None
            self.controlnet_loaded = False
            self._gpu_cleanup()

    # -------------------------------------------------------------------
    # Model path helpers
    # -------------------------------------------------------------------
    def _model_path(self, subpath: str) -> str:
        return str(Path(self.model_dir) / subpath)

    def check_models(self, pipeline_type: str) -> list[str]:
        required = REQUIRED_MODELS.get(pipeline_type, [])
        missing = []
        for name in required:
            path = Path(self.model_dir) / name
            if path.is_dir():
                if not path.exists() or not any(path.rglob("*.safetensors")):
                    missing.append(name)
            else:
                if not path.exists():
                    missing.append(name)
        return missing

    # -------------------------------------------------------------------
    # ZIT loading (VideoX-Fun model classes)
    # -------------------------------------------------------------------
    def load_zit(self, use_fp8: bool | None = None, need_controlnet: bool = True):
        """Load Z-Image-Turbo pipeline, optionally with ControlNet adapter.

        Uses VideoX-Fun's ZImageControlTransformer2DModel which handles both
        pure T2I (with zero control_context) and ControlNet (with control_image).
        FP8 via q8_kernels patch (same approach as zifk).

        Args:
            need_controlnet: If False, skip loading ControlNet adapter weights
                into the transformer. This improves LoRA face quality for pure T2I.
        """
        if use_fp8 is not None:
            self.use_fp8 = use_fp8

        # If already loaded, check if we need to reload transformer
        if self.zit_components is not None:
            need_reload = False
            if self._loaded_fp8 is not None and self._loaded_fp8 != self.use_fp8:
                logger.info("Precision changed (%s → %s), reloading transformer...",
                            "FP8" if self._loaded_fp8 else "BF16",
                            "FP8" if self.use_fp8 else "BF16")
                need_reload = True
            if self._need_controlnet != need_controlnet:
                logger.info("ControlNet mode changed (%s → %s), reloading transformer...",
                            "with CN" if self._need_controlnet else "without CN",
                            "with CN" if need_controlnet else "without CN")
                need_reload = True
                self._need_controlnet = need_controlnet
            if need_reload:
                self._reload_transformer()
            return self.zit_components

        self._need_controlnet = need_controlnet

        from diffusers import FlowMatchEulerDiscreteScheduler
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from videox_models.z_image_transformer2d_control import ZImageControlTransformer2DModel

        model_path = Path(self.model_dir) / ZIMAGE_TURBO_DIR
        transformer_dir = model_path / "transformer"

        # --- Step 1: Load transformer ---
        self._send_progress("loading_start", {"name": "ZIT Transformer", "index": 1, "total": 4})
        logger.info("Loading ZIT transformer from %s...", transformer_dir)

        transformer = ZImageControlTransformer2DModel.from_pretrained(
            str(transformer_dir),
            torch_dtype=torch.bfloat16,
            transformer_additional_kwargs=CONTROLNET_CONFIG,
        )
        transformer = transformer.to(self.device)
        transformer.eval()

        # --- Step 1b: Load ControlNet adapter weights on top (skip for pure T2I) ---
        cn_path = Path(self.model_dir) / CONTROLNET_DIR / CONTROLNET_FILENAME
        if self._need_controlnet and cn_path.exists():
            self._send_progress("loading_start", {"name": "ControlNet Adapter", "index": 1, "total": 4})
            logger.info("Loading ControlNet adapter from %s...", cn_path)
            from safetensors.torch import load_file
            cn_state = load_file(str(cn_path), device=str(self.device))
            missing, unexpected = transformer.load_state_dict(cn_state, strict=False)
            del cn_state
            self._gpu_cleanup()
            logger.info("ControlNet adapter loaded (missing=%d, unexpected=%d)", len(missing), len(unexpected))
            self.controlnet_loaded = True
        elif not self._need_controlnet:
            logger.info("Skipping ControlNet adapter (pure T2I mode)")
            self.controlnet_loaded = False
        else:
            logger.warning("ControlNet model not found at %s — T2I only", cn_path)

        # --- Step 1c: Apply FP8 q8_kernels GEMM (if enabled) ---
        self._apply_fp8_if_needed(transformer, model_path)

        # --- Step 2: Load VAE ---
        self._send_progress("loading_start", {"name": "VAE", "index": 2, "total": 4})
        logger.info("Loading VAE...")
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(
            str(model_path / "vae"),
            torch_dtype=torch.float32,
        ).to(self.device)
        vae.eval()

        # --- Step 3: Load text encoder ---
        self._send_progress("loading_start", {"name": "Text Encoder", "index": 3, "total": 4})
        logger.info("Loading text encoder...")
        text_encoder = AutoModelForCausalLM.from_pretrained(
            str(model_path / "text_encoder"),
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        text_encoder.eval()

        # --- Step 4: Load tokenizer + scheduler ---
        self._send_progress("loading_start", {"name": "Tokenizer + Scheduler", "index": 4, "total": 4})
        tokenizer_dir = model_path / "tokenizer"
        if not tokenizer_dir.exists():
            tokenizer_dir = model_path / "text_encoder"
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

        scheduler_dir = model_path / "scheduler"
        if scheduler_dir.exists():
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(scheduler_dir))
        else:
            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000, shift=3.0,
            )

        # --- Build pipeline ---
        from videox_models.pipeline_z_image_control import ZImageControlPipeline
        pipeline = ZImageControlPipeline(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )

        self.zit_components = {
            "pipeline": pipeline,
            "transformer": transformer,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "scheduler": scheduler,
        }

        self._send_progress("loading_done", {"name": "ZIT", "index": 4, "total": 4, "elapsed": 0})
        logger.info("ZIT pipeline ready (ControlNet=%s, precision=%s)",
                     "loaded" if self.controlnet_loaded else "skipped/not found",
                     "FP8" if self._loaded_fp8 else "BF16")
        return self.zit_components

    def _apply_fp8_if_needed(self, transformer, model_path: Path):
        """Apply FP8 q8_kernels patch to transformer if enabled."""
        fp8_file = model_path / "model_fp8.safetensors"
        if self.use_fp8 and fp8_file.exists() and _Q8_AVAILABLE:
            logger.info("Loading FP8 weights + q8_kernels GEMM patch...")
            from safetensors.torch import load_file
            fp8_state = load_file(str(fp8_file), device=str(self.device))
            transformer.load_state_dict(fp8_state, strict=False, assign=True)
            del fp8_state
            self._gpu_cleanup()
            weight_scales = _load_fp8_weight_scales(str(fp8_file))
            _patch_transformer_q8(transformer, weight_scales=weight_scales)
            self._loaded_fp8 = True
            logger.info("FP8 transformer ready")
        else:
            self._loaded_fp8 = False
            if not self.use_fp8:
                logger.info("Using BF16 transformer (FP8 disabled by user)")
            else:
                logger.info("Using BF16 transformer (no FP8 GEMM)")

    def _reload_transformer(self):
        """Reload transformer only (keep VAE/text_encoder/tokenizer).

        ~2x faster than full reload since VAE+text_encoder stay resident.
        """
        from videox_models.z_image_transformer2d_control import ZImageControlTransformer2DModel

        # Unload LoRA hooks first
        self._cleanup_lora_hooks()
        self._current_lora = None

        # Release old transformer
        old_transformer = self.zit_components["transformer"]
        self._release_module(old_transformer)
        del old_transformer
        self._gpu_cleanup()

        model_path = Path(self.model_dir) / ZIMAGE_TURBO_DIR
        transformer_dir = model_path / "transformer"

        self._send_progress("loading_start", {"name": "Transformer reload", "index": 1, "total": 1})
        logger.info("Reloading transformer from %s...", transformer_dir)

        transformer = ZImageControlTransformer2DModel.from_pretrained(
            str(transformer_dir),
            torch_dtype=torch.bfloat16,
            transformer_additional_kwargs=CONTROLNET_CONFIG,
        )
        transformer = transformer.to(self.device)
        transformer.eval()

        # Re-apply ControlNet adapter (only if needed)
        cn_path = Path(self.model_dir) / CONTROLNET_DIR / CONTROLNET_FILENAME
        if self._need_controlnet and cn_path.exists():
            from safetensors.torch import load_file
            cn_state = load_file(str(cn_path), device=str(self.device))
            transformer.load_state_dict(cn_state, strict=False)
            del cn_state
            self._gpu_cleanup()
            self.controlnet_loaded = True
            logger.info("ControlNet adapter re-applied")
        else:
            self.controlnet_loaded = False
            if not self._need_controlnet:
                logger.info("Skipping ControlNet adapter (pure T2I mode)")

        # Apply FP8 if needed
        self._apply_fp8_if_needed(transformer, model_path)

        # Update pipeline and components
        self.zit_components["transformer"] = transformer
        self.zit_components["pipeline"].transformer = transformer

        self._send_progress("loading_done", {"name": "Transformer reload", "index": 1, "total": 1, "elapsed": 0})
        logger.info("Transformer reloaded (precision=%s)", "FP8" if self._loaded_fp8 else "BF16")

    # -------------------------------------------------------------------
    # LoRA loading / unloading (forward-hook based)
    # -------------------------------------------------------------------
    def load_lora(self, lora_name: str, lora_scale: float = 1.0):
        """Load LoRA via forward hooks — works with FP8Linear, no PEFT needed.

        Registers a forward hook on each target module that adds the LoRA
        contribution: output += scale * (input @ A^T @ B^T).
        FP8 weights stay untouched; LoRA A/B are stored in BF16.
        """
        if not lora_name or lora_name == "None":
            self.unload_lora()
            return

        if self.zit_components is None:
            logger.warning("Cannot load LoRA — ZIT pipeline not loaded yet")
            return

        lora_path = Path(self.model_dir) / LORAS_DIR / lora_name
        if not lora_path.exists():
            logger.error("LoRA file not found: %s", lora_path)
            return

        # If same LoRA already loaded, just update scale
        if self._current_lora == lora_name:
            self._lora_scale = lora_scale
            logger.info("LoRA scale updated: %.2f", lora_scale)
            return

        # Unload previous LoRA first
        if self._current_lora is not None:
            self.unload_lora()

        try:
            from safetensors.torch import load_file
            from safetensors import safe_open

            logger.info("Loading LoRA: %s (scale=%.2f)", lora_name, lora_scale)

            # Read metadata for alpha/rank auto-scaling
            with safe_open(str(lora_path), framework="pt") as f:
                meta = f.metadata() or {}
            file_alpha = int(meta["lora_alpha"]) if "lora_alpha" in meta else None
            file_rank = int(meta["rank"]) if "rank" in meta else None
            if file_alpha is not None and file_rank is not None and file_rank > 0:
                alpha_scale = file_alpha / file_rank
                logger.info("LoRA metadata: alpha=%d, rank=%d → alpha_scale=%.4f",
                            file_alpha, file_rank, alpha_scale)
            else:
                alpha_scale = 1.0

            lora_sd = load_file(str(lora_path), device=str(self.device))

            # Parse LoRA state dict: group A/B pairs by module path
            # Keys: "context_refiner.0.attention.to_q.lora_A.default.weight"
            #    or "base_model.model.blocks.0.attn.to_q.lora_A.weight"
            lora_pairs = {}  # module_path -> {"A": tensor, "B": tensor}
            for key, tensor in lora_sd.items():
                if ".lora_A." not in key and ".lora_B." not in key:
                    continue
                if ".lora_A." in key:
                    module_path = key.split(".lora_A.")[0]
                    ab = "A"
                else:
                    module_path = key.split(".lora_B.")[0]
                    ab = "B"
                # Strip base_model.model. prefix
                module_path = module_path.removeprefix("base_model.model.")
                if module_path not in lora_pairs:
                    lora_pairs[module_path] = {}
                lora_pairs[module_path][ab] = tensor.to(torch.bfloat16)

            del lora_sd

            if not lora_pairs:
                logger.warning("No LoRA A/B pairs found in %s", lora_name)
                return

            # Detect rank
            rank = 16
            for pair in lora_pairs.values():
                if "A" in pair:
                    rank = pair["A"].shape[0]
                    break

            logger.info("LoRA: rank=%d, %d module(s)", rank, len(lora_pairs))

            # Register forward hooks on target modules
            transformer = self.zit_components["transformer"]
            self._lora_scale = lora_scale
            self._lora_alpha_scale = alpha_scale
            self._lora_hooks = []
            self._lora_params = []  # keep references to prevent GC
            hook_count = 0

            for module_path, pair in lora_pairs.items():
                if "A" not in pair or "B" not in pair:
                    logger.warning("Incomplete LoRA pair for %s, skipping", module_path)
                    continue

                # Navigate to the target module
                try:
                    target = transformer
                    for part in module_path.split("."):
                        target = getattr(target, part)
                except AttributeError:
                    logger.warning("Module not found: %s, skipping", module_path)
                    continue

                lora_A = pair["A"]  # shape: (rank, in_features)
                lora_B = pair["B"]  # shape: (out_features, rank)
                self._lora_params.extend([lora_A, lora_B])

                # Closure to capture A/B per module
                def _make_hook(A, B):
                    def hook(module, input, output):
                        x = input[0] if isinstance(input, tuple) else input
                        # LoRA: x @ A^T @ B^T * (alpha/rank) * user_scale
                        lora_out = x.to(torch.bfloat16) @ A.t() @ B.t()
                        return output + lora_out * (self._lora_alpha_scale * self._lora_scale)
                    return hook

                handle = target.register_forward_hook(_make_hook(lora_A, lora_B))
                self._lora_hooks.append(handle)
                hook_count += 1

            self._current_lora = lora_name
            logger.info("LoRA loaded: %s (%d hooks, rank=%d)", lora_name, hook_count, rank)

        except Exception as e:
            logger.error("Failed to load LoRA %s: %s", lora_name, e)
            import traceback
            traceback.print_exc()
            # Clean up partial state
            self._cleanup_lora_hooks()

    def unload_lora(self):
        """Remove LoRA forward hooks, restoring original behavior."""
        if self._current_lora is None:
            return
        self._cleanup_lora_hooks()
        logger.info("LoRA unloaded: %s", self._current_lora)
        self._current_lora = None
        self._gpu_cleanup()

    def _cleanup_lora_hooks(self):
        """Remove all registered LoRA hooks and free A/B params."""
        for handle in getattr(self, "_lora_hooks", []):
            handle.remove()
        self._lora_hooks = []
        self._lora_params = []

    # -------------------------------------------------------------------
    # Preprocessor (lazy loading)
    # -------------------------------------------------------------------
    def get_preprocessor(self, mode: str):
        """Get a preprocessor, lazy-loading on first use.

        TODO: Phase 4 — implement preprocessor loading.
        """
        if mode not in self._preprocessors:
            logger.info("Loading preprocessor: %s", mode)
            # TODO: Phase 4
            # from preprocessors import load_preprocessor
            # self._preprocessors[mode] = load_preprocessor(mode, self.model_dir)
        return self._preprocessors.get(mode)
