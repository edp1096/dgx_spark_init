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
    SCRFD_FILE,
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
    "faceswap": [f"{PREPROCESSORS_DIR}/{SCRFD_FILE}"],
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

        # FaceSwap (on-demand)
        self.faceswap_pipeline = None

        # LoRA state
        self._current_lora: str | None = None

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
    def load_zit(self):
        """Load Z-Image-Turbo + ControlNet Union as unified pipeline.

        Uses VideoX-Fun's ZImageControlTransformer2DModel which handles both
        pure T2I (with zero control_context) and ControlNet (with control_image).
        FP8 via q8_kernels patch (same approach as zifk).

        Loading order:
        1. Load base ZIT transformer weights into ZImageControlTransformer2DModel
        2. Load ControlNet adapter weights on top (strict=False)
        3. Apply FP8 q8_kernels GEMM patch
        4. Load VAE, text_encoder, tokenizer, scheduler (diffusers standard)
        """
        if self.zit_components is not None:
            return self.zit_components

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

        # --- Step 1b: Load ControlNet adapter weights on top ---
        cn_path = Path(self.model_dir) / CONTROLNET_DIR / CONTROLNET_FILENAME
        if cn_path.exists():
            self._send_progress("loading_start", {"name": "ControlNet Adapter", "index": 1, "total": 4})
            logger.info("Loading ControlNet adapter from %s...", cn_path)
            from safetensors.torch import load_file
            cn_state = load_file(str(cn_path), device=str(self.device))
            missing, unexpected = transformer.load_state_dict(cn_state, strict=False)
            del cn_state
            self._gpu_cleanup()
            logger.info("ControlNet adapter loaded (missing=%d, unexpected=%d)", len(missing), len(unexpected))
            self.controlnet_loaded = True
        else:
            logger.warning("ControlNet model not found at %s — T2I only", cn_path)

        # --- Step 1c: Apply FP8 q8_kernels GEMM ---
        fp8_file = transformer_dir / "model_fp8.safetensors"
        if fp8_file.exists() and _Q8_AVAILABLE:
            logger.info("Re-loading FP8 transformer for q8_kernels native GEMM...")
            from safetensors.torch import load_file
            fp8_state = load_file(str(fp8_file), device=str(self.device))
            transformer.load_state_dict(fp8_state, strict=False, assign=True)
            del fp8_state
            self._gpu_cleanup()
            weight_scales = _load_fp8_weight_scales(str(fp8_file))
            _patch_transformer_q8(transformer, weight_scales=weight_scales)
            logger.info("FP8 transformer loaded with q8_kernels native GEMM")
        else:
            logger.info("Using BF16 transformer (no FP8 GEMM)")

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
        logger.info("ZIT pipeline ready (ControlNet=%s)", "loaded" if self.controlnet_loaded else "not found")
        return self.zit_components

    # -------------------------------------------------------------------
    # FaceSwap loading (SCRFD cv2.dnn face detector)
    # -------------------------------------------------------------------
    def load_faceswap(self):
        """Load SCRFD face detector (cv2.dnn)."""
        if self.faceswap_pipeline is not None:
            return

        self._send_progress("loading_start", {"name": "FaceSwap (SCRFD)"})
        logger.info("Loading SCRFD face detector...")
        from face_swap import get_detector
        get_detector(self.model_dir)
        self.faceswap_pipeline = True  # flag — detector is a singleton
        self._send_progress("loading_done", {"name": "FaceSwap (SCRFD)"})
        logger.info("SCRFD face detector ready")

    # -------------------------------------------------------------------
    # LoRA loading / unloading (peft-based)
    # -------------------------------------------------------------------
    def load_lora(self, lora_name: str, lora_scale: float = 1.0):
        """Load a LoRA adapter onto the transformer via peft.

        Args:
            lora_name: filename in LORAS_DIR (e.g. "my_face.safetensors")
            lora_scale: LoRA strength (0.0 = off, 1.0 = full)
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

        transformer = self.zit_components["transformer"]

        # If same LoRA already loaded, just update scale
        if self._current_lora == lora_name:
            self._set_lora_scale(transformer, lora_scale)
            logger.info("LoRA scale updated: %.2f", lora_scale)
            return

        # Unload previous LoRA first
        if self._current_lora is not None:
            self.unload_lora()

        try:
            from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
            from safetensors.torch import load_file

            logger.info("Loading LoRA: %s (scale=%.2f)", lora_name, lora_scale)

            # Load LoRA state dict to determine target modules
            lora_sd = load_file(str(lora_path), device=str(self.device))

            # Extract target module names from LoRA keys
            # LoRA keys look like: "base_model.model.blocks.0.attn.to_q.lora_A.weight"
            # or simpler: "blocks.0.attn.to_q.lora_A.weight"
            target_modules = set()
            for key in lora_sd:
                if ".lora_A." in key or ".lora_B." in key:
                    # Extract module name before .lora_A/.lora_B
                    parts = key.split(".lora_A.")[0] if ".lora_A." in key else key.split(".lora_B.")[0]
                    module_suffix = parts.split(".")[-1]
                    target_modules.add(module_suffix)

            if not target_modules:
                # Fallback: standard attention targets
                target_modules = {"to_q", "to_k", "to_v"}

            # Detect rank from LoRA weights
            rank = 16  # default
            for key, tensor in lora_sd.items():
                if ".lora_A." in key and tensor.ndim == 2:
                    rank = tensor.shape[0]
                    break

            logger.info("LoRA config: rank=%d, targets=%s", rank, target_modules)

            lora_config = LoraConfig(
                r=rank,
                lora_alpha=rank,
                target_modules=list(target_modules),
                lora_dropout=0.0,
                bias="none",
            )

            # Wrap transformer with peft
            peft_model = get_peft_model(transformer, lora_config)

            # Load the actual LoRA weights
            # Remap keys to match peft's expected format
            peft_sd = {}
            for key, tensor in lora_sd.items():
                # Ensure keys start with base_model.model.
                if not key.startswith("base_model."):
                    peft_sd[f"base_model.model.{key}"] = tensor
                else:
                    peft_sd[key] = tensor

            missing, unexpected = peft_model.load_state_dict(peft_sd, strict=False)
            if missing:
                # Try loading without prefix remapping
                set_peft_model_state_dict(peft_model, lora_sd)

            del lora_sd, peft_sd

            # Apply scale
            self._set_lora_scale(peft_model, lora_scale)

            # Replace transformer in pipeline
            self.zit_components["transformer"] = peft_model
            self.zit_components["pipeline"].transformer = peft_model
            self._current_lora = lora_name

            self._gpu_cleanup()
            logger.info("LoRA loaded: %s", lora_name)

        except Exception as e:
            logger.error("Failed to load LoRA %s: %s", lora_name, e)
            import traceback
            traceback.print_exc()

    def unload_lora(self):
        """Remove LoRA adapter, restore original transformer (without merging)."""
        if self._current_lora is None:
            return

        if self.zit_components is None:
            self._current_lora = None
            return

        try:
            transformer = self.zit_components["transformer"]
            if hasattr(transformer, "disable_adapter_layers"):
                # Disable and remove peft adapters without merging
                transformer.disable_adapter_layers()
                base_model = transformer.base_model.model if hasattr(transformer, "base_model") else transformer
                self.zit_components["transformer"] = base_model
                self.zit_components["pipeline"].transformer = base_model
                logger.info("LoRA unloaded (no merge): %s", self._current_lora)
            elif hasattr(transformer, "merge_and_unload"):
                # Fallback: merge and unload
                base_model = transformer.merge_and_unload()
                self.zit_components["transformer"] = base_model
                self.zit_components["pipeline"].transformer = base_model
                logger.info("LoRA unloaded (merged): %s", self._current_lora)
            self._current_lora = None
            self._gpu_cleanup()
        except Exception as e:
            logger.error("Failed to unload LoRA: %s", e)
            self._current_lora = None

    @staticmethod
    def _set_lora_scale(model, scale: float):
        """Set LoRA scaling factor on all LoRA layers."""
        try:
            if hasattr(model, "set_adapter"):
                # peft >= 0.6
                for name, module in model.named_modules():
                    if hasattr(module, "scaling"):
                        for key in module.scaling:
                            module.scaling[key] = scale
        except Exception:
            pass

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
