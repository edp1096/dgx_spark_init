"""Pipeline lifecycle manager for ZIFK — Z-Image + FLUX.2 Klein.

Manages two independent model families on GPU. Only one family loaded at a time.
Switching between families triggers full cleanup of the previous.

Z-Image: native pipeline via Tongyi-MAI/Z-Image (generate function)
Klein:   native pipeline via black-forest-labs/flux2 (denoise functions)
"""

import gc
import logging
import os
from pathlib import Path

import torch

from zifk_config import (
    KLEIN_AE_FILE,
    KLEIN_BASE,
    KLEIN_BASE_MODEL_FILE,
    KLEIN_DISTILLED,
    KLEIN_MODEL_FILE,
    LORAS_KLEIN_DIR,
    LORAS_ZIMAGE_DIR,
    MODEL_DIR as DEFAULT_MODEL_DIR,
    OUTPUT_DIR,
    ZIMAGE_BASE_DIR,
    ZIMAGE_TURBO_DIR,
)

logger = logging.getLogger("zifk-ui")

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
    "zib_t2i": [ZIMAGE_BASE_DIR],
    "klein_t2i": [KLEIN_MODEL_FILE, KLEIN_AE_FILE],
    "klein_base_t2i": [KLEIN_BASE_MODEL_FILE, KLEIN_AE_FILE],
    "klein_edit": [KLEIN_MODEL_FILE, KLEIN_AE_FILE],
    "klein_multiref": [KLEIN_MODEL_FILE, KLEIN_AE_FILE],
}


def scan_lora_files(family: str = "zimage") -> list[str]:
    """Scan LoRA directory for .safetensors files."""
    subdir = LORAS_ZIMAGE_DIR if family == "zimage" else LORAS_KLEIN_DIR
    loras_dir = Path(DEFAULT_MODEL_DIR) / subdir
    if not loras_dir.exists():
        loras_dir.mkdir(parents=True, exist_ok=True)
        return []
    return sorted(f.name for f in loras_dir.glob("*.safetensors"))


class PipelineManager:
    """Manages Z-Image and Klein pipeline lifecycles — one family on GPU at a time."""

    def __init__(self, progress_queue=None) -> None:
        # Z-Image components
        self.zimage_components: dict | None = None
        self.zimage_type: str | None = None  # "turbo" or "base"

        # Klein components
        self.klein_model = None
        self.klein_text_encoder = None
        self.klein_ae = None
        self.klein_loaded: bool = False
        self.klein_variant: str | None = None

        # Current state
        self.current_family: str | None = None  # "zimage" or "klein"
        self.model_dir: str = str(DEFAULT_MODEL_DIR)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Z-Image attention backend setting
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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def cleanup_zimage(self):
        if self.zimage_components is not None:
            logger.info("Cleaning up Z-Image components")
            for key in list(self.zimage_components.keys()):
                del self.zimage_components[key]
            self.zimage_components = None
            self.zimage_type = None
            if self.current_family == "zimage":
                self.current_family = None
            self._gpu_cleanup()

    def cleanup_klein(self):
        if self.klein_loaded:
            logger.info("Cleaning up Klein components")
            del self.klein_model
            del self.klein_text_encoder
            del self.klein_ae
            self.klein_model = None
            self.klein_text_encoder = None
            self.klein_ae = None
            self.klein_loaded = False
            self.klein_variant = None
            if self.current_family == "klein":
                self.current_family = None
            self._gpu_cleanup()

    def cleanup_all(self):
        self.cleanup_zimage()
        self.cleanup_klein()

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
    # Z-Image loading
    # -------------------------------------------------------------------
    def load_zimage(self, model_type: str = "turbo"):
        """Load Z-Image native pipeline components.

        The model folder contains FP8 transformer (in-place converted) + BF16 VAE/text_encoder.
        load_from_local_dir() loads everything, then we apply q8_kernels FP8 GEMM
        to the transformer's Linear layers.
        """
        if self.current_family == "zimage" and self.zimage_type == model_type:
            return self.zimage_components

        if self.current_family == "klein":
            self.cleanup_klein()

        if self.zimage_type is not None and self.zimage_type != model_type:
            self.cleanup_zimage()

        model_dir_name = ZIMAGE_TURBO_DIR if model_type == "turbo" else ZIMAGE_BASE_DIR
        label = f"Z-Image-{'Turbo' if model_type == 'turbo' else 'Base'}"

        self._send_progress("loading_start", {"name": label, "index": 1, "total": 1})
        logger.info("Loading %s from %s...", label, model_dir_name)

        from utils import load_from_local_dir, set_attention_backend

        set_attention_backend(self.attention_backend)

        # Load all components. Transformer dir now has FP8 weights (model_fp8.safetensors).
        # load_from_local_dir casts to dtype=bfloat16 which upcasts FP8→BF16,
        # but we immediately replace with FP8 + q8_kernels patch below.
        self.zimage_components = load_from_local_dir(
            self._model_path(model_dir_name),
            device=str(self.device),
            dtype=torch.bfloat16,
            verbose=True,
            compile=False,
        )

        # Apply FP8 GEMM for Turbo only — Base uses BF16 due to FP8 quality loss
        # over 28 denoising steps (per-step quantization error accumulation).
        transformer_dir = Path(self.model_dir) / model_dir_name / "transformer"
        fp8_file = transformer_dir / "model_fp8.safetensors"
        if model_type == "turbo" and fp8_file.exists() and _Q8_AVAILABLE:
            logger.info("Re-loading FP8 transformer for q8_kernels native GEMM...")
            from safetensors.torch import load_file
            fp8_state = load_file(str(fp8_file), device=str(self.device))
            self.zimage_components["transformer"].load_state_dict(
                fp8_state, strict=False, assign=True,
            )
            del fp8_state
            self._gpu_cleanup()
            weight_scales = _load_fp8_weight_scales(str(fp8_file))
            _patch_transformer_q8(self.zimage_components["transformer"], weight_scales=weight_scales)
            self.zimage_components["transformer"].dtype = torch.bfloat16
            logger.info("FP8 transformer loaded with q8_kernels native GEMM")
        else:
            logger.info("Using BF16 transformer (no FP8 GEMM)")

        self.zimage_type = model_type
        self.current_family = "zimage"

        self._send_progress("loading_done", {"name": label, "index": 1, "total": 1, "elapsed": 0})
        logger.info("%s ready", label)
        return self.zimage_components

    # -------------------------------------------------------------------
    # Klein loading
    # -------------------------------------------------------------------
    def load_klein(self, variant: str = KLEIN_DISTILLED):
        """Load FLUX.2 Klein native pipeline components.

        Args:
            variant: "flux.2-klein-4b" (distilled) or "flux.2-klein-base-4b" (CFG)
        """
        if self.current_family == "klein" and self.klein_loaded and self.klein_variant == variant:
            return

        if self.klein_loaded and self.klein_variant != variant:
            self.cleanup_klein()

        if self.current_family == "zimage":
            self.cleanup_zimage()

        label = "Klein 4B (Distilled)" if variant == KLEIN_DISTILLED else "Klein Base 4B"
        self._send_progress("loading_start", {"name": label, "index": 1, "total": 3})
        logger.info("Loading %s...", label)

        from flux2.util import load_ae, load_flow_model, load_text_encoder

        model_file = KLEIN_MODEL_FILE if variant == KLEIN_DISTILLED else KLEIN_BASE_MODEL_FILE
        env_key = "KLEIN_4B_MODEL_PATH" if variant == KLEIN_DISTILLED else "KLEIN_4B_BASE_MODEL_PATH"
        os.environ[env_key] = self._model_path(model_file)
        os.environ["AE_MODEL_PATH"] = self._model_path(KLEIN_AE_FILE)

        self._send_progress("loading_start", {"name": "Klein Flow Model", "index": 1, "total": 3})
        self.klein_model = load_flow_model(variant, device=str(self.device))
        self.klein_model.eval()

        # Apply q8_kernels FP8 GEMM if FP8 weights detected
        has_fp8 = any(
            p.dtype == torch.float8_e4m3fn
            for p in self.klein_model.parameters()
        )
        if has_fp8 and _Q8_AVAILABLE:
            # Load weight_scale values from the FP8 checkpoint for proper dequantization
            weight_scales = _load_fp8_weight_scales(self._model_path(model_file))
            _patch_transformer_q8(self.klein_model, weight_scales=weight_scales)
            self.klein_model.dtype = torch.bfloat16
            logger.info("Klein flow model: q8_kernels FP8 GEMM applied")
        elif has_fp8:
            logger.warning("Klein FP8 weights detected but q8_kernels not available — casting to BF16")
            for name, param in self.klein_model.named_parameters():
                if param.dtype == torch.float8_e4m3fn:
                    param.data = param.data.to(torch.bfloat16)

        self._send_progress("loading_start", {"name": "Klein Text Encoder", "index": 2, "total": 3})
        self.klein_text_encoder = load_text_encoder(variant, device=str(self.device))
        self.klein_text_encoder.eval()

        self._send_progress("loading_start", {"name": "Klein AutoEncoder", "index": 3, "total": 3})
        self.klein_ae = load_ae(variant)
        self.klein_ae.eval()

        self.klein_loaded = True
        self.klein_variant = variant
        self.current_family = "klein"

        self._send_progress("loading_done", {"name": label, "index": 3, "total": 3, "elapsed": 0})
        logger.info("%s ready", label)
