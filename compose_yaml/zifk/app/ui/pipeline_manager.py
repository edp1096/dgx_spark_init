"""Pipeline lifecycle manager for ZIFK — Z-Image + FLUX.2 Klein.

Manages two independent model families on GPU. Only one family loaded at a time.
Switching between families triggers full cleanup of the previous.

Z-Image: native pipeline via Tongyi-MAI/Z-Image (generate function)
Klein:   native pipeline via black-forest-labs/flux2 (denoise functions)
"""

import gc
import logging
import os
import types
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
# q8_kernels FP8 matmul — native FP8 GEMM
# ---------------------------------------------------------------------------
_Q8_AVAILABLE = False
_fp8_mm = None
try:
    from q8_kernels.modules.linear import FP8Linear as _Q8FP8Linear
    from q8_kernels.functional.linear import fp8_mm as _fp8_mm
    _Q8_AVAILABLE = True
except ImportError:
    pass


def _fp8_forward_no_hadamard(self, x, x_scales=None, out_dtype=None):
    """FP8 forward WITHOUT Hadamard rotation on input.

    q8_kernels' default FP8LinearFunc always applies Hadamard rotation to
    16-bit inputs before quantizing to FP8, but fp8_gemm does NOT apply the
    inverse transform — producing ~500% relative error. This forward simply
    casts the input to FP8 and calls fp8_gemm directly.
    Output stays as fp8_mm's natural dtype (float32) to match QK normalization.
    """
    orig_shape = x.shape
    x_fp8 = x.to(torch.float8_e4m3fn).view(-1, orig_shape[-1])
    bias = self.bias.data if self.bias is not None else None
    o = _fp8_mm(x_fp8, self.weight.data, bias, False)
    return o.view(*orig_shape[:-1], self.weight.shape[0])


def _patch_transformer_q8(model):
    """Replace FP8 nn.Linear layers with q8_kernels FP8Linear for native FP8 GEMM.

    Swaps upcast-forward Linear layers for FP8Linear which runs matmul directly in FP8.
    Uses a custom forward that skips the Hadamard rotation.
    After patching, casts remaining FP8 parameters (embeddings, norms, tokens) to BF16.
    """
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
                module.bias.data.to(torch.float32), requires_grad=False,
            )
        fp8.forward = types.MethodType(_fp8_forward_no_hadamard, fp8)
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], fp8)
        count += 1

    # Cast remaining FP8 parameters (non-Linear: embeddings, norms, pad tokens, etc.) to BF16
    cast_count = 0
    for name, param in model.named_parameters():
        if param.dtype == torch.float8_e4m3fn:
            param.data = param.data.to(torch.bfloat16)
            cast_count += 1
    for name, buf in model.named_buffers():
        if buf.dtype == torch.float8_e4m3fn:
            buf.data = buf.data.to(torch.bfloat16)
            cast_count += 1

    if count > 0 or cast_count > 0:
        torch.cuda.empty_cache()
        logger.info("q8_kernels: %d Linear → FP8Linear, %d non-Linear params cast to BF16", count, cast_count)


# ---------------------------------------------------------------------------
# Z-Image FP8 compatibility monkey-patches
#
# The Z-Image source code assumes uniform dtype (BF16) throughout.
# With FP8Linear (float32 output) + BF16 norms, dtype mismatches occur.
# These patches fix the specific incompatible methods.
# ---------------------------------------------------------------------------
def _patch_zimage_fp8_compat(model):
    """Monkey-patch Z-Image transformer methods for FP8 dtype compatibility."""

    # 1. TimestepEmbedder: uses weight.dtype to cast input → FP8 if weight is FP8
    for name, module in model.named_modules():
        if type(module).__name__ == "TimestepEmbedder":
            _orig_te_forward = module.forward

            def _te_forward_patched(self_te, t, _orig=_orig_te_forward):
                t_freq = self_te.timestep_embedding(t, self_te.frequency_embedding_size)
                # Original: casts to weight.dtype which may be FP8
                # Fix: always use bfloat16 for timestep frequency input
                t_freq = t_freq.to(torch.bfloat16)
                t_emb = self_te.mlp(t_freq)
                return t_emb

            module.forward = types.MethodType(_te_forward_patched, module)
            logger.debug("Patched TimestepEmbedder.forward for FP8 compat")

    # 2. ZImageAttention: value not cast to query.dtype after QK norm
    for name, module in model.named_modules():
        if type(module).__name__ == "ZImageAttention":
            _orig_attn_forward = module.forward

            def _attn_forward_patched(self_attn, hidden_states, attention_mask=None, freqs_cis=None):
                from utils.attention import dispatch_attention

                query = self_attn.to_q(hidden_states)
                key = self_attn.to_k(hidden_states)
                value = self_attn.to_v(hidden_states)

                query = query.unflatten(-1, (self_attn.n_heads, -1))
                key = key.unflatten(-1, (self_attn.n_kv_heads, -1))
                value = value.unflatten(-1, (self_attn.n_kv_heads, -1))

                if self_attn.norm_q is not None:
                    query = self_attn.norm_q(query)
                if self_attn.norm_k is not None:
                    key = self_attn.norm_k(key)

                if freqs_cis is not None:
                    from zimage.transformer import apply_rotary_emb
                    query = apply_rotary_emb(query, freqs_cis)
                    key = apply_rotary_emb(key, freqs_cis)

                # Fix: cast ALL of q/k/v to the same dtype
                dtype = query.dtype
                query = query.to(dtype)
                key = key.to(dtype)
                value = value.to(dtype)  # ← Original code misses this

                hidden_states = dispatch_attention(
                    query, key, value,
                    attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                    backend=self_attn._attention_backend,
                )
                hidden_states = hidden_states.flatten(2, 3).to(dtype)
                output = self_attn.to_out[0](hidden_states)
                return output

            module.forward = types.MethodType(_attn_forward_patched, module)

    # 3. Pad tokens (x_pad_token, cap_pad_token): BF16 but x is float32 from FP8Linear
    #    Fix: cast pad tokens to float32
    for attr_name in ("x_pad_token", "cap_pad_token"):
        if hasattr(model, attr_name):
            param = getattr(model, attr_name)
            if isinstance(param, torch.nn.Parameter):
                param.data = param.data.to(torch.float32)
            elif isinstance(param, torch.Tensor):
                setattr(model, attr_name, param.to(torch.float32))

    attn_count = sum(1 for _, m in model.named_modules() if type(m).__name__ == "ZImageAttention")
    logger.info("Patched %d ZImageAttention + TimestepEmbedder + pad tokens for FP8 compat", attn_count)


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

        # Re-load transformer weights as FP8 (bypassing dtype cast) + apply q8_kernels
        transformer_dir = Path(self.model_dir) / model_dir_name / "transformer"
        fp8_file = transformer_dir / "model_fp8.safetensors"
        if fp8_file.exists() and _Q8_AVAILABLE:
            logger.info("Re-loading FP8 transformer for q8_kernels native GEMM...")
            from safetensors.torch import load_file
            fp8_state = load_file(str(fp8_file), device=str(self.device))
            self.zimage_components["transformer"].load_state_dict(
                fp8_state, strict=False, assign=True,
            )
            del fp8_state
            self._gpu_cleanup()
            _patch_transformer_q8(self.zimage_components["transformer"])
            _patch_zimage_fp8_compat(self.zimage_components["transformer"])
            # Set explicit dtype so pipeline.py uses BF16 for inputs, not FP8
            self.zimage_components["transformer"].dtype = torch.bfloat16
            logger.info("FP8 transformer loaded with q8_kernels native GEMM")
        elif fp8_file.exists():
            # q8_kernels not available — load_from_local_dir already cast to BF16, use as-is
            logger.warning("q8_kernels not available — using BF16 (cast from FP8 file)")

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
            _patch_transformer_q8(self.klein_model)
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
