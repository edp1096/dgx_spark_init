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

        # LoRA state (multi-LoRA stack)
        self._current_lora: str | None = None  # backward compat
        self._current_lora_stack: list[dict] = []  # [{"name": str, "scale": float}, ...]
        self._lora_scales: dict[str, float] = {}  # name -> user scale
        self._lora_alpha_scales: dict[str, float] = {}  # name -> alpha/rank scale

        # Precision state
        self._loaded_precision = "BF16"

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
    def load_zit(self, need_controlnet: bool = True):
        """Load Z-Image-Turbo pipeline (BF16), optionally with ControlNet adapter.

        Args:
            need_controlnet: If False, skip loading ControlNet adapter weights
                into the transformer. This improves LoRA face quality for pure T2I.
        """
        # If already loaded, check if ControlNet mode changed
        if self.zit_components is not None:
            if self._need_controlnet != need_controlnet:
                logger.info("ControlNet mode changed (%s → %s), reloading transformer...",
                            "with CN" if self._need_controlnet else "without CN",
                            "with CN" if need_controlnet else "without CN")
                self._need_controlnet = need_controlnet
                self._reload_transformer()
            return self.zit_components

        self._need_controlnet = need_controlnet

        from diffusers import FlowMatchEulerDiscreteScheduler
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from videox_models.z_image_transformer2d_control import ZImageControlTransformer2DModel

        model_path = Path(self.model_dir) / ZIMAGE_TURBO_DIR
        transformer_dir = model_path / "transformer"

        # --- Step 1: Load transformer (BF16 from_pretrained + ControlNet adapter) ---
        self._send_progress("loading_start", {"name": "ZIT Transformer (BF16)", "index": 1, "total": 4})
        from helpers import fast_load_file

        logger.info("Loading BF16 transformer from %s ...", transformer_dir)
        transformer = ZImageControlTransformer2DModel.from_pretrained(
            str(model_path),
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            transformer_additional_kwargs=CONTROLNET_CONFIG,
        ).to(self.device)

        # Load ControlNet adapter weights
        cn_path = Path(self.model_dir) / CONTROLNET_DIR / CONTROLNET_FILENAME
        if self._need_controlnet and cn_path.exists():
            logger.info("Loading ControlNet adapter from %s ...", cn_path)
            cn_state = fast_load_file(str(cn_path), device=str(self.device))
            m, u = transformer.load_state_dict(cn_state, strict=False)
            del cn_state
            logger.info("ControlNet adapter loaded (missing=%d, unexpected=%d)", len(m), len(u))
            self.controlnet_loaded = True
        elif not self._need_controlnet:
            logger.info("Skipping ControlNet adapter (pure T2I mode)")
            self.controlnet_loaded = False
        else:
            logger.warning("ControlNet model not found at %s", cn_path)
            self.controlnet_loaded = False

        self._gpu_cleanup()
        transformer.eval()
        self._loaded_precision = "BF16"
        logger.info("BF16 transformer ready (ControlNet=%s)", "loaded" if self.controlnet_loaded else "skipped")

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
                     self._loaded_precision)
        return self.zit_components

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

        self._send_progress("loading_start", {"name": "Transformer reload (BF16)", "index": 1, "total": 1})
        from helpers import fast_load_file
        from videox_models.z_image_transformer2d_control import ZImageControlTransformer2DModel

        logger.info("Reloading BF16 transformer...")
        transformer = ZImageControlTransformer2DModel.from_pretrained(
            str(model_path),
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            transformer_additional_kwargs=CONTROLNET_CONFIG,
        ).to(self.device)

        cn_path = Path(self.model_dir) / CONTROLNET_DIR / CONTROLNET_FILENAME
        if self._need_controlnet and cn_path.exists():
            cn_state = fast_load_file(str(cn_path), device=str(self.device))
            transformer.load_state_dict(cn_state, strict=False)
            del cn_state
            self.controlnet_loaded = True
        else:
            self.controlnet_loaded = False
        self._gpu_cleanup()
        transformer.eval()
        self._loaded_precision = "BF16"

        # Update pipeline and components
        self.zit_components["transformer"] = transformer
        self.zit_components["pipeline"].transformer = transformer

        self._send_progress("loading_done", {"name": "Transformer reload", "index": 1, "total": 1, "elapsed": 0})
        logger.info("Transformer reloaded (precision=%s)", self._loaded_precision)

    # -------------------------------------------------------------------
    # LoRA loading / unloading (forward-hook based, multi-LoRA stack)
    # -------------------------------------------------------------------
    def load_lora(self, lora_name: str, lora_scale: float = 1.0):
        """Backward-compatible single LoRA loader."""
        if not lora_name or lora_name == "None":
            self.unload_all_loras()
            return
        self.load_lora_stack([{"name": lora_name, "scale": lora_scale}])

    def load_lora_stack(self, lora_stack: list[dict]):
        """Load multiple LoRAs via forward hooks. Each entry: {"name": str, "scale": float}.

        - If only scales changed (same names), update scales without reload.
        - If names changed, full unload + reload.
        - LoRA A/B stored in BF16.
        """
        if not lora_stack:
            self.unload_all_loras()
            return

        if self.zit_components is None:
            logger.warning("Cannot load LoRA — ZIT pipeline not loaded yet")
            return

        # Normalize stack
        new_stack = [{"name": e["name"], "scale": float(e.get("scale", 1.0))}
                     for e in lora_stack if e.get("name") and e["name"] != "None"]
        if not new_stack:
            self.unload_all_loras()
            return

        # Check if only scales changed (same LoRA names in same order)
        old_names = [e["name"] for e in self._current_lora_stack]
        new_names = [e["name"] for e in new_stack]
        if old_names == new_names:
            # Scale-only update — no reload needed
            for entry in new_stack:
                self._lora_scales[entry["name"]] = entry["scale"]
            self._current_lora_stack = new_stack
            self._current_lora = new_stack[0]["name"] if new_stack else None
            logger.info("LoRA scales updated: %s",
                        ", ".join(f"{e['name']}={e['scale']:.2f}" for e in new_stack))
            return

        # Names differ — full reload
        self.unload_all_loras()

        from helpers import fast_load_file, fast_safe_metadata
        transformer = self.zit_components["transformer"]
        self._lora_hooks = []
        self._lora_params = []
        self._lora_scales = {}
        self._lora_alpha_scales = {}
        total_hooks = 0

        for entry in new_stack:
            lora_name = entry["name"]
            lora_scale = entry["scale"]
            lora_path = Path(self.model_dir) / LORAS_DIR / lora_name

            if not lora_path.exists():
                logger.error("LoRA file not found: %s", lora_path)
                continue

            try:
                logger.info("Loading LoRA: %s (scale=%.2f)", lora_name, lora_scale)

                # Read metadata for alpha/rank auto-scaling
                meta = fast_safe_metadata(str(lora_path))
                file_alpha = int(meta["lora_alpha"]) if "lora_alpha" in meta else None
                file_rank = int(meta["rank"]) if "rank" in meta else None
                if file_alpha is not None and file_rank is not None and file_rank > 0:
                    alpha_scale = file_alpha / file_rank
                else:
                    alpha_scale = 1.0

                self._lora_scales[lora_name] = lora_scale
                self._lora_alpha_scales[lora_name] = alpha_scale

                lora_sd = fast_load_file(str(lora_path), device=str(self.device))

                # Parse LoRA state dict: group A/B pairs by module path
                lora_pairs = {}
                for key, tensor in lora_sd.items():
                    if ".lora_A." not in key and ".lora_B." not in key:
                        continue
                    if ".lora_A." in key:
                        module_path = key.split(".lora_A.")[0]
                        ab = "A"
                    else:
                        module_path = key.split(".lora_B.")[0]
                        ab = "B"
                    module_path = module_path.removeprefix("base_model.model.")
                    if module_path not in lora_pairs:
                        lora_pairs[module_path] = {}
                    lora_pairs[module_path][ab] = tensor.to(torch.bfloat16)

                del lora_sd

                if not lora_pairs:
                    logger.warning("No LoRA A/B pairs found in %s", lora_name)
                    continue

                rank = 16
                for pair in lora_pairs.values():
                    if "A" in pair:
                        rank = pair["A"].shape[0]
                        break

                hook_count = 0
                for module_path, pair in lora_pairs.items():
                    if "A" not in pair or "B" not in pair:
                        continue
                    try:
                        target = transformer
                        for part in module_path.split("."):
                            target = getattr(target, part)
                    except AttributeError:
                        continue

                    lora_A = pair["A"]
                    lora_B = pair["B"]
                    self._lora_params.extend([lora_A, lora_B])

                    def _make_hook(A, B, name):
                        def hook(module, input, output):
                            x = input[0] if isinstance(input, tuple) else input
                            lora_out = x.to(torch.bfloat16) @ A.t() @ B.t()
                            scale = self._lora_scales.get(name, 1.0)
                            alpha_s = self._lora_alpha_scales.get(name, 1.0)
                            return output + lora_out * (alpha_s * scale)
                        return hook

                    handle = target.register_forward_hook(_make_hook(lora_A, lora_B, lora_name))
                    self._lora_hooks.append(handle)
                    hook_count += 1

                total_hooks += hook_count
                logger.info("LoRA loaded: %s (%d hooks, rank=%d, alpha_scale=%.4f)",
                            lora_name, hook_count, rank, alpha_scale)

            except Exception as e:
                logger.error("Failed to load LoRA %s: %s", lora_name, e)
                import traceback
                traceback.print_exc()

        self._current_lora_stack = new_stack
        self._current_lora = new_stack[0]["name"] if new_stack else None
        logger.info("LoRA stack loaded: %d LoRA(s), %d total hooks", len(new_stack), total_hooks)

    def unload_lora(self):
        """Backward-compatible unload."""
        self.unload_all_loras()

    def unload_all_loras(self):
        """Remove all LoRA forward hooks, restoring original behavior."""
        if not self._current_lora_stack and self._current_lora is None:
            return
        self._cleanup_lora_hooks()
        names = [e["name"] for e in self._current_lora_stack] or [self._current_lora]
        logger.info("LoRA unloaded: %s", ", ".join(n for n in names if n))
        self._current_lora = None
        self._current_lora_stack = []
        self._lora_scales = {}
        self._lora_alpha_scales = {}
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
