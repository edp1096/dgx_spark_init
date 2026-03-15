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
    ZIMAGE_BASE_FP8_FILE,
    ZIMAGE_TURBO_DIR,
    ZIMAGE_TURBO_FP8_FILE,
)

logger = logging.getLogger("zifk-ui")

os.makedirs(str(OUTPUT_DIR), exist_ok=True)
os.makedirs(str(DEFAULT_MODEL_DIR), exist_ok=True)


# ---------------------------------------------------------------------------
# Required models per generation type
# ---------------------------------------------------------------------------
REQUIRED_MODELS = {
    "zit_t2i": [ZIMAGE_TURBO_DIR, ZIMAGE_TURBO_FP8_FILE],
    "zib_t2i": [ZIMAGE_BASE_DIR, ZIMAGE_BASE_FP8_FILE],
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
        self.klein_variant: str | None = None  # "flux.2-klein-4b" or "flux.2-klein-base-4b"

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

        Args:
            model_type: "turbo" or "base"
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

        self.zimage_components = load_from_local_dir(
            self._model_path(model_dir_name),
            device=str(self.device),
            dtype=torch.bfloat16,
            verbose=True,
            compile=False,
        )

        # Replace transformer weights with FP8 if available
        fp8_file = ZIMAGE_TURBO_FP8_FILE if model_type == "turbo" else ZIMAGE_BASE_FP8_FILE
        fp8_path = Path(self.model_dir) / fp8_file
        if fp8_path.exists():
            logger.info("Loading FP8 transformer weights from %s", fp8_file)
            from safetensors.torch import load_file
            fp8_state = load_file(str(fp8_path), device=str(self.device))
            self.zimage_components["transformer"].load_state_dict(fp8_state, strict=False, assign=True)
            del fp8_state
            self._gpu_cleanup()
            logger.info("FP8 transformer weights loaded (%s)", fp8_file)

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

        # Different Klein variant → reload
        if self.klein_loaded and self.klein_variant != variant:
            self.cleanup_klein()

        if self.current_family == "zimage":
            self.cleanup_zimage()

        label = "Klein 4B (Distilled)" if variant == KLEIN_DISTILLED else "Klein Base 4B"
        self._send_progress("loading_start", {"name": label, "index": 1, "total": 3})
        logger.info("Loading %s...", label)

        from flux2.util import load_ae, load_flow_model, load_text_encoder

        # Set model paths via environment variables
        model_file = KLEIN_MODEL_FILE if variant == KLEIN_DISTILLED else KLEIN_BASE_MODEL_FILE
        env_key = "KLEIN_4B_MODEL_PATH" if variant == KLEIN_DISTILLED else "KLEIN_4B_BASE_MODEL_PATH"
        os.environ[env_key] = self._model_path(model_file)
        os.environ["AE_MODEL_PATH"] = self._model_path(KLEIN_AE_FILE)

        # Load flow model
        self._send_progress("loading_start", {"name": "Klein Flow Model", "index": 1, "total": 3})
        self.klein_model = load_flow_model(variant, device=str(self.device))
        self.klein_model.eval()

        # Load text encoder
        self._send_progress("loading_start", {"name": "Klein Text Encoder", "index": 2, "total": 3})
        self.klein_text_encoder = load_text_encoder(variant, device=str(self.device))
        self.klein_text_encoder.eval()

        # Load autoencoder
        self._send_progress("loading_start", {"name": "Klein AutoEncoder", "index": 3, "total": 3})
        self.klein_ae = load_ae(variant)
        self.klein_ae.eval()

        self.klein_loaded = True
        self.klein_variant = variant
        self.current_family = "klein"

        self._send_progress("loading_done", {"name": label, "index": 3, "total": 3, "elapsed": 0})
        logger.info("%s ready", label)
