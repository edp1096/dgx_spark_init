"""Pipeline lifecycle management for Z-Image using diffusers."""

import gc
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger("zimage-ui")

from config import MODEL_DIR as DEFAULT_MODEL_DIR, OUTPUT_DIR, RESOLUTION_CHOICES

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(str(DEFAULT_MODEL_DIR), exist_ok=True)

TURBO_DIR = "Z-Image-Turbo"
BASE_DIR = "Z-Image"
LORAS_DIR = Path(DEFAULT_MODEL_DIR) / "loras"
os.makedirs(str(LORAS_DIR), exist_ok=True)

REQUIRED_MODELS = {
    "turbo": [TURBO_DIR],
    "base": [BASE_DIR],
    "img2img_turbo": [TURBO_DIR],
    "img2img_base": [BASE_DIR],
}

SAMPLE_PROMPTS = [
    "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Soft-lit outdoor night background, silhouetted tiered pagoda, blurred colorful distant lights.",
    "A golden retriever puppy sitting on a wooden porch at sunset, warm golden light creating a soft glow on its fur, shallow depth of field, shot on 35mm film, Kodak Portra 800, slight film grain.",
    "Aerial view of a winding river through autumn mountains, vibrant orange and red foliage reflecting on the water surface, morning mist rising, cinematic landscape photography, 4:3 aspect ratio.",
    "A steaming cup of matcha latte with intricate foam art on a marble countertop, soft window light, minimalist Japanese cafe interior, shallow focus, clean aesthetic.",
    "Studio portrait of an elderly Korean man with deep wrinkles and kind eyes, wearing a traditional hanbok, Rembrandt lighting, medium format scan, hyper-realistic skin texture.",
]


def scan_lora_files() -> list[str]:
    """Scan {model_dir}/loras/ for .safetensors files."""
    loras_dir = Path(DEFAULT_MODEL_DIR) / "loras"
    if not loras_dir.exists():
        return []
    return sorted(f.name for f in loras_dir.glob("*.safetensors"))


class PipelineManager:
    """Manages Z-Image pipeline lifecycle — one active pipeline at a time."""

    def __init__(self, progress_queue=None) -> None:
        self.current_pipeline = None
        self.current_type: str | None = None
        self.model_dir: str = str(DEFAULT_MODEL_DIR)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._current_loras: list[dict] = []
        # Progress tracking
        self._is_generating: bool = False
        self._current_loading: str | None = None
        self.progress_queue = progress_queue
        self._current_task_id: str | None = None

    def _cleanup(self) -> None:
        if self.current_pipeline is not None:
            logger.info("Cleaning up pipeline: %s", self.current_type)
            del self.current_pipeline
            self.current_pipeline = None
            self.current_type = None
            self._current_loras = []
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def check_models(self, pipeline_type: str) -> list[str]:
        required = REQUIRED_MODELS.get(pipeline_type, [])
        missing = []
        for d in required:
            path = Path(self.model_dir) / d
            if not path.exists() or not any(path.rglob("*.safetensors")):
                missing.append(d)
        return missing

    def _model_path(self, subdir: str) -> str:
        return str(Path(self.model_dir) / subdir)

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

    def get_turbo(self, loras: list[dict] | None = None):
        loras = loras or []
        if self.current_type == "turbo" and self._current_loras == loras:
            return self.current_pipeline
        self._cleanup()
        from diffusers import ZImagePipeline

        self._send_progress("loading_start", {"name": "Z-Image-Turbo", "index": 1, "total": 1})
        logger.info("Loading Z-Image-Turbo pipeline...")

        self.current_pipeline = ZImagePipeline.from_pretrained(
            self._model_path(TURBO_DIR),
            torch_dtype=torch.bfloat16,
        )
        self.current_pipeline.to(self.device)
        self._apply_loras(loras)
        self.current_type = "turbo"
        self._current_loras = loras
        self._send_progress("loading_done", {"name": "Z-Image-Turbo", "index": 1, "total": 1, "elapsed": 0})
        logger.info("Z-Image-Turbo ready")
        return self.current_pipeline

    def get_base(self, loras: list[dict] | None = None):
        loras = loras or []
        if self.current_type == "base" and self._current_loras == loras:
            return self.current_pipeline
        self._cleanup()
        from diffusers import ZImagePipeline

        self._send_progress("loading_start", {"name": "Z-Image-Base", "index": 1, "total": 1})
        logger.info("Loading Z-Image-Base pipeline...")

        self.current_pipeline = ZImagePipeline.from_pretrained(
            self._model_path(BASE_DIR),
            torch_dtype=torch.bfloat16,
        )
        self.current_pipeline.to(self.device)
        self._apply_loras(loras)
        self.current_type = "base"
        self._current_loras = loras
        self._send_progress("loading_done", {"name": "Z-Image-Base", "index": 1, "total": 1, "elapsed": 0})
        logger.info("Z-Image-Base ready")
        return self.current_pipeline

    def get_img2img(self, use_base: bool = False, loras: list[dict] | None = None):
        loras = loras or []
        target = "img2img_base" if use_base else "img2img_turbo"
        if self.current_type == target and self._current_loras == loras:
            return self.current_pipeline
        self._cleanup()
        from diffusers import ZImageImg2ImgPipeline

        repo_dir = BASE_DIR if use_base else TURBO_DIR
        label = "Z-Image-Img2Img-Base" if use_base else "Z-Image-Img2Img-Turbo"

        self._send_progress("loading_start", {"name": label, "index": 1, "total": 1})
        logger.info("Loading %s pipeline...", label)

        self.current_pipeline = ZImageImg2ImgPipeline.from_pretrained(
            self._model_path(repo_dir),
            torch_dtype=torch.bfloat16,
        )
        self.current_pipeline.to(self.device)
        self._apply_loras(loras)
        self.current_type = target
        self._current_loras = loras
        self._send_progress("loading_done", {"name": label, "index": 1, "total": 1, "elapsed": 0})
        logger.info("%s ready", label)
        return self.current_pipeline

    def _apply_loras(self, loras: list[dict]):
        """Load LoRA weights into the current pipeline."""
        if not loras or self.current_pipeline is None:
            return
        for entry in loras:
            path = Path(self.model_dir) / "loras" / entry["filename"]
            if path.exists():
                strength = entry.get("strength", 1.0)
                logger.info("Loading LoRA: %s (strength=%.2f)", entry["filename"], strength)
                self.current_pipeline.load_lora_weights(
                    str(path.parent), weight_name=path.name,
                )
                self.current_pipeline.fuse_lora(lora_scale=strength)
            else:
                logger.warning("LoRA not found: %s", path)
