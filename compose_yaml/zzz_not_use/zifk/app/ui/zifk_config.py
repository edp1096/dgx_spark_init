"""ZIFK configuration — paths, constants, resolution presets."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
MODEL_DIR = Path(os.environ.get("ZIFK_MODEL_DIR", Path.home() / ".cache" / "huggingface" / "hub" / "zifk"))
OUTPUT_DIR = Path(os.environ.get("ZIFK_OUTPUT_DIR", "/tmp/zifk-outputs"))
LOG_DIR = Path(os.environ.get("ZIFK_LOG_DIR", "./logs"))

# ---------------------------------------------------------------------------
# Z-Image model subdirectories (inside MODEL_DIR)
# Each folder contains transformer/ (FP8 in-place), vae/, text_encoder/, etc.
# ---------------------------------------------------------------------------
ZIMAGE_TURBO_DIR = "Z-Image-Turbo"
ZIMAGE_BASE_DIR = "Z-Image"

# ---------------------------------------------------------------------------
# FLUX.2 Klein model files (inside MODEL_DIR, FP8 official)
# ---------------------------------------------------------------------------
KLEIN_MODEL_FILE = "flux-2-klein-4b-fp8.safetensors"
KLEIN_BASE_MODEL_FILE = "flux-2-klein-base-4b-fp8.safetensors"
KLEIN_AE_FILE = "ae.safetensors"

# Klein model variant names (keys in FLUX2_MODEL_INFO)
KLEIN_DISTILLED = "flux.2-klein-4b"
KLEIN_BASE = "flux.2-klein-base-4b"

# ---------------------------------------------------------------------------
# LoRA directories (inside MODEL_DIR)
# ---------------------------------------------------------------------------
LORAS_ZIMAGE_DIR = "loras/zimage"
LORAS_KLEIN_DIR = "loras/klein"

# ---------------------------------------------------------------------------
# HuggingFace repo IDs
# ---------------------------------------------------------------------------
ZIMAGE_TURBO_REPO = "Tongyi-MAI/Z-Image-Turbo"
ZIMAGE_BASE_REPO = "Tongyi-MAI/Z-Image"
KLEIN_4B_FP8_REPO = "black-forest-labs/FLUX.2-klein-4b-fp8"
KLEIN_BASE_4B_FP8_REPO = "black-forest-labs/FLUX.2-klein-base-4b-fp8"
KLEIN_AE_REPO = "black-forest-labs/FLUX.2-dev"
KLEIN_TEXT_ENCODER_REPO = "Qwen/Qwen3-4B-FP8"

# ---------------------------------------------------------------------------
# Resolution presets
# ---------------------------------------------------------------------------
RESOLUTION_CHOICES = [
    "512x768",
    "768x512",
    "512x512",
    "768x1024",
    "1024x768",
    "1024x1024",
    "1280x720",
    "720x1280",
    "1536x1024",
    "1024x1536",
    "1920x1080",
    "1080x1920",
]

# ---------------------------------------------------------------------------
# Sample prompts
# ---------------------------------------------------------------------------
SAMPLE_PROMPTS = [
    "Young Korean woman in red Hanbok is smile, intricate embroidery. Impeccable makeup. Soft-lit outdoor night background, silhouetted tiered pagoda, blurred colorful distant lights.",
    "A golden retriever puppy sitting on a wooden porch at sunset, warm golden light creating a soft glow on its fur, shallow depth of field, shot on 35mm film, Kodak Portra 800, slight film grain.",
    "Aerial view of a winding river through autumn mountains, vibrant orange and red foliage reflecting on the water surface, morning mist rising, cinematic landscape photography.",
    "A steaming cup of matcha latte with intricate foam art on a marble countertop, soft window light, minimalist Japanese cafe interior, shallow focus, clean aesthetic.",
    "Studio portrait of an elderly Korean man with deep wrinkles and kind eyes, wearing a traditional hanbok, Rembrandt lighting, medium format scan, hyper-realistic skin texture.",
]
