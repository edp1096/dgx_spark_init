"""ZIT configuration — paths, constants, resolution presets."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
MODEL_DIR = Path(os.environ.get("ZIT_MODEL_DIR", Path.home() / ".cache" / "huggingface" / "hub" / "zit"))
OUTPUT_DIR = Path(os.environ.get("ZIT_OUTPUT_DIR", "/tmp/zit-outputs"))
LOG_DIR = Path(os.environ.get("ZIT_LOG_DIR", "./logs"))

# ---------------------------------------------------------------------------
# Z-Image Turbo (base model, only model)
# ---------------------------------------------------------------------------
ZIMAGE_TURBO_DIR = "Z-Image-Turbo"
ZIMAGE_TURBO_REPO = "Tongyi-MAI/Z-Image-Turbo"

# ---------------------------------------------------------------------------
# ControlNet Union (inside MODEL_DIR/controlnet/)
# ---------------------------------------------------------------------------
CONTROLNET_DIR = "controlnet"
CONTROLNET_FILENAME = "Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps.safetensors"
CONTROLNET_REPO = "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1"

# ---------------------------------------------------------------------------
# Preprocessor models (inside MODEL_DIR/preprocessors/)
# ---------------------------------------------------------------------------
PREPROCESSORS_DIR = "preprocessors"
DWPOSE_DET_FILE = "yolox_l.onnx"
DWPOSE_POSE_FILE = "dw-ll_ucoco_384.onnx"
ZOEDEPTH_FILE = "ZoeD_M12_N.pt"
HED_FILE = "ControlNetHED.pth"

# ---------------------------------------------------------------------------
# FaceSwap models (inside MODEL_DIR/faceswap/)
# ---------------------------------------------------------------------------
# Face detection for auto-mask (inside MODEL_DIR/preprocessors/)
# SCRFD is used to detect face bbox + landmarks → generate inpaint mask automatically
SCRFD_FILE = "scrfd_10g_bnkps.onnx"
SCRFD_URL = "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/scrfd_10g_bnkps.onnx"

# ---------------------------------------------------------------------------
# LoRA directories (inside MODEL_DIR)
# ---------------------------------------------------------------------------
LORAS_DIR = "loras"

# ---------------------------------------------------------------------------
# HuggingFace repo IDs for preprocessor weights
# ---------------------------------------------------------------------------
DWPOSE_DET_URL = "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx"
DWPOSE_POSE_URL = "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx"
ZOEDEPTH_URL = "https://huggingface.co/lllyasviel/Annotators/resolve/main/ZoeD_M12_N.pt"
HED_URL = "https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth"

# ---------------------------------------------------------------------------
# Control modes (ControlNet Union 2602)
# ---------------------------------------------------------------------------
CONTROL_MODES = ["canny", "pose", "depth", "hed", "scribble", "gray"]

# ---------------------------------------------------------------------------
# ControlNet config (for ZImageControlTransformer2DModel)
# ---------------------------------------------------------------------------
CONTROLNET_CONFIG = {
    "control_layers_places": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
    "control_refiner_layers_places": [0, 1],
    "add_control_noise_refiner": True,
    "add_control_noise_refiner_correctly": True,
    "control_in_dim": 33,
}

# ---------------------------------------------------------------------------
# Default generation parameters
# ---------------------------------------------------------------------------
DEFAULT_STEPS = 8
DEFAULT_TIME_SHIFT = 3.0
DEFAULT_GUIDANCE = 0.5
DEFAULT_CFG_TRUNCATION = 0.9
DEFAULT_MAX_SEQ_LENGTH = 512
DEFAULT_CONTROL_SCALE = 0.65

# Inpaint-specific defaults (aligned with official predict_i2i_inpaint_2.1.py)
DEFAULT_INPAINT_STEPS = 25
DEFAULT_INPAINT_GUIDANCE = 4.0
DEFAULT_INPAINT_CFG_TRUNCATION = 1.0
DEFAULT_INPAINT_CONTROL_SCALE = 0.9

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
