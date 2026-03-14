"""Shared configuration — single source of truth for paths and constants."""

import os
from pathlib import Path

MODEL_DIR = Path(os.environ.get("ZIMAGE_MODEL_DIR", str(Path.home() / ".cache" / "huggingface" / "hub" / "zimage")))
OUTPUT_DIR = os.environ.get("ZIMAGE_OUTPUT_DIR", "/tmp/zimage-outputs")
LOG_DIR = Path(os.environ.get("ZIMAGE_LOG_DIR", str(Path(__file__).resolve().parent.parent / "logs")))

TURBO_REPO = "Tongyi-MAI/Z-Image-Turbo"
BASE_REPO = "Tongyi-MAI/Z-Image"

RESOLUTION_CHOICES = [
    "1024x1024",
    "1280x720", "720x1280",
    "1536x1024", "1024x1536",
    "1920x1080", "1080x1920",
    "768x1024", "1024x768",
    "512x768", "768x512",
    "512x512",
]
