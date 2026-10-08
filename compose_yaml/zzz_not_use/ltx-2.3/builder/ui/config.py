"""Shared configuration — single source of truth for paths and constants."""

import os
from pathlib import Path

MODEL_DIR = Path(os.environ.get("LTX_MODEL_DIR", str(Path.home() / ".cache" / "huggingface" / "hub" / "ltx23")))
OUTPUT_DIR = os.environ.get("LTX_OUTPUT_DIR", "/tmp/ltx2-outputs")
LOG_DIR = Path(os.environ.get("LTX_LOG_DIR", str(Path(__file__).resolve().parent.parent / "logs")))
