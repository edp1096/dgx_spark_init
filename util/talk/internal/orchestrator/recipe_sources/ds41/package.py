#!/usr/bin/env python3
"""Use the shared, checked manifest instead of maintaining a second file list."""
from pathlib import Path
import subprocess
root = Path(__file__).resolve().parents[4]
subprocess.run(['go', 'run', './cmd/package-recipes'], cwd=root, check=True)
