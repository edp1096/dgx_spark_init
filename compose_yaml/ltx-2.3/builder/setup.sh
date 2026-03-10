#!/bin/bash

# Not required. Just use compose.yml

# LTX-2.3 Gradio UI — Setup Script for NGC PyTorch 26.02-py3 Container
# Downloads all dependencies from GitHub, installs packages, prepares environment.
#
# IMPORTANT: Uses --no-deps for ltx packages to preserve NGC's custom torch.
#
# Usage:
#   bash /root/ltx2-ui/setup.sh            # Full setup
#   bash /root/ltx2-ui/setup.sh --skip-q8  # Skip q8_kernels

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LTX_DIR="$SCRIPT_DIR/LTX-2"
UI_DIR="$SCRIPT_DIR/ui"
MODEL_DIR="${LTX_MODEL_DIR:-$HOME/.cache/huggingface/hub/ltx23}"
LTX_COMMIT="9e8a28e"
SKIP_Q8=false

# GitHub raw URL base
GITHUB_REPO="edp1096/dgx_spark_init"
GITHUB_BRANCH="main"
GITHUB_RAW="https://raw.githubusercontent.com/${GITHUB_REPO}/${GITHUB_BRANCH}/compose_yaml/ltx-2.3/builder"

# UI Python files to download
UI_FILES=(
    config.py
    app.py
    generators.py
    pipeline_manager.py
    i18n.py
    download_models.py
    convert_fp8.py
    test_pipeline.py
    test_memory_profile.py
)

for arg in "$@"; do
    case $arg in
        --skip-q8) SKIP_Q8=true ;;
    esac
done

echo "=============================================="
echo "  LTX-2.3 Gradio UI Setup"
echo "=============================================="
echo "  LTX-2 source: $LTX_DIR"
echo "  UI files:     $UI_DIR"
echo "  Models:       $MODEL_DIR"
echo ""

# -----------------------------------------------
# 0. Clone LTX-2 if not present
# -----------------------------------------------
if [ ! -d "$LTX_DIR" ]; then
    echo "[0] Cloning LTX-2 (commit $LTX_COMMIT)..."
    git clone https://github.com/Lightricks/LTX-2.git "$LTX_DIR"
    cd "$LTX_DIR"
    git checkout "$LTX_COMMIT"
    cd "$SCRIPT_DIR"
else
    echo "[0] LTX-2 already exists — skipping clone"
    CURRENT=$(cd "$LTX_DIR" && git rev-parse --short HEAD)
    echo "  Current commit: $CURRENT (expected: $LTX_COMMIT)"
fi

# -----------------------------------------------
# 1. Download UI files, patches & apply
# -----------------------------------------------
echo "[1/5] Downloading UI files and applying patches..."
mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$UI_DIR"
mkdir -p "$SCRIPT_DIR/patches"

# Download UI Python files
for fname in "${UI_FILES[@]}"; do
    echo "  Downloading: ui/$fname"
    curl -sfL "${GITHUB_RAW}/ui/${fname}" -o "$UI_DIR/$fname"
done

# Download patch
echo "  Downloading: patches/ltx2-compat.patch"
curl -sfL "${GITHUB_RAW}/patches/ltx2-compat.patch" -o "$SCRIPT_DIR/patches/ltx2-compat.patch"

# Apply patch
PATCH_FILE="$SCRIPT_DIR/patches/ltx2-compat.patch"
if [ -f "$PATCH_FILE" ]; then
    cd "$LTX_DIR"
    if git apply --check "$PATCH_FILE" 2>/dev/null; then
        git apply "$PATCH_FILE"
        echo "  Applied: ltx2-compat.patch"
    else
        echo "  Patch already applied or conflicts — skipping"
    fi
    cd "$SCRIPT_DIR"
fi

# Copy UI files into LTX-2 directory (where packages are importable)
cp "$UI_DIR"/*.py "$LTX_DIR/"
echo "  Copied UI files to LTX-2/"

# -----------------------------------------------
# 2. Install Python packages (NGC torch 보존)
#    WARNING: torch/torchaudio/torchvision은 절대 일반 pip install 금지!
#    반드시 --no-deps 사용. 안 하면 NGC torch가 CPU 버전으로 다운그레이드됨.
# -----------------------------------------------
echo "[2/5] Installing Python packages..."
pip install -q \
    gradio \
    accelerate \
    safetensors \
    huggingface_hub \
    transformers \
    einops \
    scipy \
    psutil \
    fastsafetensors

# torchaudio — 소스 빌드 필수 (PyPI wheel은 CPU-only, NGC torch 다운그레이드 유발)
# --no-deps: torch 의존성 끌어오기 방지
# --no-build-isolation: NGC torch를 직접 참조하여 SM 12.1 CUDA 커널 포함
TORCHAUDIO_SRC="/tmp/torchaudio-src"
if ! python3 -c "import torchaudio" 2>/dev/null; then
    echo "  Building torchaudio from source..."
    if [ ! -d "$TORCHAUDIO_SRC" ]; then
        git clone --depth 250 https://github.com/pytorch/audio.git "$TORCHAUDIO_SRC" && cd "$TORCHAUDIO_SRC" && git checkout 11ed357 && cd "$SCRIPT_DIR"
    fi
    pip install -q "$TORCHAUDIO_SRC" --no-deps --no-build-isolation
fi

# PyAV (av) — 시스템 라이브러리 필요
if ! python3 -c "import av" 2>/dev/null; then
    echo "  Installing system libs for PyAV..."
    apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        ffmpeg libavcodec-dev libavformat-dev libavdevice-dev libavutil-dev \
        libswscale-dev libswresample-dev libavfilter-dev libpostproc-dev \
        sox libsox-dev libsox-fmt-all
    pip install -q av
fi

# -----------------------------------------------
# 3. Install ltx-core and ltx-pipelines
#    --no-deps: NGC의 custom torch(2.11.0a0+nv26.2)를 보존
#    ltx-core가 torch~=2.7을 요구해서 pip이 torch를 다운그레이드함
# -----------------------------------------------
echo "[3/5] Installing ltx-core and ltx-pipelines..."
pip install -q --no-deps -e "$LTX_DIR/packages/ltx-core"
pip install -q --no-deps -e "$LTX_DIR/packages/ltx-pipelines"

# -----------------------------------------------
# 4. Install q8_kernels (native FP8 GEMM)
# -----------------------------------------------
if [ "$SKIP_Q8" = false ]; then
    echo "[4/5] Installing q8_kernels..."
    Q8_WHEEL_URL="${GITHUB_RAW}/q8_kernels-0.0.5-cp312-cp312-linux_aarch64.whl"
    if ! python3 -c "import q8_kernels" 2>/dev/null; then
        echo "  Installing from GitHub: $Q8_WHEEL_URL"
        pip install -q "$Q8_WHEEL_URL"
    else
        echo "  q8_kernels already installed — skipping"
    fi
else
    echo "[4/5] Skipping q8_kernels (--skip-q8)"
fi

# -----------------------------------------------
# 5. Verify
# -----------------------------------------------
echo "[5/5] Verifying installation..."

# Check torch version (should be NGC's 2.11.0a0)
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
echo "  torch: $TORCH_VER"
if [[ "$TORCH_VER" == *"cpu"* ]] || [[ "$TORCH_VER" == "2.10"* ]]; then
    echo "  WARNING: NGC torch may have been overwritten! Container rebuild may be needed."
fi

# Check CUDA
python3 -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}, {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Check models
mkdir -p "$MODEL_DIR"
echo ""
echo "  Model files ($MODEL_DIR):"
MISSING=0
check_file() {
    if [ -e "$MODEL_DIR/$1" ]; then
        echo "    [  OK  ] $1"
    else
        echo "    [MISSING] $1"
        MISSING=$((MISSING + 1))
    fi
}

check_file "ltx-2.3-22b-dev-fp8.safetensors"
check_file "ltx-2.3-22b-distilled-fp8.safetensors"
check_file "ltx-2.3-22b-distilled-lora-384.safetensors"
check_file "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
check_file "gemma-3-12b-it-qat-q4_0-unquantized"

if [ $MISSING -gt 0 ]; then
    echo ""
    echo "  WARNING: $MISSING model file(s) missing!"
    echo "  Run: cd $LTX_DIR && python download_models.py"
fi

# -----------------------------------------------
# Done
# -----------------------------------------------
echo ""
echo "=============================================="
echo "  Setup complete!"
echo "=============================================="
echo ""
echo "  To run UI:   cd $LTX_DIR && PYTHONUNBUFFERED=1 python app.py --server-name 0.0.0.0"
echo "  With log:    cd $LTX_DIR && PYTHONUNBUFFERED=1 nohup python app.py --server-name 0.0.0.0 > $SCRIPT_DIR/logs/gradio.log 2>&1 &"
echo "  To test:     cd $LTX_DIR && python test_pipeline.py --pipeline distilled --fp8 --frames 9"
echo ""
