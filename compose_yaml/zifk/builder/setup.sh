#!/bin/bash

# Z-Image Gradio UI — Setup Script for NGC PyTorch 26.02-py3 Container
#
# Usage:
#   bash ~/zimage-ui/setup.sh
#   bash ~/zimage-ui/setup.sh --skip-q8
#   bash ~/zimage-ui/setup.sh --skip-bnb

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UI_DIR="$SCRIPT_DIR/ui"
MODEL_DIR="${ZIMAGE_MODEL_DIR:-$HOME/.cache/huggingface/hub/zimage}"
BNB_COMMIT="925d83e"
SKIP_Q8=false
SKIP_BNB=false

GITHUB_REPO="edp1096/dgx_spark_init"
GITHUB_BRANCH="main"
GITHUB_RAW="https://raw.githubusercontent.com/${GITHUB_REPO}/${GITHUB_BRANCH}/compose_yaml/z-image/builder"

UI_FILES=(
    config.py
    app.py
    generators.py
    worker.py
    pipeline_manager.py
    download_models.py
)

for arg in "$@"; do
    case $arg in
        --skip-q8)  SKIP_Q8=true ;;
        --skip-bnb) SKIP_BNB=true ;;
    esac
done

echo "=============================================="
echo "  Z-Image Gradio UI Setup"
echo "=============================================="
echo "  UI files:  $UI_DIR"
echo "  Models:    $MODEL_DIR"
echo ""

# -----------------------------------------------
# 1. Download UI files
# -----------------------------------------------
echo "[1/5] Downloading UI files..."
mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$UI_DIR"

for fname in "${UI_FILES[@]}"; do
    echo "  Downloading: ui/$fname"
    curl -sfL "${GITHUB_RAW}/ui/${fname}" -o "$UI_DIR/$fname"
done

cp "$UI_DIR"/*.py "$SCRIPT_DIR/"
echo "  Copied UI files to $SCRIPT_DIR/"

# -----------------------------------------------
# 2. Install Python packages
# -----------------------------------------------
echo "[2/5] Installing Python packages..."
pip install -q \
    gradio \
    accelerate \
    safetensors \
    huggingface_hub \
    psutil \
    loguru

pip install -q git+https://github.com/huggingface/diffusers
pip install -q transformers peft

# -----------------------------------------------
# 3. Install q8_kernels
# -----------------------------------------------
if [ "$SKIP_Q8" = false ]; then
    echo "[3/5] Installing q8_kernels..."
    Q8_WHEEL_URL="https://raw.githubusercontent.com/${GITHUB_REPO}/${GITHUB_BRANCH}/compose_yaml/ltx-2.3/builder/q8_kernels-0.0.5-cp312-cp312-linux_aarch64.whl"
    if ! python3 -c "import q8_kernels" 2>/dev/null; then
        pip install -q "$Q8_WHEEL_URL"
    else
        echo "  q8_kernels already installed"
    fi
else
    echo "[3/5] Skipping q8_kernels (--skip-q8)"
fi

# -----------------------------------------------
# 4. Install bitsandbytes
# -----------------------------------------------
if [ "$SKIP_BNB" = false ]; then
    echo "[4/5] Installing bitsandbytes from source..."
    if ! python3 -c "import bitsandbytes" 2>/dev/null; then
        BNB_SRC="/tmp/bitsandbytes-src"
        if [ ! -d "$BNB_SRC" ]; then
            git clone --depth 250 https://github.com/bitsandbytes-foundation/bitsandbytes.git "$BNB_SRC" && cd "$BNB_SRC" && git checkout "$BNB_COMMIT" && cd "$SCRIPT_DIR"
        fi
        cd "$BNB_SRC"
        cmake -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY="121" -S . -B build
        cmake --build build -j$(nproc)
        pip install -q .
        cd "$SCRIPT_DIR"
        echo "  bitsandbytes installed from source"
    else
        echo "  bitsandbytes already installed"
    fi
else
    echo "[4/5] Skipping bitsandbytes (--skip-bnb)"
fi

# -----------------------------------------------
# 5. Verify
# -----------------------------------------------
echo "[5/5] Verifying installation..."
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
echo "  torch: $TORCH_VER"
python3 -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}, {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
python3 -c "import gradio; print(f'  gradio: {gradio.__version__}')"
python3 -c "import diffusers; print(f'  diffusers: {diffusers.__version__}')"

echo ""
echo "=============================================="
echo "  Setup complete!"
echo "=============================================="
echo ""
echo "  To run UI:   cd $SCRIPT_DIR && PYTHONUNBUFFERED=1 python app.py --server-name 0.0.0.0 --port 7861"
echo "  To download: cd $SCRIPT_DIR && python download_models.py"
echo ""
