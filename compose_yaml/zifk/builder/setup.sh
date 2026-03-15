#!/bin/bash

# ZIFK Gradio UI — Setup Script for NGC PyTorch 26.02-py3 Container
# Clones Z-Image and flux2 repos, installs dependencies, prepares environment.
#
# Usage:
#   bash ~/zifk-ui/setup.sh
#   bash ~/zifk-ui/setup.sh --skip-q8
#   bash ~/zifk-ui/setup.sh --skip-bnb

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$SCRIPT_DIR/app"
UI_DIR="$APP_DIR/ui"
ZIMAGE_DIR="$SCRIPT_DIR/Z-Image"
FLUX2_DIR="$SCRIPT_DIR/flux2"
MODEL_DIR="${ZIFK_MODEL_DIR:-$HOME/.cache/huggingface/hub/zifk}"
BNB_COMMIT="925d83e"
SKIP_Q8=false
SKIP_BNB=false

GITHUB_REPO="edp1096/dgx_spark_init"
GITHUB_BRANCH="main"
GITHUB_RAW="https://raw.githubusercontent.com/${GITHUB_REPO}/${GITHUB_BRANCH}/compose_yaml/zifk"

APP_FILES=(app.py)
UI_FILES=(zifk_config.py pipeline_manager.py worker.py generators.py download_models.py)

for arg in "$@"; do
    case $arg in
        --skip-q8)  SKIP_Q8=true ;;
        --skip-bnb) SKIP_BNB=true ;;
    esac
done

echo "=============================================="
echo "  ZIFK Gradio UI Setup"
echo "=============================================="
echo "  App:      $APP_DIR"
echo "  Z-Image:  $ZIMAGE_DIR"
echo "  flux2:    $FLUX2_DIR"
echo "  Models:   $MODEL_DIR"
echo ""

# -----------------------------------------------
# 1. Clone source repos
# -----------------------------------------------
echo "[1/6] Cloning source repos..."

if [ ! -d "$ZIMAGE_DIR" ]; then
    echo "  Cloning Z-Image..."
    git clone https://github.com/Tongyi-MAI/Z-Image.git "$ZIMAGE_DIR"
else
    echo "  Z-Image already exists — skipping"
fi

if [ ! -d "$FLUX2_DIR" ]; then
    echo "  Cloning flux2..."
    git clone https://github.com/black-forest-labs/flux2.git "$FLUX2_DIR"
else
    echo "  flux2 already exists — skipping"
fi

# -----------------------------------------------
# 2. Install source repos (editable, no deps)
# -----------------------------------------------
echo "[2/6] Installing source repos (editable, --no-deps)..."
cd "$ZIMAGE_DIR" && pip install -q -e . --no-deps
cd "$FLUX2_DIR" && pip install -q -e . --no-deps
cd "$SCRIPT_DIR"

# -----------------------------------------------
# 3. Download UI files
# -----------------------------------------------
echo "[3/6] Downloading UI files..."
mkdir -p "$APP_DIR" "$UI_DIR" "$SCRIPT_DIR/logs"

for fname in "${APP_FILES[@]}"; do
    echo "  Downloading: app/$fname"
    curl -sfL "${GITHUB_RAW}/app/${fname}" -o "$APP_DIR/$fname"
done

for fname in "${UI_FILES[@]}"; do
    echo "  Downloading: app/ui/$fname"
    curl -sfL "${GITHUB_RAW}/app/ui/${fname}" -o "$UI_DIR/$fname"
done

# -----------------------------------------------
# 4. Install Python packages
# -----------------------------------------------
echo "[4/6] Installing Python packages..."
pip install -q \
    gradio \
    accelerate \
    safetensors \
    huggingface_hub \
    psutil \
    loguru \
    einops \
    fire \
    transformers \
    peft

# -----------------------------------------------
# 5. Install q8_kernels
# -----------------------------------------------
if [ "$SKIP_Q8" = false ]; then
    echo "[5/6] Installing q8_kernels..."
    Q8_WHEEL_URL="https://raw.githubusercontent.com/${GITHUB_REPO}/${GITHUB_BRANCH}/compose_yaml/ltx-2.3/builder/q8_kernels-0.0.5-cp312-cp312-linux_aarch64.whl"
    if ! python3 -c "import q8_kernels" 2>/dev/null; then
        pip install -q "$Q8_WHEEL_URL"
    else
        echo "  q8_kernels already installed"
    fi
else
    echo "[5/6] Skipping q8_kernels (--skip-q8)"
fi

# -----------------------------------------------
# 6. Install bitsandbytes
# -----------------------------------------------
if [ "$SKIP_BNB" = false ]; then
    echo "[6/6] Installing bitsandbytes from source..."
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
    echo "[6/6] Skipping bitsandbytes (--skip-bnb)"
fi

# -----------------------------------------------
# Verify
# -----------------------------------------------
echo ""
echo "Verifying installation..."
python3 -c "import torch; print(f'  torch: {torch.__version__}')"
python3 -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}, {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
python3 -c "import gradio; print(f'  gradio: {gradio.__version__}')"
python3 -c "import zimage; print('  zimage: ok')"
python3 -c "import flux2; print('  flux2: ok')"

echo ""
echo "=============================================="
echo "  Setup complete!"
echo "=============================================="
echo ""
echo "  To run UI:   cd $SCRIPT_DIR && PYTHONUNBUFFERED=1 python $APP_DIR/app.py --server-name 0.0.0.0 --port 7861"
echo "  To download: cd $UI_DIR && python download_models.py"
echo ""
