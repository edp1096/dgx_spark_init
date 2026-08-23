#!/bin/bash

# ZIT Gradio UI — Setup Script for NGC PyTorch 26.02-py3 Container
# Clones Z-Image repo, installs dependencies, prepares environment.
#
# Usage:
#   bash ~/zit-ui/setup.sh
#   bash ~/zit-ui/setup.sh --skip-q8
#   bash ~/zit-ui/setup.sh --skip-bnb

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$SCRIPT_DIR/app"
UI_DIR="$APP_DIR/ui"
ZIMAGE_DIR="$SCRIPT_DIR/Z-Image"
ZIMAGE_COMMIT="26f23ed"
MODEL_DIR="${ZIT_MODEL_DIR:-$HOME/.cache/huggingface/hub/zit}"
BNB_COMMIT="925d83e"
SKIP_Q8=false
SKIP_BNB=false

GITHUB_REPO="edp1096/dgx_spark_init"
GITHUB_BRANCH="main"
GITHUB_RAW="https://raw.githubusercontent.com/${GITHUB_REPO}/${GITHUB_BRANCH}/compose_yaml/zit"

APP_FILES=(app.py)
UI_FILES=(zit_config.py pipeline_manager.py worker.py generators.py download_models.py i18n.py face_swap.py)
VIDEOX_FILES=(__init__.py attention_kernel.py attention_utils.py fp8_optimization.py pipeline_z_image_control.py z_image_transformer2d.py z_image_transformer2d_control.py)
PREPROCESSOR_FILES=(__init__.py canny.py depth.py dwpose.py gray.py hed.py)
DWPOSE_FILES=(__init__.py onnxdet.py onnxpose.py wholebody.py util.py)

for arg in "$@"; do
    case $arg in
        --skip-q8)  SKIP_Q8=true ;;
        --skip-bnb) SKIP_BNB=true ;;
    esac
done

echo "=============================================="
echo "  ZIT Gradio UI Setup"
echo "=============================================="
echo "  App:      $APP_DIR"
echo "  Z-Image:  $ZIMAGE_DIR"
echo "  Models:   $MODEL_DIR"
echo ""

# -----------------------------------------------
# 1. Clone Z-Image repo
# -----------------------------------------------
echo "[1/6] Cloning Z-Image repo..."

if [ ! -d "$ZIMAGE_DIR" ]; then
    echo "  Cloning Z-Image (commit $ZIMAGE_COMMIT)..."
    git clone https://github.com/Tongyi-MAI/Z-Image.git "$ZIMAGE_DIR"
    cd "$ZIMAGE_DIR" && git checkout "$ZIMAGE_COMMIT" && cd "$SCRIPT_DIR"
else
    echo "  Z-Image already exists — skipping"
fi

# -----------------------------------------------
# 2. Install Z-Image (editable, no deps)
# -----------------------------------------------
echo "[2/6] Installing Z-Image (editable, --no-deps)..."
cd "$ZIMAGE_DIR" && pip install -q -e . --no-deps
cd "$SCRIPT_DIR"

# -----------------------------------------------
# 3. Download UI files
# -----------------------------------------------
echo "[3/6] Downloading UI files..."
mkdir -p "$APP_DIR" "$UI_DIR" "$UI_DIR/videox_models" "$UI_DIR/preprocessors/dwpose_utils" "$SCRIPT_DIR/logs"

for fname in "${APP_FILES[@]}"; do
    echo "  app/$fname"
    curl -sfL "${GITHUB_RAW}/app/${fname}" -o "$APP_DIR/$fname"
done

for fname in "${UI_FILES[@]}"; do
    echo "  app/ui/$fname"
    curl -sfL "${GITHUB_RAW}/app/ui/${fname}" -o "$UI_DIR/$fname"
done

for fname in "${VIDEOX_FILES[@]}"; do
    echo "  videox_models/$fname"
    curl -sfL "${GITHUB_RAW}/app/ui/videox_models/${fname}" -o "$UI_DIR/videox_models/$fname"
done

for fname in "${PREPROCESSOR_FILES[@]}"; do
    echo "  preprocessors/$fname"
    curl -sfL "${GITHUB_RAW}/app/ui/preprocessors/${fname}" -o "$UI_DIR/preprocessors/$fname"
done

for fname in "${DWPOSE_FILES[@]}"; do
    echo "  preprocessors/dwpose_utils/$fname"
    curl -sfL "${GITHUB_RAW}/app/ui/preprocessors/dwpose_utils/${fname}" -o "$UI_DIR/preprocessors/dwpose_utils/$fname"
done

echo "  Note: ZoeDepth zoe/ directory must be copied manually or via update.sh"

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
    peft \
    timm \
    diffusers \
    sentencepiece \
    opencv-python-headless

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
python3 -c "import timm; print(f'  timm: {timm.__version__}')"

echo ""
echo "=============================================="
echo "  Setup complete!"
echo "=============================================="
echo ""
echo "  To run UI:   cd $SCRIPT_DIR && PYTHONUNBUFFERED=1 python $APP_DIR/app.py --server-name 0.0.0.0 --port 7862"
echo "  To download: cd $UI_DIR && python download_models.py"
echo ""
