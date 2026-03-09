#!/bin/bash
# LTX-2.3 Gradio UI — Setup Script for NGC PyTorch 26.02-py3 Container
# Run this after `docker compose up` to install dependencies.
#
# IMPORTANT: Uses --no-deps for ltx packages to preserve NGC's custom torch.
#
# Usage:
#   bash /root/ltx2-ui/setup.sh            # Full setup
#   bash /root/ltx2-ui/setup.sh --skip-q8  # Skip q8_kernels

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LTX_DIR="$SCRIPT_DIR/LTX-2"
MODEL_DIR="$SCRIPT_DIR/models/ltx-2.3"
SKIP_Q8=false

for arg in "$@"; do
    case $arg in
        --skip-q8) SKIP_Q8=true ;;
    esac
done

echo "=============================================="
echo "  LTX-2.3 Gradio UI Setup"
echo "=============================================="
echo "  LTX-2 source: $LTX_DIR"
echo "  Models:       $MODEL_DIR"
echo ""

# -----------------------------------------------
# 1. Install Python packages (NGC torch 보존)
# -----------------------------------------------
echo "[1/4] Installing Python packages..."
pip install -q \
    gradio \
    accelerate \
    safetensors \
    huggingface_hub \
    transformers \
    einops \
    scipy

# -----------------------------------------------
# 2. Install ltx-core and ltx-pipelines
#    --no-deps: NGC의 custom torch(2.11.0a0+nv26.2)를 보존
#    ltx-core가 torch~=2.7을 요구해서 pip이 torch를 다운그레이드함
# -----------------------------------------------
echo "[2/4] Installing ltx-core and ltx-pipelines..."
pip install -q --no-deps -e "$LTX_DIR/packages/ltx-core"
pip install -q --no-deps -e "$LTX_DIR/packages/ltx-pipelines"

# -----------------------------------------------
# 3. Install q8_kernels (native FP8 GEMM)
# -----------------------------------------------
if [ "$SKIP_Q8" = false ]; then
    echo "[3/4] Installing q8_kernels..."
    Q8_WHEEL="$SCRIPT_DIR/q8_kernels-0.0.5-cp312-cp312-linux_aarch64.whl"
    if [ -f "$Q8_WHEEL" ]; then
        echo "  Using pre-built wheel: $Q8_WHEEL"
        pip install -q "$Q8_WHEEL"
    else
        echo "  No pre-built wheel found. Building from source (~5min)..."
        Q8_SRC="/tmp/LTX-Video-Q8-Kernels"
        if [ ! -d "$Q8_SRC" ]; then
            git clone --recursive https://github.com/Lightricks/LTX-Video-Q8-Kernels.git "$Q8_SRC"
        fi
        pip install -q "$Q8_SRC"
    fi
else
    echo "[3/4] Skipping q8_kernels (--skip-q8)"
fi

# -----------------------------------------------
# 4. Verify
# -----------------------------------------------
echo "[4/4] Verifying installation..."

# Check torch version (should be NGC's 2.11.0a0)
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
echo "  torch: $TORCH_VER"
if [[ "$TORCH_VER" == *"cpu"* ]] || [[ "$TORCH_VER" == "2.10"* ]]; then
    echo "  WARNING: NGC torch may have been overwritten! Container rebuild may be needed."
fi

# Check CUDA
python3 -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}, {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Check models
echo ""
echo "  Model files:"
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
echo "  To run UI:   cd $LTX_DIR && python app.py"
echo "  To test:     cd $LTX_DIR && python test_pipeline.py --pipeline distilled --fp8 --frames 9"
echo ""
