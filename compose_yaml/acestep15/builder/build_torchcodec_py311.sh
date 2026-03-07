#!/bin/bash
set -e

python3.11 --version

cd /tmp
rm -rf build_venv
python3.11 -m venv build_venv
source build_venv/bin/activate

pip install --upgrade pip
pip install torch==2.10.0+cu128 torchaudio==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install build pybind11

rm -rf torchcodec
git clone https://github.com/pytorch/torchcodec.git
cd torchcodec
git checkout v0.10.0

export ENABLE_CUDA=1
export BUILD_AGAINST_ALL_FFMPEG_FROM_S3=1
export I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1
export CMAKE_BUILD_PARALLEL_LEVEL=8

python -m build --wheel --no-isolation

cd dist
ORIGINAL_WHEEL=$(ls torchcodec-*.whl)
echo "Original wheel: $ORIGINAL_WHEEL"

# Unpack wheel
mkdir temp && cd temp
unzip ../$ORIGINAL_WHEEL

# Add FFmpeg libraries
mkdir -p torchcodec/.libs
cp /usr/lib/aarch64-linux-gnu/libavcodec.so.* torchcodec/.libs/
cp /usr/lib/aarch64-linux-gnu/libavformat.so.* torchcodec/.libs/
cp /usr/lib/aarch64-linux-gnu/libavutil.so.* torchcodec/.libs/
cp /usr/lib/aarch64-linux-gnu/libavfilter.so.* torchcodec/.libs/
cp /usr/lib/aarch64-linux-gnu/libswscale.so.* torchcodec/.libs/
cp /usr/lib/aarch64-linux-gnu/libswresample.so.* torchcodec/.libs/

# Verify .dist-info exists
ls -la | grep dist-info
if [ ! -d "torchcodec-"*".dist-info" ]; then
    echo "ERROR: .dist-info directory not found!"
    exit 1
fi

# Repack with ALL files (including .dist-info)
cd ..
rm $ORIGINAL_WHEEL
cd temp
zip -r ../$ORIGINAL_WHEEL * -i '*'
cd ..

# Verify repacked wheel
unzip -l $ORIGINAL_WHEEL | grep dist-info
rm -rf temp

WHEEL_PATH=$(pwd)/$ORIGINAL_WHEEL
echo "================================"
echo "Final wheel: $WHEEL_PATH"
ls -lh $ORIGINAL_WHEEL
echo "================================"

# Install to ACE-Step
deactivate
cd /root/play/ACE-Step-1.5
uv pip install $WHEEL_PATH --force-reinstall

# Test
uv run python -c "import torchcodec; print(f'torchcodec {torchcodec.__version__}')"