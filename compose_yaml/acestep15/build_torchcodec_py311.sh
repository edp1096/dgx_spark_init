#!/bin/bash
set -e

python3.11 --version

# Build torchcodec with system Python 3.11
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
WHEEL=$(ls torchcodec-*.whl)

mkdir temp && cd temp
unzip ../$WHEEL

mkdir -p torchcodec/.libs
cp /usr/lib/aarch64-linux-gnu/libavcodec.so.* torchcodec/.libs/
cp /usr/lib/aarch64-linux-gnu/libavformat.so.* torchcodec/.libs/
cp /usr/lib/aarch64-linux-gnu/libavutil.so.* torchcodec/.libs/
cp /usr/lib/aarch64-linux-gnu/libavfilter.so.* torchcodec/.libs/
cp /usr/lib/aarch64-linux-gnu/libswscale.so.* torchcodec/.libs/
cp /usr/lib/aarch64-linux-gnu/libswresample.so.* torchcodec/.libs/

rm ../$WHEEL
zip -r ../$WHEEL *
cd ..
rm -rf temp

WHEEL_PATH=$(pwd)/$WHEEL
echo "================================"
echo "Wheel: $WHEEL_PATH"
ls -lh $WHEEL
echo "================================"

# Install to ACE-Step using uv
deactivate
cd /root/play/ACE-Step-1.5
uv pip install $WHEEL_PATH

# Test
uv run python -c "import torchcodec; print(torchcodec.__version__)"