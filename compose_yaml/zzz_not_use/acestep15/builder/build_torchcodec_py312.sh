apt update
apt install software-properties-common -y
add-apt-repository ppa:deadsnakes/ppa -y
apt update
apt install -y python3.11 python3.11-venv python3.11-dev

# Clean and clone torchcodec
cd /tmp
rm -rf torchcodec
git clone https://github.com/pytorch/torchcodec.git
cd torchcodec
git checkout v0.10.0

# Build configuration
export ENABLE_CUDA=1
export BUILD_AGAINST_ALL_FFMPEG_FROM_S3=1
export I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1

# Install build dependencies
pip3 install build pybind11 --break-system-packages



# Build wheel
python3 -m build --wheel --no-isolation

# Bundle FFmpeg libraries into wheel
cd dist
mkdir temp && cd temp

apt install -y zip
unzip ../torchcodec-*.whl

mkdir -p torchcodec/.libs
cp /usr/lib/aarch64-linux-gnu/libavcodec.so.60* torchcodec/.libs/
cp /usr/lib/aarch64-linux-gnu/libavformat.so.60* torchcodec/.libs/
cp /usr/lib/aarch64-linux-gnu/libavutil.so.58* torchcodec/.libs/
cp /usr/lib/aarch64-linux-gnu/libavfilter.so.9* torchcodec/.libs/
cp /usr/lib/aarch64-linux-gnu/libswscale.so.7* torchcodec/.libs/
cp /usr/lib/aarch64-linux-gnu/libswresample.so.4* torchcodec/.libs/

rm ../torchcodec-*.whl
zip -r ../torchcodec-0.10.0a0-cp312-cp312-linux_aarch64.whl *
cd ..
rm -rf temp

ls -lh torchcodec-*.whl

# Install and test
pip3 install torchcodec-*.whl --force-reinstall --break-system-packages
python3 -c "import torchcodec; print(torchcodec.__version__)"