cd ~/play

apt update
apt install -y ffmpeg libavcodec60 libavformat60 libavutil58 libavfilter9 libswscale7 libswresample4

# git clone https://github.com/ace-step/ACE-Step-1.5.git -b v0.1.0-beta.2
git clone https://github.com/ace-step/ACE-Step-1.5.git

cd ACE-Step-1.5

git reset --hard
git pull

sed -i '/torchcodec/d' pyproject.toml
uv add "torchcodec @ https://github.com/edp1096/dgx_spark_init/raw/refs/heads/main/compose_yaml/acestep15/torchcodec-0.10.0a0-cp311-cp311-linux_aarch64.whl"
sed -i 's|"torchcodec>=0.9.1",|"torchcodec @ https://github.com/edp1096/dgx_spark_init/raw/refs/heads/main/compose_yaml/acestep15/torchcodec-0.10.0a0-cp311-cp311-linux_aarch64.whl ; sys_platform == '\''linux'\'' and platform_machine == '\''aarch64'\''",|' pyproject.toml
sed -i '/^\[tool\.hatch\.build\.targets\.wheel\]$/i\[tool.hatch.metadata]\nallow-direct-references = true\n' pyproject.toml
rm -f uv.lock
uv sync

source ~/play/ACE-Step-1.5/.venv/bin/activate
uv pip install https://github.com/edp1096/dgx_spark_init/raw/refs/heads/main/compose_yaml/acestep15/torchcodec-0.10.0a0-cp311-cp311-linux_aarch64.whl
deactivate
# python3 -c "import torchcodec; print(torchcodec.__version__)"

# Those are maybe not needed.
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ENFORCE_EAGER=1

sed -i 's/# SHARE="--share"/SHARE="--share"/' start_gradio_ui.sh
sed -i 's/^# SERVER_NAME="0.0.0.0"/SERVER_NAME="0.0.0.0"/' start_gradio_ui.sh
sed -i 's/enforce_eager=enforce_eager,/enforce_eager=True,/g' acestep/llm_inference.py

./start_gradio_ui.sh