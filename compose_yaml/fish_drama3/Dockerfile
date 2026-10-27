ARG FISH_BASE_IMAGE=ghcr.io/edp1096/sglang-qwen38-qad:sm121-v5@sha256:c79141f03a89b8888920f629ac638a6915eec5abb0dabe75688977c3c58b75f5
FROM ${FISH_BASE_IMAGE}
ENV PYTHONPATH=""
RUN git clone https://github.com/sgl-project/sglang-omni.git /opt/fish-omni
WORKDIR /opt/fish-omni
RUN git checkout 1240a6993c8073593aa0ad074859ccee2797adff
RUN pip install --no-cache-dir --no-deps sglang==0.5.21 sglang-kernel==0.4.7 flashinfer-python==0.6.18 descript-audiotools==0.7.2 descript-audio-codec==1.0.0 \
 && pip install --no-cache-dir --no-deps flashinfer-cubin==0.6.18 --index-url https://flashinfer.ai/whl \
 && pip install --no-cache-dir --no-deps flashinfer-jit-cache==0.6.18+cu130 --index-url https://flashinfer.ai/whl/cu130 \
 && pip install --no-cache-dir accelerate==1.15.0 hydra-core==1.3.7 librosa==1.0.0 soundfile==0.13.1 pyloudnorm==0.2.0 julius==0.2.8 argbind==0.3.9 flatten-dict==0.5.0 markdown2==2.5.5 randomname==0.2.1 ffmpy==1.0.0 pystoi==0.4.1 tensorboard==2.21.0 torch-stoi==0.2.3 silero-vad==6.2.3 onnxruntime==1.31.0 importlib-resources==7.1.0 transformers==5.12.1 tokenizers==0.22.2 \
 && pip install --no-cache-dir --no-deps -e .
COPY patches/ sglang_omni/models/fishaudio_drama3/
WORKDIR /opt/fish
COPY api.py speech.py voice_catalog.py config.json reference.wav reference-uncle-fu.wav ./
COPY voices/ ./voices/
ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_HOME=/hf DRAMA3_HADAMARD8=1 OMP_NUM_THREADS=8
ENTRYPOINT ["python", "api.py"]
