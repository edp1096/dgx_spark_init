FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
WORKDIR /app
# Preserve the NVIDIA CUDA PyTorch build; install only the embedding runtime.
RUN python -m pip install --no-cache-dir sentence-transformers==6.1.0 transformers==5.19.0 huggingface-hub==1.33.0 hf-xet==1.7.0 \
    && python -c "import torch; assert torch.version.cuda is not None; from transformers import EmbeddingGemma2Model"
COPY server.py prepare.py ./
ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
CMD ["python", "/app/server.py"]
