FROM nvcr.io/nvidia/pytorch:26.07-py3

ARG DIFFUSERS_COMMIT=77c13f8a49c457f10a89862bb129bdc930b88929

ENV PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    API_PORT=8707

RUN pip install --no-cache-dir \
      "git+https://github.com/huggingface/diffusers.git@${DIFFUSERS_COMMIT}" \
      transformers accelerate sentencepiece protobuf \
      bitsandbytes==0.50.2 \
      fastapi uvicorn python-multipart

WORKDIR /opt/sana-api
COPY api.py editing.py start.sh ./
RUN chmod +x start.sh

EXPOSE 8707
ENTRYPOINT ["/opt/sana-api/start.sh"]
