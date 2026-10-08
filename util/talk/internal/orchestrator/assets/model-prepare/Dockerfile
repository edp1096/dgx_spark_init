FROM python:3.11-slim-bookworm
RUN pip install --no-cache-dir huggingface-hub==0.36.2
COPY --chmod=0644 prepare.py /opt/prepare.py
ENTRYPOINT ["python", "-u", "/opt/prepare.py"]
