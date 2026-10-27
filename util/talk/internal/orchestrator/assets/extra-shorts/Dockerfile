FROM node:24-bookworm-slim
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HYPERFRAMES_BROWSER_PATH=/usr/bin/chromium \
    PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium PRODUCER_LOW_MEMORY_MODE=true \
    HF_TELEMETRY_DISABLED=1 HYPERFRAMES_TELEMETRY_DISABLED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-dev python3-venv \
    ffmpeg chromium fonts-noto-cjk fonts-noto-core libcairo2-dev libpango1.0-dev \
    pkg-config g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/shorts
COPY requirements.txt package.json package-lock.json ./
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt \
    && npm ci --omit=dev --no-audit --no-fund
ENV PATH=/opt/venv/bin:/opt/shorts/node_modules/.bin:$PATH
COPY . /opt/shorts
RUN mkdir -p /data /voices && chown -R 1000:1000 /data /voices /opt/shorts
USER 1000:1000
CMD ["python3", "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8725"]
