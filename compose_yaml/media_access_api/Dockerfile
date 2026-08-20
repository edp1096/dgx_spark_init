ARG POT_PROVIDER_VERSION=1.3.1
ARG POT_PROVIDER_DIGEST=sha256:bea3cfda79245700d7ad90500052b4358b1c1828fdc0961929624b83933121bc
ARG BROWSEFORGE_VERSION=v2.1.12
ARG BROWSEFORGE_DIGEST=sha256:3f76a2a7d90d41d5c6e05cf75f7274e62fffe326c716b723b8da79acb8731f12
ARG PLAYWRIGHT_VERSION=1.62.0
ARG PLAYWRIGHT_DIGEST=sha256:aa81288e738725378becba5b3e06cb0f3a7f012a610e87e8d767a090ea3f740d
ARG YTDLP_VERSION=2026.7.4
ARG CAMOUFOX_VERSION=0.5.5
ARG PLAYWRIGHT_CAPTCHA_VERSION=0.1.5
ARG CAMOUFOX_BROWSER_VERSION=152.0.4-beta.28
ARG CAMOUFOX_BROWSER_SHA256=3a105a2fc929e80a79b4b7fce2c93ed62c4fb2c877f3c1ed2a5d66a1c4fe968f

FROM brainicism/bgutil-ytdlp-pot-provider:${POT_PROVIDER_VERSION}-deno@${POT_PROVIDER_DIGEST} AS pot-provider
FROM ghcr.io/nczz/browseforge:${BROWSEFORGE_VERSION}@${BROWSEFORGE_DIGEST} AS browseforge

FROM mcr.microsoft.com/playwright/python:v${PLAYWRIGHT_VERSION}-noble@${PLAYWRIGHT_DIGEST}

LABEL org.opencontainers.image.source="https://github.com/edp1096/dgx_spark_init"

ARG POT_PROVIDER_VERSION
ARG PLAYWRIGHT_VERSION
ARG YTDLP_VERSION
ARG CAMOUFOX_VERSION
ARG PLAYWRIGHT_CAPTCHA_VERSION
ARG CAMOUFOX_BROWSER_VERSION
ARG CAMOUFOX_BROWSER_SHA256

COPY --from=pot-provider /usr/bin/deno /usr/local/bin/deno
COPY --from=pot-provider /app /opt/bgutil
COPY --from=pot-provider /usr/share/common-licenses/GPL-3 /usr/share/licenses/bgutil-ytdlp-pot-provider/GPL-3
COPY --from=browseforge /app/BrowseForge /opt/browseforge/BrowseForge
COPY --from=browseforge /opt/browseforge/browsers/browseforge-chromium /opt/browseforge/browsers/browseforge-chromium
COPY --from=browseforge /root/.cache/ms-playwright-go /root/.cache/ms-playwright-go

ENV DEBIAN_FRONTEND=noninteractive \
    DENO_DIR=/opt/bgutil/.cache/deno \
    DENO_NO_PROMPT=1 \
    DENO_NO_UPDATE_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Upstream 1.3.1 binds its HTTP helper to every interface and does not expose a
# host option. It is an implementation detail of this image, so keep it on the
# container loopback interface beside BrowseForge.
RUN sed -i \
    -e 's/host: "::"/host: "127.0.0.1"/' \
    -e 's/host: "0.0.0.0"/host: "127.0.0.1"/' \
    /opt/bgutil/src/main.ts

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl ffmpeg python3-venv tesseract-ocr unzip xclip xdotool \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/camoufox-venv \
    && /opt/camoufox-venv/bin/pip install --no-cache-dir \
        "camoufox==${CAMOUFOX_VERSION}" \
        "playwright-captcha==${PLAYWRIGHT_CAPTCHA_VERSION}" \
        "playwright==1.60.0" \
    && python -m pip install --no-cache-dir \
        fastapi==0.136.3 \
        bgutil-ytdlp-pot-provider==${POT_PROVIDER_VERSION} \
        playwright==${PLAYWRIGHT_VERSION} \
        python-multipart==0.0.32 \
        uvicorn==0.51.0 \
        "yt-dlp[default,curl-cffi]==${YTDLP_VERSION}"

RUN mkdir -p /opt/direct-chromium \
    && ln -s "$(python -c 'from playwright.sync_api import sync_playwright; p=sync_playwright().start(); print(p.chromium.executable_path); p.stop()')" \
        /opt/direct-chromium/chrome

RUN curl -fL --retry 3 \
        "https://github.com/daijro/camoufox/releases/download/v${CAMOUFOX_BROWSER_VERSION}/camoufox-${CAMOUFOX_BROWSER_VERSION}-lin.arm64.zip" \
        -o /tmp/camoufox.zip \
    && echo "${CAMOUFOX_BROWSER_SHA256}  /tmp/camoufox.zip" | sha256sum -c - \
    && mkdir -p /opt/browseforge/browsers/camoufox \
    && unzip -q /tmp/camoufox.zip -d /opt/browseforge/browsers/camoufox \
    && rm /tmp/camoufox.zip \
    && mkdir -p /usr/share/licenses/camoufox /usr/share/licenses/playwright-captcha \
    && cp /usr/share/common-licenses/MPL-2.0 /usr/share/licenses/camoufox/MPL-2.0 \
    && cp /opt/camoufox-venv/lib/python3.12/site-packages/playwright_captcha-*.dist-info/licenses/LICENSE \
        /usr/share/licenses/playwright-captcha/LICENSE

WORKDIR /app
COPY BROWSEFORGE_LICENSE /usr/share/licenses/browseforge/LICENSE
COPY api.py /app/api.py
COPY attached_chromium.py /app/attached_chromium.py
COPY browseforge_client.py /app/browseforge_client.py
COPY camoufox_client.py /app/camoufox_client.py
COPY camoufox_worker.py /app/camoufox_worker.py
COPY direct_camoufox_client.py /app/direct_camoufox_client.py
COPY direct_camoufox_worker.py /app/direct_camoufox_worker.py
COPY site_adapters.py /app/site_adapters.py
COPY browseforge-config.json /app/browseforge-config.json
COPY entrypoint.sh /app/entrypoint.sh

RUN chmod 0755 /app/entrypoint.sh /opt/browseforge/BrowseForge \
    /opt/browseforge/browsers/browseforge-chromium/chrome \
    /opt/browseforge/browsers/camoufox/camoufox \
    /app/camoufox_worker.py /app/direct_camoufox_worker.py

EXPOSE 8697
ENTRYPOINT ["/app/entrypoint.sh"]
