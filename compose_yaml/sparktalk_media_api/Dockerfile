ARG RUNTIME_BASE_IMAGE=ubuntu:24.04
FROM ${RUNTIME_BASE_IMAGE} AS builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends golang-go \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY go.mod ./
COPY *.go ./
RUN CGO_ENABLED=0 go test ./... \
    && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/sparktalk-media-api .

FROM ${RUNTIME_BASE_IMAGE} AS tools
ARG TARGETARCH
ARG YTDLP_VERSION=2026.07.04
ARG DENO_VERSION=2.8.1
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl unzip \
    && rm -rf /var/lib/apt/lists/* \
    && case "${TARGETARCH}" in \
         amd64) ytdlp_asset=yt-dlp_linux; deno_asset=deno-x86_64-unknown-linux-gnu.zip ;; \
         arm64) ytdlp_asset=yt-dlp_linux_aarch64; deno_asset=deno-aarch64-unknown-linux-gnu.zip ;; \
         *) echo "unsupported architecture: ${TARGETARCH}" >&2; exit 1 ;; \
       esac \
    && curl -fsSLo /tmp/yt-dlp "https://github.com/yt-dlp/yt-dlp/releases/download/${YTDLP_VERSION}/${ytdlp_asset}" \
    && curl -fsSLo /tmp/SHA2-256SUMS "https://github.com/yt-dlp/yt-dlp/releases/download/${YTDLP_VERSION}/SHA2-256SUMS" \
    && ytdlp_sha="$(awk -v asset="${ytdlp_asset}" '$2 == asset {print $1}' /tmp/SHA2-256SUMS)" \
    && test -n "${ytdlp_sha}" \
    && echo "${ytdlp_sha}  /tmp/yt-dlp" | sha256sum -c - \
    && chmod 0555 /tmp/yt-dlp \
    && curl -fsSLo "/tmp/${deno_asset}" "https://github.com/denoland/deno/releases/download/v${DENO_VERSION}/${deno_asset}" \
    && curl -fsSLo "/tmp/${deno_asset}.sha256sum" "https://github.com/denoland/deno/releases/download/v${DENO_VERSION}/${deno_asset}.sha256sum" \
    && deno_sha="$(awk '{print $1}' "/tmp/${deno_asset}.sha256sum")" \
    && test -n "${deno_sha}" \
    && echo "${deno_sha}  /tmp/${deno_asset}" | sha256sum -c - \
    && unzip -q "/tmp/${deno_asset}" -d /tmp/deno-bin \
    && chmod 0555 /tmp/deno-bin/deno

FROM ${RUNTIME_BASE_IMAGE}

LABEL org.opencontainers.image.source="https://github.com/edp1096/dgx_spark_init"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates ffmpeg tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /out/sparktalk-media-api /usr/local/bin/sparktalk-media-api
COPY --from=tools /tmp/yt-dlp /usr/local/bin/yt-dlp
COPY --from=tools /tmp/deno-bin/deno /usr/local/bin/deno

EXPOSE 8698
ENTRYPOINT ["/usr/local/bin/sparktalk-media-api"]
