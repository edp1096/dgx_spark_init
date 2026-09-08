FROM rust:1.93.1-bookworm AS hwp
RUN apt-get update && apt-get install -y --no-install-recommends python3 && rm -rf /var/lib/apt/lists/*
RUN rustup target add wasm32-unknown-unknown
ENV CARGO_BUILD_JOBS=2 CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
RUN cargo install wasm-bindgen-cli --version 0.2.127 --locked
COPY hwp-engine/ /build/
RUN sh /build/fetch.sh
RUN python3 /build/patch.py
WORKDIR /src
RUN --mount=type=cache,target=/src/target cargo build --locked --release --target wasm32-unknown-unknown --lib -p rhwp && wasm-bindgen --target web --out-dir /out --out-name rhwp target/wasm32-unknown-unknown/release/rhwp.wasm
RUN python3 /build/licenses.py

FROM golang:1.26.5-bookworm AS calc
WORKDIR /src
COPY calc/go.mod calc/go.sum ./
RUN go mod download
COPY calc/ ./
RUN go run ./patch /tmp/excelize-fixed
RUN go test ./...
RUN CGO_ENABLED=0 go build -trimpath -ldflags='-s -w' -o /document-calc .

RUN mkdir -p /licenses && go list -m -f '{{if not .Main}}{{.Path}} {{.Dir}}{{end}}' all | while read module directory; do \
    if [ -n "$directory" ]; then name=$(printf '%s' "$module" | tr / _); mkdir -p "/licenses/$name"; \
    for file in "$directory"/LICENSE* "$directory"/COPYING* "$directory"/NOTICE*; do if [ -f "$file" ]; then cp "$file" "/licenses/$name/"; fi; done; fi; done

FROM node:24-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends fonts-noto-cjk && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY package.json package-lock.json ./
# PptxGenJS text-only output does not use its image-size dependency.
RUN npm ci --omit=dev --ignore-scripts && node -e "require('fs').rmSync('node_modules/image-size', {recursive:true,force:true})" && npm cache clean --force
COPY --from=hwp /out/ /app/node_modules/@rhwp/core/
COPY --from=hwp /hwp-licenses /licenses/rhwp
COPY --from=calc /licenses /licenses/go
COPY THIRD_PARTY_NOTICES.md /licenses/THIRD_PARTY_NOTICES.md
COPY --from=calc /document-calc /usr/local/bin/document-calc
COPY *.mjs ./
RUN node patch-exceljs.mjs && node -e "require('exceljs')" && node license-manifest.mjs
USER node
ENV BIND_ADDRESS=0.0.0.0 PORT=8696
EXPOSE 8696
CMD ["node", "server.mjs"]
