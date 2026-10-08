#!/usr/bin/env bash
set -Eeuo pipefail

readonly browseforge_root=/data/browseforge
readonly display_number=:99
readonly direct_display_number=:98
readonly pot_url=http://127.0.0.1:4416

mkdir -p \
  "$browseforge_root/profiles" \
  "$browseforge_root/data" \
  "$browseforge_root/logs"
find "$browseforge_root/profiles" -type f \
  \( -name SingletonLock -o -name SingletonCookie -o -name SingletonSocket \) \
  -delete 2>/dev/null || true
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99 /tmp/.X98-lock /tmp/.X11-unix/X98

export DISPLAY="$display_number"
export LIBGL_ALWAYS_SOFTWARE=1
Xvfb "$display_number" -screen 0 1920x1080x24 -nolisten tcp +extension GLX +render &
xvfb_pid=$!
Xvfb "$direct_display_number" -screen 0 1365x768x24 -nolisten tcp +extension GLX +render &
direct_xvfb_pid=$!

(
  cd /opt/bgutil
  exec deno run \
    --allow-env \
    --allow-net \
    --allow-ffi=/opt/bgutil/node_modules \
    --allow-read=/opt/bgutil/node_modules \
    /opt/bgutil/src/main.ts
) &
pot_pid=$!

/opt/browseforge/BrowseForge \
  --base-dir "$browseforge_root" \
  --config /app/browseforge-config.json \
  serve --host 127.0.0.1 --no-sandbox --no-open &
browseforge_pid=$!

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  kill "${api_pid:-}" "$browseforge_pid" "$pot_pid" "$xvfb_pid" "$direct_xvfb_pid" 2>/dev/null || true
  wait "${api_pid:-}" "$browseforge_pid" "$pot_pid" "$xvfb_pid" "$direct_xvfb_pid" 2>/dev/null || true
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 90); do
  if ! kill -0 "$pot_pid" 2>/dev/null; then
    echo "PO Token provider exited before becoming ready" >&2
    exit 1
  fi
  if curl -fsS "$pot_url/ping" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS "$pot_url/ping" >/dev/null; then
  echo "PO Token provider did not become ready" >&2
  exit 1
fi

for _ in $(seq 1 90); do
  if ! kill -0 "$browseforge_pid" 2>/dev/null; then
    echo "BrowseForge exited before becoming ready" >&2
    exit 1
  fi
  if [[ -s "$browseforge_root/data/.api-token" ]] \
    && curl -fsS http://127.0.0.1:19280/api/status >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS http://127.0.0.1:19280/api/status >/dev/null; then
  echo "BrowseForge did not become ready" >&2
  exit 1
fi

uvicorn api:app --host 0.0.0.0 --port 8697 &
api_pid=$!

wait -n "$api_pid" "$browseforge_pid" "$pot_pid" "$xvfb_pid" "$direct_xvfb_pid"
