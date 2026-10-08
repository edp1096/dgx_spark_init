#!/bin/sh
set -eu
runtime="${MEDIA_DATA_DIR:-/data}/runtimes/yt-dlp/current"
if [ -x "$runtime/venv/bin/yt-dlp" ]; then
    exec "$runtime/venv/bin/yt-dlp" "$@"
fi
exec /usr/local/bin/yt-dlp-pinned "$@"
