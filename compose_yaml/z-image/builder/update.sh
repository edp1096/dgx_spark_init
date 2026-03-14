#!/bin/bash

# Z-Image UI Hot-Update Script
# Downloads latest UI files from GitHub API (dynamic file list).
#
# Usage (inside container):
#   bash ~/zimage-ui/update.sh              # Update & restart Gradio
#   bash ~/zimage-ui/update.sh --no-restart # Update only
#   bash ~/zimage-ui/update.sh --diff       # Show changes without applying

GITHUB_REPO="edp1096/dgx_spark_init"
GITHUB_BRANCH="main"
GITHUB_RAW="https://raw.githubusercontent.com/${GITHUB_REPO}/${GITHUB_BRANCH}/compose_yaml/z-image/builder"
GITHUB_API="https://api.github.com/repos/${GITHUB_REPO}/contents/compose_yaml/z-image/builder"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UI_DIR="${SCRIPT_DIR}/ui"
BACKUP_DIR="${SCRIPT_DIR}/backup/$(date +%Y%m%d_%H%M%S)"

FAILED_FILES=()

fail() {
    FAILED_FILES+=("$1")
}

list_remote_files() {
    local api_path="$1"
    curl -sf "${GITHUB_API}/${api_path}?ref=${GITHUB_BRANCH}" | python3 -c "
import sys, json
items = json.load(sys.stdin)
for item in items:
    name = item['name']
    if name == '__pycache__':
        continue
    if name.endswith(('.pyc', '.whl')):
        continue
    if item['type'] == 'file':
        print(name)
" 2>/dev/null
}

print_summary() {
    echo ""
    echo "=============================================="
    if [ ${#FAILED_FILES[@]} -eq 0 ]; then
        echo "  Update complete! (no errors)"
    else
        echo "  Update complete with ${#FAILED_FILES[@]} error(s)"
        echo ""
        echo "  FAILED:"
        for f in "${FAILED_FILES[@]}"; do
            echo "    - ${f}"
        done
    fi
    echo "=============================================="
}

NO_RESTART=false
DIFF_ONLY=false
FORCE=false

for arg in "$@"; do
    case $arg in
        --no-restart) NO_RESTART=true ;;
        --diff)       DIFF_ONLY=true ;;
        --force)      FORCE=true ;;
        --help|-h)
            echo "Usage: bash update.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-restart  Update files but don't restart Gradio"
            echo "  --diff        Show diff without applying"
            echo "  --force       Skip backup"
            echo "  -h, --help    Show this help"
            exit 0
            ;;
    esac
done

echo "Fetching file lists from GitHub API..."
UI_FILES=()
while IFS= read -r f; do
    [ -n "$f" ] && UI_FILES+=("$f")
done < <(list_remote_files "ui")

if [ ${#UI_FILES[@]} -eq 0 ]; then
    echo "ERROR: Failed to fetch file list from GitHub API."
    echo "  Check network or API rate limit."
    exit 1
fi

echo "  UI files: ${#UI_FILES[@]} (${UI_FILES[*]})"
echo ""

if [ "$DIFF_ONLY" = true ]; then
    echo "=== Checking for changes ==="
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT

    changed=0
    for fname in "${UI_FILES[@]}"; do
        if ! curl -sfL "${GITHUB_RAW}/ui/${fname}" -o "${TEMP_DIR}/${fname}" 2>/dev/null; then
            fail "ui/${fname} (download)"
            continue
        fi
        current="${SCRIPT_DIR}/${fname}"
        if [ -f "$current" ]; then
            if ! diff -q "$current" "${TEMP_DIR}/${fname}" > /dev/null 2>&1; then
                echo "--- CHANGED: ui/${fname} ---"
                diff --color=auto -u "$current" "${TEMP_DIR}/${fname}" || true
                changed=$((changed + 1))
            fi
        else
            echo "  NEW: ui/${fname}"
            changed=$((changed + 1))
        fi
    done

    if [ $changed -eq 0 ] && [ ${#FAILED_FILES[@]} -eq 0 ]; then
        echo "  No changes detected."
    else
        [ $changed -gt 0 ] && echo "  ${changed} file(s) changed."
    fi
    exit 0
fi

echo "=============================================="
echo "  Z-Image UI Update"
echo "=============================================="

# [1/4] Backup
if [ "$FORCE" = false ] && [ -d "$SCRIPT_DIR" ]; then
    echo "[1/4] Backing up current files..."
    mkdir -p "$BACKUP_DIR"
    for fname in "${UI_FILES[@]}"; do
        if [ -f "${SCRIPT_DIR}/${fname}" ]; then
            cp "${SCRIPT_DIR}/${fname}" "${BACKUP_DIR}/${fname}"
        fi
    done
    echo "  Backup: ${BACKUP_DIR}"
else
    echo "[1/4] Skipping backup (--force)"
fi

# [2/4] Download UI files
echo "[2/4] Downloading UI files..."
mkdir -p "$UI_DIR"
for fname in "${UI_FILES[@]}"; do
    if curl -sfL "${GITHUB_RAW}/ui/${fname}" -o "${UI_DIR}/${fname}"; then
        echo "  OK: ui/${fname}"
    else
        echo "  FAIL: ui/${fname}"
        fail "ui/${fname}"
    fi
done

# Self-update
if curl -sfL "${GITHUB_RAW}/update.sh" -o "${SCRIPT_DIR}/update.sh.new"; then
    mv "${SCRIPT_DIR}/update.sh.new" "${SCRIPT_DIR}/update.sh"
    chmod +x "${SCRIPT_DIR}/update.sh"
    echo "  OK: update.sh (self-updated)"
else
    rm -f "${SCRIPT_DIR}/update.sh.new"
    fail "update.sh (self-update)"
fi

# [3/4] Copy to working directory
echo "[3/4] Applying updates..."
cp "${UI_DIR}"/*.py "${SCRIPT_DIR}/"
echo "  Copied ${#UI_FILES[@]} UI files"

# [4/4] Dependency check
echo ""
echo "[4/4] Checking dependencies..."

if ! python3 -c 'import bitsandbytes' 2>/dev/null; then
    echo "  bitsandbytes not found — building from source (aarch64)..."
    BNB_COMMIT=925d83e
    BNB_SRC="/tmp/bnb-src"
    rm -rf "$BNB_SRC"
    if git clone --depth 250 https://github.com/bitsandbytes-foundation/bitsandbytes.git "$BNB_SRC" && \
       cd "$BNB_SRC" && git checkout "$BNB_COMMIT" && \
       cmake -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY="121" -S . -B build && \
       cmake --build build -j$(nproc) && \
       pip install -q .; then
        echo "  bitsandbytes installed from source"
    else
        echo "  WARNING: bitsandbytes build failed"
        fail "bitsandbytes (source build)"
    fi
    cd "$SCRIPT_DIR"
    rm -rf "$BNB_SRC"
else
    echo "  bitsandbytes: ok"
fi

if ! python3 -c 'import diffusers' 2>/dev/null; then
    echo "  diffusers not found — installing..."
    pip install -q git+https://github.com/huggingface/diffusers
else
    echo "  diffusers: ok"
fi

print_summary

if [ ${#FAILED_FILES[@]} -gt 0 ]; then
    exit 1
fi

# Restart
if [ "$NO_RESTART" = true ]; then
    echo "  Skipping restart (--no-restart)"
else
    GRADIO_PID=$(pgrep -f "python.*app.py" 2>/dev/null || true)
    if [ -n "$GRADIO_PID" ]; then
        echo "  Stopping Gradio (PID: ${GRADIO_PID})..."
        kill "$GRADIO_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$GRADIO_PID" 2>/dev/null || true
        echo "  Starting Gradio..."
        mkdir -p "${SCRIPT_DIR}/logs"
        cd "$SCRIPT_DIR"
        PYTHONUNBUFFERED=1 nohup python app.py --server-name 0.0.0.0 --port 7861 > "${SCRIPT_DIR}/logs/gradio.log" 2>&1 &
        echo "  Gradio restarted (PID: $!)"
    else
        echo "  Gradio not running. Start with:"
        echo "    cd ${SCRIPT_DIR} && PYTHONUNBUFFERED=1 python app.py --server-name 0.0.0.0 --port 7861"
    fi
fi
