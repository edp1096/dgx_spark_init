#!/bin/bash

# ZIFK UI Hot-Update Script
# Downloads latest UI files from GitHub API (dynamic file list).
#
# Usage (inside container):
#   bash ~/zifk-ui/update.sh              # Update & restart Gradio
#   bash ~/zifk-ui/update.sh --no-restart # Update only
#   bash ~/zifk-ui/update.sh --diff       # Show changes without applying

GITHUB_REPO="edp1096/dgx_spark_init"
GITHUB_BRANCH="main"
GITHUB_RAW="https://raw.githubusercontent.com/${GITHUB_REPO}/${GITHUB_BRANCH}/compose_yaml/zifk"
GITHUB_API="https://api.github.com/repos/${GITHUB_REPO}/contents/compose_yaml/zifk"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="${SCRIPT_DIR}/app"
UI_DIR="${APP_DIR}/ui"
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

# Fetch app/ files
APP_FILES=()
while IFS= read -r f; do
    [ -n "$f" ] && APP_FILES+=("$f")
done < <(list_remote_files "app")

# Fetch app/ui/ files
UI_FILES=()
while IFS= read -r f; do
    [ -n "$f" ] && UI_FILES+=("$f")
done < <(list_remote_files "app/ui")

if [ ${#APP_FILES[@]} -eq 0 ] && [ ${#UI_FILES[@]} -eq 0 ]; then
    echo "ERROR: Failed to fetch file list from GitHub API."
    echo "  Check network or API rate limit."
    exit 1
fi

echo "  App files: ${#APP_FILES[@]} (${APP_FILES[*]})"
echo "  UI files:  ${#UI_FILES[@]} (${UI_FILES[*]})"
echo ""

if [ "$DIFF_ONLY" = true ]; then
    echo "=== Checking for changes ==="
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT
    mkdir -p "$TEMP_DIR/ui"

    changed=0
    for fname in "${APP_FILES[@]}"; do
        if ! curl -sfL "${GITHUB_RAW}/app/${fname}" -o "${TEMP_DIR}/${fname}" 2>/dev/null; then
            fail "app/${fname} (download)"
            continue
        fi
        current="${APP_DIR}/${fname}"
        if [ -f "$current" ]; then
            if ! diff -q "$current" "${TEMP_DIR}/${fname}" > /dev/null 2>&1; then
                echo "--- CHANGED: app/${fname} ---"
                diff --color=auto -u "$current" "${TEMP_DIR}/${fname}" || true
                changed=$((changed + 1))
            fi
        else
            echo "  NEW: app/${fname}"
            changed=$((changed + 1))
        fi
    done

    for fname in "${UI_FILES[@]}"; do
        if ! curl -sfL "${GITHUB_RAW}/app/ui/${fname}" -o "${TEMP_DIR}/ui/${fname}" 2>/dev/null; then
            fail "app/ui/${fname} (download)"
            continue
        fi
        current="${UI_DIR}/${fname}"
        if [ -f "$current" ]; then
            if ! diff -q "$current" "${TEMP_DIR}/ui/${fname}" > /dev/null 2>&1; then
                echo "--- CHANGED: app/ui/${fname} ---"
                diff --color=auto -u "$current" "${TEMP_DIR}/ui/${fname}" || true
                changed=$((changed + 1))
            fi
        else
            echo "  NEW: app/ui/${fname}"
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
echo "  ZIFK UI Update"
echo "=============================================="

# [1/3] Backup
if [ "$FORCE" = false ]; then
    echo "[1/3] Backing up current files..."
    mkdir -p "$BACKUP_DIR/ui"
    for fname in "${APP_FILES[@]}"; do
        [ -f "${APP_DIR}/${fname}" ] && cp "${APP_DIR}/${fname}" "${BACKUP_DIR}/${fname}"
    done
    for fname in "${UI_FILES[@]}"; do
        [ -f "${UI_DIR}/${fname}" ] && cp "${UI_DIR}/${fname}" "${BACKUP_DIR}/ui/${fname}"
    done
    echo "  Backup: ${BACKUP_DIR}"
else
    echo "[1/3] Skipping backup (--force)"
fi

# [2/3] Download
echo "[2/3] Downloading files..."
mkdir -p "$APP_DIR" "$UI_DIR"

for fname in "${APP_FILES[@]}"; do
    if curl -sfL "${GITHUB_RAW}/app/${fname}" -o "${APP_DIR}/${fname}"; then
        echo "  OK: app/${fname}"
    else
        echo "  FAIL: app/${fname}"
        fail "app/${fname}"
    fi
done

for fname in "${UI_FILES[@]}"; do
    if curl -sfL "${GITHUB_RAW}/app/ui/${fname}" -o "${UI_DIR}/${fname}"; then
        echo "  OK: app/ui/${fname}"
    else
        echo "  FAIL: app/ui/${fname}"
        fail "app/ui/${fname}"
    fi
done

# Self-update
if curl -sfL "${GITHUB_RAW}/builder/update.sh" -o "${SCRIPT_DIR}/update.sh.new"; then
    mv "${SCRIPT_DIR}/update.sh.new" "${SCRIPT_DIR}/update.sh"
    chmod +x "${SCRIPT_DIR}/update.sh"
    echo "  OK: update.sh (self-updated)"
else
    rm -f "${SCRIPT_DIR}/update.sh.new"
    fail "update.sh (self-update)"
fi

# [3/3] Verify
echo "[3/3] Verifying..."
python3 -c "import zimage; print('  zimage: ok')" 2>/dev/null || echo "  WARNING: zimage not importable"
python3 -c "import flux2; print('  flux2: ok')" 2>/dev/null || echo "  WARNING: flux2 not importable"

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
        PYTHONUNBUFFERED=1 nohup python app/app.py --server-name 0.0.0.0 --port 7861 > "${SCRIPT_DIR}/logs/gradio.log" 2>&1 &
        echo "  Gradio restarted (PID: $!)"
    else
        echo "  Gradio not running. Start with:"
        echo "    cd ${SCRIPT_DIR} && PYTHONUNBUFFERED=1 python app/app.py --server-name 0.0.0.0 --port 7861"
    fi
fi
