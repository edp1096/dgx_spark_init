#!/bin/bash

# ZIT UI Hot-Update Script
# Downloads latest UI files from GitHub API (recursive directory traversal).
#
# Usage (inside container):
#   bash ~/zit-ui/update.sh              # Update & restart Gradio
#   bash ~/zit-ui/update.sh --no-restart # Update only
#   bash ~/zit-ui/update.sh --diff       # Show changes without applying

GITHUB_REPO="edp1096/dgx_spark_init"
GITHUB_BRANCH="main"
GITHUB_RAW="https://raw.githubusercontent.com/${GITHUB_REPO}/${GITHUB_BRANCH}/compose_yaml/zit"
GITHUB_API="https://api.github.com/repos/${GITHUB_REPO}/contents/compose_yaml/zit"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="${SCRIPT_DIR}/app"
UI_DIR="${APP_DIR}/ui"
BACKUP_DIR="${SCRIPT_DIR}/backup/$(date +%Y%m%d_%H%M%S)"

FAILED_FILES=()

fail() {
    FAILED_FILES+=("$1")
}

# List remote files recursively via GitHub API.
# Outputs relative paths (e.g. "depth.py", "zoe/__init__.py", "zoe/zoedepth/models/builder.py").
list_remote_files_recursive() {
    local api_path="$1"
    local prefix="$2"  # relative prefix for output paths

    curl -sf "${GITHUB_API}/${api_path}?ref=${GITHUB_BRANCH}" | python3 -c "
import sys, json
items = json.load(sys.stdin)
for item in items:
    name = item['name']
    if name == '__pycache__':
        continue
    if name.endswith(('.pyc', '.whl')):
        continue
    prefix = '${prefix}'
    rel = (prefix + '/' + name) if prefix else name
    if item['type'] == 'file':
        print('FILE ' + rel)
    elif item['type'] == 'dir':
        print('DIR ' + rel)
" 2>/dev/null
}

# Collect all files under an API path recursively.
# Results stored in the global ALL_FILES array.
ALL_FILES=()
collect_files() {
    local api_path="$1"
    local prefix="$2"

    while IFS= read -r line; do
        [ -z "$line" ] && continue
        local kind="${line%% *}"
        local rel="${line#* }"
        if [ "$kind" = "FILE" ]; then
            ALL_FILES+=("$rel")
        elif [ "$kind" = "DIR" ]; then
            # Recurse into subdirectory
            if [ -n "$prefix" ]; then
                collect_files "${api_path%/}/${rel#${prefix}/}" "$rel"
            else
                collect_files "${api_path}/${rel}" "$rel"
            fi
        fi
    done < <(list_remote_files_recursive "$api_path" "$prefix")
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

echo "Fetching file lists from GitHub API (recursive)..."

# Fetch app/ files (flat — app.py only)
ALL_FILES=()
collect_files "app" ""
APP_FILES=("${ALL_FILES[@]}")

# Fetch app/ui/ files (recursive — includes videox_models/, preprocessors/zoe/ etc.)
ALL_FILES=()
collect_files "app/ui" ""
UI_FILES=("${ALL_FILES[@]}")

# Fetch tests/ files
ALL_FILES=()
collect_files "tests" ""
TEST_FILES=("${ALL_FILES[@]}")

if [ ${#APP_FILES[@]} -eq 0 ] && [ ${#UI_FILES[@]} -eq 0 ]; then
    echo "ERROR: Failed to fetch file list from GitHub API."
    echo "  Check network or API rate limit."
    exit 1
fi

echo "  App files:  ${#APP_FILES[@]}"
echo "  UI files:   ${#UI_FILES[@]}"
echo "  Test files: ${#TEST_FILES[@]}"
echo ""

# Helper: ensure parent directory exists for a relative path
ensure_parent() {
    local dir
    dir="$(dirname "$1")"
    [ "$dir" != "." ] && mkdir -p "$dir"
}

if [ "$DIFF_ONLY" = true ]; then
    echo "=== Checking for changes ==="
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT

    changed=0
    for fname in "${APP_FILES[@]}"; do
        mkdir -p "${TEMP_DIR}/app/$(dirname "$fname")"
        if ! curl -sfL "${GITHUB_RAW}/app/${fname}" -o "${TEMP_DIR}/app/${fname}" 2>/dev/null; then
            fail "app/${fname} (download)"
            continue
        fi
        current="${APP_DIR}/${fname}"
        if [ -f "$current" ]; then
            if ! diff -q "$current" "${TEMP_DIR}/app/${fname}" > /dev/null 2>&1; then
                echo "--- CHANGED: app/${fname} ---"
                diff --color=auto -u "$current" "${TEMP_DIR}/app/${fname}" || true
                changed=$((changed + 1))
            fi
        else
            echo "  NEW: app/${fname}"
            changed=$((changed + 1))
        fi
    done

    for fname in "${UI_FILES[@]}"; do
        mkdir -p "${TEMP_DIR}/ui/$(dirname "$fname")"
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

    for fname in "${TEST_FILES[@]}"; do
        mkdir -p "${TEMP_DIR}/tests/$(dirname "$fname")"
        if ! curl -sfL "${GITHUB_RAW}/tests/${fname}" -o "${TEMP_DIR}/tests/${fname}" 2>/dev/null; then
            fail "tests/${fname} (download)"
            continue
        fi
        current="${SCRIPT_DIR}/tests/${fname}"
        if [ -f "$current" ]; then
            if ! diff -q "$current" "${TEMP_DIR}/tests/${fname}" > /dev/null 2>&1; then
                echo "--- CHANGED: tests/${fname} ---"
                diff --color=auto -u "$current" "${TEMP_DIR}/tests/${fname}" || true
                changed=$((changed + 1))
            fi
        else
            echo "  NEW: tests/${fname}"
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
echo "  ZIT UI Update"
echo "=============================================="

# [1/3] Backup
if [ "$FORCE" = false ]; then
    echo "[1/3] Backing up current files..."
    mkdir -p "$BACKUP_DIR"
    for fname in "${APP_FILES[@]}"; do
        if [ -f "${APP_DIR}/${fname}" ]; then
            mkdir -p "${BACKUP_DIR}/app/$(dirname "$fname")"
            cp "${APP_DIR}/${fname}" "${BACKUP_DIR}/app/${fname}"
        fi
    done
    for fname in "${UI_FILES[@]}"; do
        if [ -f "${UI_DIR}/${fname}" ]; then
            mkdir -p "${BACKUP_DIR}/ui/$(dirname "$fname")"
            cp "${UI_DIR}/${fname}" "${BACKUP_DIR}/ui/${fname}"
        fi
    done
    for fname in "${TEST_FILES[@]}"; do
        if [ -f "${SCRIPT_DIR}/tests/${fname}" ]; then
            mkdir -p "${BACKUP_DIR}/tests/$(dirname "$fname")"
            cp "${SCRIPT_DIR}/tests/${fname}" "${BACKUP_DIR}/tests/${fname}"
        fi
    done
    echo "  Backup: ${BACKUP_DIR}"
else
    echo "[1/3] Skipping backup (--force)"
fi

# [2/3] Download
echo "[2/3] Downloading files..."

for fname in "${APP_FILES[@]}"; do
    mkdir -p "${APP_DIR}/$(dirname "$fname")"
    if curl -sfL "${GITHUB_RAW}/app/${fname}" -o "${APP_DIR}/${fname}"; then
        echo "  OK: app/${fname}"
    else
        echo "  FAIL: app/${fname}"
        fail "app/${fname}"
    fi
done

for fname in "${UI_FILES[@]}"; do
    mkdir -p "${UI_DIR}/$(dirname "$fname")"
    if curl -sfL "${GITHUB_RAW}/app/ui/${fname}" -o "${UI_DIR}/${fname}"; then
        echo "  OK: app/ui/${fname}"
    else
        echo "  FAIL: app/ui/${fname}"
        fail "app/ui/${fname}"
    fi
done

TESTS_DIR="${SCRIPT_DIR}/tests"
mkdir -p "$TESTS_DIR"
for fname in "${TEST_FILES[@]}"; do
    mkdir -p "${TESTS_DIR}/$(dirname "$fname")"
    if curl -sfL "${GITHUB_RAW}/tests/${fname}" -o "${TESTS_DIR}/${fname}"; then
        echo "  OK: tests/${fname}"
    else
        echo "  FAIL: tests/${fname}"
        fail "tests/${fname}"
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
        PYTHONUNBUFFERED=1 nohup python app/app.py --server-name 0.0.0.0 --port 7862 > "${SCRIPT_DIR}/logs/gradio.log" 2>&1 &
        echo "  Gradio restarted (PID: $!)"
    else
        echo "  Gradio not running. Start with:"
        echo "    cd ${SCRIPT_DIR} && PYTHONUNBUFFERED=1 python app/app.py --server-name 0.0.0.0 --port 7862"
    fi
fi
