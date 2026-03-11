#!/bin/bash

# LTX-2.3 UI Hot-Update Script
# Downloads latest UI & patch files from GitHub API (dynamic file list).
#
# Usage (inside container):
#   bash ~/ltx2-ui/update.sh              # Update & restart Gradio
#   bash ~/ltx2-ui/update.sh --no-restart # Update only, don't restart
#   bash ~/ltx2-ui/update.sh --diff       # Show changes without applying
#
# First-time bootstrap (downloads this script itself):
#   curl -sfL https://raw.githubusercontent.com/edp1096/dgx_spark_init/main/compose_yaml/ltx-2.3/builder/update.sh -o ~/ltx2-ui/update.sh && bash ~/ltx2-ui/update.sh

GITHUB_REPO="edp1096/dgx_spark_init"
GITHUB_BRANCH="main"
GITHUB_RAW="https://raw.githubusercontent.com/${GITHUB_REPO}/${GITHUB_BRANCH}/compose_yaml/ltx-2.3/builder"
GITHUB_API="https://api.github.com/repos/${GITHUB_REPO}/contents/compose_yaml/ltx-2.3/builder"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LTX_DIR="${SCRIPT_DIR}/LTX-2"
UI_DIR="${SCRIPT_DIR}/ui"
PATCH_DIR="${SCRIPT_DIR}/patches"
BACKUP_DIR="${SCRIPT_DIR}/backup/$(date +%Y%m%d_%H%M%S)"

# Failure tracking
FAILED_FILES=()

fail() {
    FAILED_FILES+=("$1")
}

# ---------------------------------------------------------------------------
# GitHub API → file list (excludes __pycache__, .whl, .pyc)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Print summary — called at the end of update or diff
# ---------------------------------------------------------------------------
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
        echo ""
        echo "  Check network or GitHub URL."
    fi
    echo "=============================================="
}

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
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
            echo "  --diff        Download to temp and show diff, don't apply"
            echo "  --force       Skip backup, overwrite directly"
            echo "  -h, --help    Show this help"
            exit 0
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Fetch file lists from GitHub API
# ---------------------------------------------------------------------------
echo "Fetching file lists from GitHub API..."
UI_FILES=()
while IFS= read -r f; do
    [ -n "$f" ] && UI_FILES+=("$f")
done < <(list_remote_files "ui")

PATCH_FILES=()
while IFS= read -r f; do
    [ -n "$f" ] && PATCH_FILES+=("$f")
done < <(list_remote_files "patches")

if [ ${#UI_FILES[@]} -eq 0 ]; then
    echo "ERROR: Failed to fetch file list from GitHub API."
    echo "  Check network or API rate limit (60 req/hr unauthenticated)."
    echo ""
    echo "  Hint: set GITHUB_TOKEN env var to increase rate limit:"
    echo "    export GITHUB_TOKEN=ghp_xxxxx"
    exit 1
fi

echo "  UI files: ${#UI_FILES[@]} (${UI_FILES[*]})"
echo "  Patches:  ${#PATCH_FILES[@]} (${PATCH_FILES[*]})"
echo ""

# ---------------------------------------------------------------------------
# Diff mode — download to temp, compare, exit
# ---------------------------------------------------------------------------
if [ "$DIFF_ONLY" = true ]; then
    echo "=== Checking for changes ==="
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT

    changed=0

    # Check UI files
    for fname in "${UI_FILES[@]}"; do
        if ! curl -sfL "${GITHUB_RAW}/ui/${fname}" -o "${TEMP_DIR}/${fname}" 2>/dev/null; then
            fail "ui/${fname} (download)"
            continue
        fi
        current="${LTX_DIR}/${fname}"
        if [ -f "$current" ]; then
            if ! diff -q "$current" "${TEMP_DIR}/${fname}" > /dev/null 2>&1; then
                echo ""
                echo "--- CHANGED: ui/${fname} ---"
                diff --color=auto -u "$current" "${TEMP_DIR}/${fname}" || true
                changed=$((changed + 1))
            fi
        else
            echo "  NEW: ui/${fname}"
            changed=$((changed + 1))
        fi
    done

    # Check patch files
    for fname in "${PATCH_FILES[@]}"; do
        if ! curl -sfL "${GITHUB_RAW}/patches/${fname}" -o "${TEMP_DIR}/${fname}" 2>/dev/null; then
            fail "patches/${fname} (download)"
            continue
        fi
        current="${PATCH_DIR}/${fname}"
        if [ -f "$current" ]; then
            if ! diff -q "$current" "${TEMP_DIR}/${fname}" > /dev/null 2>&1; then
                echo ""
                echo "--- CHANGED: patches/${fname} ---"
                diff --color=auto -u "$current" "${TEMP_DIR}/${fname}" || true
                changed=$((changed + 1))
            fi
        else
            echo "  NEW: patches/${fname}"
            changed=$((changed + 1))
        fi
    done

    # Check update.sh itself
    if curl -sfL "${GITHUB_RAW}/update.sh" -o "${TEMP_DIR}/update.sh" 2>/dev/null; then
        if [ -f "${SCRIPT_DIR}/update.sh" ]; then
            if ! diff -q "${SCRIPT_DIR}/update.sh" "${TEMP_DIR}/update.sh" > /dev/null 2>&1; then
                echo ""
                echo "--- CHANGED: update.sh ---"
                diff --color=auto -u "${SCRIPT_DIR}/update.sh" "${TEMP_DIR}/update.sh" || true
                changed=$((changed + 1))
            fi
        fi
    fi

    if [ $changed -eq 0 ] && [ ${#FAILED_FILES[@]} -eq 0 ]; then
        echo "  No changes detected."
    else
        [ $changed -gt 0 ] && echo "" && echo "  ${changed} file(s) changed. Run without --diff to apply."
    fi

    if [ ${#FAILED_FILES[@]} -gt 0 ]; then
        print_summary
        exit 1
    fi
    exit 0
fi

# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------
echo "=============================================="
echo "  LTX-2.3 UI Update"
echo "=============================================="
echo "  Source: ${GITHUB_RAW}"
echo "  Target: ${LTX_DIR}"
echo ""

# [1/4] Backup current files (unless --force)
if [ "$FORCE" = false ] && [ -d "$LTX_DIR" ]; then
    echo "[1/4] Backing up current files..."
    mkdir -p "$BACKUP_DIR"
    for fname in "${UI_FILES[@]}"; do
        if [ -f "${LTX_DIR}/${fname}" ]; then
            cp "${LTX_DIR}/${fname}" "${BACKUP_DIR}/${fname}"
        fi
    done
    for fname in "${PATCH_FILES[@]}"; do
        if [ -f "${PATCH_DIR}/${fname}" ]; then
            mkdir -p "${BACKUP_DIR}/patches"
            cp "${PATCH_DIR}/${fname}" "${BACKUP_DIR}/patches/${fname}"
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

# [3/4] Download patches
echo "[3/4] Downloading patches..."
mkdir -p "$PATCH_DIR"
for fname in "${PATCH_FILES[@]}"; do
    if curl -sfL "${GITHUB_RAW}/patches/${fname}" -o "${PATCH_DIR}/${fname}"; then
        echo "  OK: patches/${fname}"
    else
        echo "  FAIL: patches/${fname}"
        fail "patches/${fname}"
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

# [4/4] Copy to LTX-2 directory
echo "[4/4] Applying updates..."
if [ -d "$LTX_DIR" ]; then
    cp "${UI_DIR}"/*.py "${LTX_DIR}/"
    echo "  Copied ${#UI_FILES[@]} UI files to LTX-2/"

    # Re-apply patches if needed
    cd "$LTX_DIR"
    for pf in "${PATCH_FILES[@]}"; do
        if [ -f "${PATCH_DIR}/${pf}" ]; then
            if git apply --check "${PATCH_DIR}/${pf}" 2>/dev/null; then
                git apply "${PATCH_DIR}/${pf}"
                echo "  Patch applied: ${pf}"
            else
                echo "  Patch skipped (already applied): ${pf}"
            fi
        fi
    done
    cd "$SCRIPT_DIR"
else
    echo "  WARNING: LTX-2 directory not found: ${LTX_DIR}"
    echo "  Files downloaded to ${UI_DIR} but not deployed."
    fail "LTX-2 directory not found"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print_summary

# Exit early if there were failures
if [ ${#FAILED_FILES[@]} -gt 0 ]; then
    exit 1
fi

# ---------------------------------------------------------------------------
# Restart Gradio
# ---------------------------------------------------------------------------
if [ "$NO_RESTART" = true ]; then
    echo "  Skipping restart (--no-restart)"
    echo "  To restart manually: pkill -f 'python app.py'; cd ${LTX_DIR} && python app.py --server-name 0.0.0.0 &"
else
    GRADIO_PID=$(pgrep -f "python.*app.py" 2>/dev/null || true)
    if [ -n "$GRADIO_PID" ]; then
        echo "  Stopping Gradio (PID: ${GRADIO_PID})..."
        kill "$GRADIO_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$GRADIO_PID" 2>/dev/null || true
        echo "  Starting Gradio..."
        mkdir -p "${SCRIPT_DIR}/logs"
        cd "$LTX_DIR"
        PYTHONUNBUFFERED=1 nohup python app.py --server-name 0.0.0.0 > "${SCRIPT_DIR}/logs/gradio.log" 2>&1 &
        NEW_PID=$!
        echo "  Gradio restarted (PID: ${NEW_PID})"
        echo "  Log: ${SCRIPT_DIR}/logs/gradio.log"
    else
        echo "  Gradio not running. Start with:"
        echo "    cd ${LTX_DIR} && PYTHONUNBUFFERED=1 python app.py --server-name 0.0.0.0"
    fi
fi
