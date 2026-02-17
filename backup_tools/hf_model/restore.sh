#!/bin/bash
# restore.sh - Restore with skip-existing logic

SOURCE_DIR="/home/edp1096/workspace/hf_models/hub"
BACKUP_DIR="/mnt/ssd_t5/huggingface_backup"
SKIPPED_ITEMS=()

if [ "$EUID" -ne 0 ]; then
    echo "Error: This script must be run as root or with sudo"
    exit 1
fi

if [ ! -d "$BACKUP_DIR" ]; then
    echo "Error: Backup directory not found: $BACKUP_DIR"
    exit 1
fi

mkdir -p "$SOURCE_DIR"

do_restore() {
    local tar_file=$1
    local model_name=$(basename "$tar_file" .tar)
    
    if [ -d "$SOURCE_DIR/$model_name" ]; then
        echo "  [SKIP] Model directory already exists: $model_name"
        SKIPPED_ITEMS+=("$model_name")
        return
    fi
    
    echo "  [RUN] Restoring: $model_name"
    tar -xf "$tar_file" -C "$SOURCE_DIR"
    if [ $? -eq 0 ]; then
        echo "  Success"
    else
        echo "  Error: Failed to restore $model_name"
    fi
}

if [ -n "$1" ]; then
    tar_name=$(basename "$1" .tar)
    do_restore "$BACKUP_DIR/${tar_name}.tar"
else
    tar_files=("$BACKUP_DIR"/models--*.tar)
    tar_count=${#tar_files[@]}
    echo "Found $tar_count backup files"
    
    current=0
    for tar_file in "${tar_files[@]}"; do
        if [ -f "$tar_file" ]; then
            current=$((current + 1))
            echo "[$current/$tar_count] Processing: $(basename "$tar_file")"
            do_restore "$tar_file"
            echo ""
        fi
    done
fi

echo "------------------------------------------"
echo "Restore Process Finished."
if [ ${#SKIPPED_ITEMS[@]} -ne 0 ]; then
    echo "Skipped items (already exist in source):"
    for item in "${SKIPPED_ITEMS[@]}"; do
        echo " - $item"
    done
else
    echo "No items were skipped."
fi
