#!/bin/bash
# backup.sh - Backup with skip-existing logic

SOURCE_DIR="/home/edp1096/workspace/models/hub"
BACKUP_DIR="/mnt/ssd_t5/huggingface_backup"
SKIPPED_ITEMS=()

if [ "$EUID" -ne 0 ]; then
    echo "Error: This script must be run as root or with sudo"
    exit 1
fi

if [ ! -d "/mnt/ssd_t5" ] || ! mountpoint -q "/mnt/ssd_t5"; then
    echo "Error: /mnt/ssd_t5 is not mounted or not accessible"
    exit 1
fi

mkdir -p "$BACKUP_DIR"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory not found: $SOURCE_DIR"
    exit 1
fi

do_backup() {
    local model_name=$1
    local tar_path="$BACKUP_DIR/${model_name}.tar"
    
    if [ -f "$tar_path" ]; then
        echo "  [SKIP] Backup already exists: ${model_name}.tar"
        SKIPPED_ITEMS+=("$model_name")
        return
    fi
    
    echo "  [RUN] Backing up: $model_name"
    tar -cf "$tar_path" -C "$SOURCE_DIR" "$model_name"
    if [ $? -eq 0 ]; then
        echo "  Created: ${model_name}.tar"
    else
        echo "  Error: Failed to backup $model_name"
    fi
}

if [ -n "$1" ]; then
    model_name=$(basename "$1")
    do_backup "$model_name"
else
    model_dirs=("$SOURCE_DIR"/models--*)
    model_count=${#model_dirs[@]}
    echo "Found $model_count models to backup"
    
    current=0
    for model_dir in "${model_dirs[@]}"; do
        if [ -d "$model_dir" ]; then
            current=$((current + 1))
            model_name=$(basename "$model_dir")
            echo "[$current/$model_count] Processing: $model_name"
            do_backup "$model_name"
            echo ""
        fi
    done
fi

echo "------------------------------------------"
echo "Backup Process Finished."
if [ ${#SKIPPED_ITEMS[@]} -ne 0 ]; then
    echo "Skipped items (already exist):"
    for item in "${SKIPPED_ITEMS[@]}"; do
        echo " - $item"
    done
else
    echo "No items were skipped."
fi
