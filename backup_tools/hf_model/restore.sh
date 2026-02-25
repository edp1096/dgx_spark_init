#!/bin/bash
# restore.sh - Selective or full restore with skip-existing logic

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

show_backups() {
    echo "Available backup files:"
    printf "%-6s %-70s %s\n" "NUM" "MODEL NAME" "SIZE"
    printf "%-6s %-70s %s\n" "---" "----------" "----"
    local i=1
    for tar_file in "$BACKUP_DIR"/models--*.tar; do
        [ -f "$tar_file" ] || continue
        local size=$(du -sh "$tar_file" | cut -f1)
        printf "%-6s %-70s %s\n" "$i" "$(basename "$tar_file" .tar)" "$size"
        i=$((i + 1))
    done
    echo ""
}

run_all() {
    local tar_files=("$BACKUP_DIR"/models--*.tar)
    local tar_count=${#tar_files[@]}
    echo "Found $tar_count backup files"

    local current=0
    for tar_file in "${tar_files[@]}"; do
        if [ -f "$tar_file" ]; then
            current=$((current + 1))
            echo "[$current/$tar_count] Processing: $(basename "$tar_file")"
            do_restore "$tar_file"
            echo ""
        fi
    done
}

if [ -n "$1" ]; then
    if [ "$1" = "all" ]; then
        run_all
    else
        tar_name=$(basename "$1" .tar)
        do_restore "$BACKUP_DIR/${tar_name}.tar"
    fi
else
    tar_files=("$BACKUP_DIR"/models--*.tar)
    if [ ! -f "${tar_files[0]}" ]; then
        echo "No backup files found in $BACKUP_DIR"
        exit 1
    fi

    show_backups

    read -p "Enter backup number(s) to restore (comma-separated, or 'all' for all): " selection

    if [ "$selection" = "all" ]; then
        echo ""
        run_all
    else
        echo ""
        # Build indexed array of tar files for selection
        mapfile -t indexed_files < <(find "$BACKUP_DIR" -name "models--*.tar" | sort)

        IFS=',' read -ra NUMS <<< "$selection"
        for num in "${NUMS[@]}"; do
            num=$(echo "$num" | xargs)
            idx=$((num - 1))

            if [ "$idx" -ge 0 ] && [ "$idx" -lt "${#indexed_files[@]}" ]; then
                tar_file="${indexed_files[$idx]}"
                echo "Processing: $(basename "$tar_file")"
                do_restore "$tar_file"
                echo ""
            else
                echo "Invalid selection: $num"
            fi
        done
    fi
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