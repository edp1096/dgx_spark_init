#!/bin/bash
# list.sh - List available Docker backups

BACKUP_DIR="/mnt/ssd_t5/docker_backup"

if [ ! -d "$BACKUP_DIR" ]; then
    echo "Error: Backup directory not found: $BACKUP_DIR"
    exit 1
fi

echo "Available Docker image backups:"
echo ""

for tar_file in "$BACKUP_DIR"/*.tar; do
    if [ -f "$tar_file" ]; then
        filename=$(basename "$tar_file")
        size=$(du -sh "$tar_file" | cut -f1)
        echo "  $filename ($size)"
    fi
done
