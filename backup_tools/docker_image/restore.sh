#!/bin/bash
# restore.sh - Restore Docker images from external SSD

BACKUP_DIR="/mnt/ssd_t5/docker_backup"

if [ "$EUID" -ne 0 ]; then
    echo "Error: This script must be run as root or with sudo"
    exit 1
fi

if [ ! -d "$BACKUP_DIR" ]; then
    echo "Error: Backup directory not found: $BACKUP_DIR"
    exit 1
fi

if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running"
    exit 1
fi

if [ -z "$1" ]; then
    echo "Restoring all Docker images..."
    echo ""
    
    tar_files=$(find "$BACKUP_DIR" -name "*.tar")
    
    if [ -z "$tar_files" ]; then
        echo "No backup files found in $BACKUP_DIR"
        exit 1
    fi
    
    total=$(echo "$tar_files" | wc -l)
    current=0
    
    for tar_file in $tar_files; do
        current=$((current + 1))
        echo "[$current/$total] Restoring: $(basename $tar_file)"
        
        docker load -i "$tar_file"
        
        if [ $? -eq 0 ]; then
            echo "  Success"
        else
            echo "  Error: Failed to restore $(basename $tar_file)"
        fi
        echo ""
    done
    
    echo "All images restored"
else
    tar_file="$BACKUP_DIR/$1.tar"
    
    if [ ! -f "$tar_file" ]; then
        echo "Error: Backup file not found: $tar_file"
        echo ""
        echo "Available backups:"
        ls -1 "$BACKUP_DIR"/*.tar 2>/dev/null | xargs -n 1 basename
        exit 1
    fi
    
    echo "Restoring: $1"
    docker load -i "$tar_file"
    
    if [ $? -eq 0 ]; then
        echo "Restore completed"
    else
        echo "Error: Failed to restore"
        exit 1
    fi
fi

echo ""
echo "Current Docker images:"
docker images
