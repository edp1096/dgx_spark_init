#!/bin/bash
# restore.sh - Selective or full Docker image restore from external SSD

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

show_backups() {
    echo "Available backup files:"
    printf "%-6s %-50s %s\n" "NUM" "IMAGE NAME" "SIZE"
    printf "%-6s %-50s %s\n" "---" "----------" "----"
    find "$BACKUP_DIR" -name "*.tar" | sort | nl | while read num file; do
        size=$(du -sh "$file" | cut -f1)
        printf "%-6s %-50s %s\n" "$num" "$(basename $file .tar)" "$size"
    done
    echo ""
}

restore_image() {
    local tar_file="$1"
    
    echo "Restoring: $(basename $tar_file .tar)"
    echo "  File: $(basename $tar_file)"
    
    docker load -i "$tar_file"
    
    if [ $? -eq 0 ]; then
        echo "  Success"
    else
        echo "  Error: Failed to restore $(basename $tar_file)"
        return 1
    fi
    echo ""
}

if [ -z "$1" ]; then
    tar_files=$(find "$BACKUP_DIR" -name "*.tar" | sort)
    
    if [ -z "$tar_files" ]; then
        echo "No backup files found in $BACKUP_DIR"
        exit 1
    fi
    
    show_backups
    
    read -p "Enter backup number(s) to restore (comma-separated, or 'all' for all): " selection
    
    if [ "$selection" = "all" ]; then
        echo ""
        echo "Restoring all images..."
        echo ""
        
        total=$(echo "$tar_files" | wc -l)
        current=0
        
        while IFS= read -r tar_file; do
            current=$((current + 1))
            echo "[$current/$total]"
            restore_image "$tar_file"
        done <<< "$tar_files"
    else
        echo ""
        IFS=',' read -ra NUMS <<< "$selection"
        
        for num in "${NUMS[@]}"; do
            num=$(echo "$num" | xargs)
            tar_file=$(echo "$tar_files" | sed -n "${num}p")
            
            if [ -n "$tar_file" ]; then
                restore_image "$tar_file"
            else
                echo "Invalid selection: $num"
            fi
        done
    fi
else
    if [ "$1" = "all" ]; then
        tar_files=$(find "$BACKUP_DIR" -name "*.tar" | sort)
        
        if [ -z "$tar_files" ]; then
            echo "No backup files found in $BACKUP_DIR"
            exit 1
        fi
        
        total=$(echo "$tar_files" | wc -l)
        current=0
        
        echo "Restoring all images..."
        echo ""
        
        while IFS= read -r tar_file; do
            current=$((current + 1))
            echo "[$current/$total]"
            restore_image "$tar_file"
        done <<< "$tar_files"
    else
        tar_file="$BACKUP_DIR/$1.tar"
        
        if [ ! -f "$tar_file" ]; then
            echo "Error: Backup file not found: $tar_file"
            echo ""
            echo "Available backups:"
            find "$BACKUP_DIR" -name "*.tar" | sort | xargs -n 1 basename
            exit 1
        fi
        
        restore_image "$tar_file"
    fi
fi

echo "Restore completed!"
echo ""
echo "Current Docker images:"
docker images